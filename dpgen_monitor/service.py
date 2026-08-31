from __future__ import annotations

from pathlib import Path
import threading
import time
import traceback

from .config import MonitorConfig
from .dpgen import DpgenObserver, IterationSnapshot
from .evaluation import DeepMDEvaluator, EvaluationResult
from .evaluation_data import (
    current_iteration_fp_evaluation_description,
    evaluation_data_description,
    resolve_current_iteration_fp_data,
    resolve_evaluation_data,
)
from .evaluation_plots import best_force_model, load_force_parities
from .events import MonitorEvent
from .notifiers import Notifier, build_notifier
from .render import format_percentage, render_evaluation, render_statistics_trend
from .state import StateStore


class MonitorService:
    def __init__(
        self,
        config: MonitorConfig,
        *,
        enable_notifiers: bool = True,
    ):
        self.config = config
        self.config.project.output_dir.mkdir(parents=True, exist_ok=True)
        self.state = StateStore(self.config.project.output_dir / "monitor.sqlite3")
        self.observer = DpgenObserver(self.config.project.run_dir, self.config.dpgen)
        self.notifiers: list[Notifier] = [
            build_notifier(item)
            for item in self.config.notifications
            if enable_notifiers and item.enabled
        ]
        self.evaluator = (
            DeepMDEvaluator(
                self.config.evaluation,
                self.config.project.output_dir,
                self.state,
            )
            if self.config.evaluation.enabled
            else None
        )
        self._last_status: dict[str, str] = {}

    def close(self) -> None:
        self.state.close()

    def run(self, stop_event: threading.Event) -> None:
        interval = self.config.project.check_interval
        heartbeat_interval = self.config.project.heartbeat_interval
        print(
            f"[监控] 已启动；扫描间隔 {interval}s；"
            + (
                f"心跳间隔 {heartbeat_interval}s"
                if heartbeat_interval
                else "心跳已关闭"
            )
        )
        last_heartbeat = time.monotonic()
        while not stop_event.is_set():
            try:
                summary = self.scan_once(evaluate=True, notify=True)
                now = time.monotonic()
                if heartbeat_interval and now - last_heartbeat >= heartbeat_interval:
                    print(
                        "[监控心跳] 运行正常；"
                        f"最新发现 {summary['iterations']} 个迭代，"
                        f"完整探索统计 {summary['statistics_ready']} 轮"
                    )
                    last_heartbeat = now
            except Exception as exc:
                print(f"[监控][错误] 扫描失败: {exc}")
                traceback.print_exc()
            if stop_event.wait(interval):
                break
        print("[监控] 已安全停止")

    def scan_once(self, evaluate: bool, notify: bool) -> dict[str, int]:
        snapshots, inspections = self.observer.scan()
        for snapshot in snapshots:
            self.state.upsert_stage(snapshot.iteration, snapshot.task)

        ready_stats = 0
        for iteration, inspection in sorted(inspections.items()):
            if iteration < 0 or inspection.status != "ready" or not inspection.stats:
                continue
            self.state.upsert_statistics(iteration, inspection.stats)
            ready_stats += 1
            if iteration < self.config.dpgen.statistics_start_iteration:
                continue
            event = self._statistics_event(iteration, inspection.stats)
            self._deliver(event, notify)

        eligible_pending = [
            inspection
            for iteration, inspection in inspections.items()
            if iteration >= self.config.dpgen.statistics_start_iteration
            and inspection.status != "ready"
        ]
        if eligible_pending:
            latest = max(eligible_pending, key=lambda item: item.iteration)
            self._log_changed(
                "statistics-pending",
                f"[统计监控] {latest.message()}",
            )

        evaluations_complete = 0
        if evaluate and self.evaluator:
            for snapshot in snapshots:
                if snapshot.iteration < self.config.evaluation.start_iteration:
                    continue
                phases = [("absorption", self.config.evaluation.absorption_ready_task)]
                if self.config.evaluation.blind_spot_enabled:
                    phases.append(
                        ("blind_spot", self.config.evaluation.blind_spot_ready_task)
                    )
                for phase, required_task in phases:
                    event_key = self._evaluation_event_key(
                        snapshot.iteration, phase
                    )
                    if not self._has_pending_delivery(event_key):
                        continue
                    ready, reason = self._phase_readiness(
                        snapshot, phase, required_task
                    )
                    status_key = f"evaluation:{phase}:iter.{snapshot.iteration:06d}"
                    if not ready:
                        label = "吸收评估" if phase == "absorption" else "盲区评估"
                        self._log_changed(
                            status_key,
                            f"[模型监控][{label}] "
                            f"iter.{snapshot.iteration:06d} {reason}",
                        )
                        continue

                    self._log_changed(
                        status_key,
                        f"[模型监控] iter.{snapshot.iteration:06d} "
                        f"{self._phase_label(phase)}已就绪，开始评估",
                    )
                    results = self.evaluator.evaluate_iteration(snapshot, phase)
                    if not results:
                        continue
                    complete = [item for item in results if item.status == "complete"]
                    evaluations_complete += len(complete)
                    failures = [item for item in results if item.status == "failed"]
                    for failure in failures:
                        self._deliver(
                            self._evaluation_error_event(snapshot, failure), notify
                        )
                    if len(complete) == len(self.config.evaluation.model_ids):
                        self._deliver(
                            self._evaluation_event(snapshot, phase, results), notify
                        )
                    else:
                        waiting = [
                            item.model_id
                            for item in results
                            if item.status == "waiting"
                        ]
                        self._log_changed(
                            status_key,
                            f"[模型监控] iter.{snapshot.iteration:06d} "
                            f"{self._phase_label(phase)}完成 "
                            f"{len(complete)}/{len(self.config.evaluation.model_ids)}"
                            + (f"；等待模型 {waiting}" if waiting else ""),
                        )

        summary = {
            "iterations": len(snapshots),
            "statistics_ready": ready_stats,
            "evaluations_complete": evaluations_complete,
        }
        self._log_changed(
            "scan-summary",
            "[监控总览] "
            f"发现迭代 {summary['iterations']}；"
            f"完整探索统计 {summary['statistics_ready']} 轮；"
            f"本轮新增模型结果 {summary['evaluations_complete']} 个",
        )
        return summary

    def _phase_readiness(
        self,
        snapshot: IterationSnapshot,
        phase: str,
        required_task: int,
    ) -> tuple[bool, str]:
        if snapshot.train_dir is None:
            return False, "等待训练目录生成"
        if snapshot.task is None or snapshot.task < required_task:
            current = "未知" if snapshot.task is None else f"task {snapshot.task:02d}"
            return False, f"当前 {current}；等待 task {required_task:02d}"
        if phase == "blind_spot":
            data_path = resolve_current_iteration_fp_data(
                snapshot.train_dir, snapshot.iteration
            )
        else:
            data_path = resolve_evaluation_data(
                snapshot.train_dir,
                snapshot.iteration,
                self.config.evaluation.test_data,
                self.config.evaluation.initial_test_data,
            )
        if not data_path.is_dir():
            return False, f"等待评估数据目录 {data_path}"
        return True, "评估条件已满足"

    @staticmethod
    def _phase_label(phase: str) -> str:
        return "上一轮 FP 吸收评估" if phase == "absorption" else "本轮 FP 盲区评估"

    @staticmethod
    def _evaluation_event_key(iteration: int, phase: str) -> str:
        return f"evaluation:{phase}:iter.{iteration:06d}"

    def _log_changed(self, key: str, message: str) -> None:
        if self._last_status.get(key) == message:
            return
        self._last_status[key] = message
        print(message)

    def _statistics_event(self, iteration: int, stats: dict) -> MonitorEvent:
        event_key = f"statistics:iter.{iteration:06d}"
        image_paths: tuple[Path, ...] = ()
        if self._has_pending_delivery(event_key):
            image = render_statistics_trend(
                self.state.list_statistics(),
                self.config.project.output_dir / "artifacts" / "dpgen_stats_trend.png",
            )
            image_paths = (image,)
        message = (
            f"candidate {stats['candidate_count']}/{stats['candidate_total']} "
            f"({format_percentage(stats['candidate_percent'])}), "
            f"failed {stats['failed_count']}/{stats['failed_total']} "
            f"({format_percentage(stats['failed_percent'])}), "
            f"accurate {format_percentage(stats['accurate_percent'])}"
        )
        return MonitorEvent(
            key=event_key,
            event_type="exploration_stats_ready",
            title=f"DP-GEN iter.{iteration:06d} 探索统计",
            message=message,
            iteration=iteration,
            image_paths=image_paths,
            payload=dict(stats),
        )

    def _evaluation_event(
        self,
        snapshot: IterationSnapshot,
        phase: str,
        results: list[EvaluationResult],
    ) -> MonitorEvent:
        event_key = self._evaluation_event_key(snapshot.iteration, phase)
        images: tuple[Path, ...] = ()
        if self._has_pending_delivery(event_key):
            images = render_evaluation(
                snapshot.iteration,
                results,
                self.config.project.output_dir
                / "artifacts"
                / f"iter.{snapshot.iteration:06d}"
                / phase,
                phase=phase,
            )
        test_data = next((item.test_data for item in results if item.test_data), None)
        description = (
            evaluation_data_description(snapshot.iteration)
            if phase == "absorption"
            else current_iteration_fp_evaluation_description(snapshot.iteration)
        )
        force_rows = load_force_parities({
            item.model_id: item.force_file
            for item in results
            if item.force_file is not None
        })
        best = best_force_model(force_rows)
        baseline_count = sum(item.baseline_force_file is not None for item in results)
        return MonitorEvent(
            key=event_key,
            event_type="model_evaluation_ready",
            title=f"DP-GEN iter.{snapshot.iteration:06d} {description}",
            message=(
                f"{len(results)} 个模型已经完成测试；最佳模型 {best.model_id}："
                f"F_MAE={best.mae:.4f} eV/Å，F_RMSE={best.rmse:.4f} eV/Å。"
                f" 吸收前基线 {baseline_count}/{len(results)} 个模型可用。"
                + (f" 数据来源：{test_data}" if test_data else "")
            ),
            iteration=snapshot.iteration,
            image_paths=images,
            payload={
                "models": [item.model_id for item in results],
                "test_data": str(test_data) if test_data else None,
                "best_model": best.model_id,
                "force_mae": best.mae,
                "force_rmse": best.rmse,
                "baseline_models": baseline_count,
                "phase": phase,
            },
        )

    @classmethod
    def _evaluation_error_event(
        cls,
        snapshot: IterationSnapshot,
        result: EvaluationResult,
    ) -> MonitorEvent:
        return MonitorEvent(
            key=(
                f"evaluation-error:{result.phase}:"
                f"iter.{snapshot.iteration:06d}:{result.model_id}"
            ),
            event_type="model_evaluation_failed",
            title=(
                f"DP-GEN iter.{snapshot.iteration:06d} "
                f"{cls._phase_label(result.phase)}失败"
            ),
            message=f"模型 {result.model_id}: {result.error}",
            iteration=snapshot.iteration,
            payload={"model_id": result.model_id, "error": result.error},
        )

    def _has_pending_delivery(self, event_key: str) -> bool:
        return any(
            not self.state.is_delivered(event_key, notifier.name)
            for notifier in self.notifiers
        )

    def _deliver(self, event: MonitorEvent, enabled: bool) -> None:
        pending = [
            notifier
            for notifier in self.notifiers
            if not self.state.is_delivered(event.key, notifier.name)
        ]
        if not pending:
            return
        if not enabled:
            print(f"[检查模式] 待发送事件: {event.key} — {event.title}")
            return
        for notifier in pending:
            try:
                notifier.send(event)
                self.state.record_delivery(event.key, notifier.name, True)
                print(f"[通知] {event.key} -> {notifier.name} 成功")
            except Exception as exc:
                self.state.record_delivery(event.key, notifier.name, False, str(exc))
                print(f"[通知][错误] {event.key} -> {notifier.name} 失败: {exc}")

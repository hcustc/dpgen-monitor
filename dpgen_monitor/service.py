from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
import time
import traceback

from .committee_replay import CommitteeReplayEvaluator, CommitteeReplayResult
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
from .proposals import ParameterProposalAdvisor
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
        self.replay_evaluator = (
            CommitteeReplayEvaluator(
                self.config.committee_replay,
                self.config.project.run_dir,
                self.config.project.output_dir,
                self.state,
            )
            if self.config.committee_replay.enabled
            else None
        )
        self.proposal_advisor = (
            ParameterProposalAdvisor(
                self.config.parameter_proposals,
                self.config.committee_replay,
                self.config.project.output_dir,
                self.state,
            )
            if self.config.parameter_proposals.enabled
            else None
        )
        self._last_status: dict[str, str] = {}
        self._stop_event: threading.Event | None = None

    def close(self) -> None:
        self.state.close()

    def run(self, stop_event: threading.Event) -> None:
        self._stop_event = stop_event
        if self.evaluator:
            self.evaluator.set_stop_event(stop_event)
        if self.replay_evaluator:
            self.replay_evaluator.set_stop_event(stop_event)
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
        generations, resets = self.state.reconcile_iterations(snapshots)
        snapshots = [
            replace(snapshot, generation=generations[snapshot.iteration])
            for snapshot in snapshots
        ]
        for reason in resets:
            print(f"[监控][运行回退] {reason}；旧状态已失效")
        for snapshot in snapshots:
            self.state.upsert_stage(snapshot.iteration, snapshot.task)

        ready_stats = 0
        statistics_events: list[MonitorEvent] = []
        active_iterations = {snapshot.iteration for snapshot in snapshots}
        for iteration in sorted(active_iterations):
            inspection = inspections.get(iteration)
            if inspection is None:
                continue
            if iteration < 0 or inspection.status != "ready" or not inspection.stats:
                continue
            self.state.upsert_statistics(iteration, inspection.stats)
            ready_stats += 1
            if iteration < self.config.dpgen.statistics_start_iteration:
                continue
            statistics_events.append(
                self._statistics_event(iteration, inspection.stats)
            )

        eligible_pending = [
            inspection
            for iteration, inspection in inspections.items()
            if iteration in active_iterations
            and iteration >= self.config.dpgen.statistics_start_iteration
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
            evaluations_complete += self._process_evaluation_phase(
                snapshots,
                "absorption",
                self.config.evaluation.absorption_ready_task,
                notify,
            )

        replays_complete = 0
        replay_evaluator = getattr(self, "replay_evaluator", None)
        if evaluate and replay_evaluator and not self._stop_requested():
            replays_complete += self._process_committee_replays(snapshots, notify)

        proposal = None
        proposal_advisor = getattr(self, "proposal_advisor", None)
        if evaluate and proposal_advisor and not self._stop_requested():
            proposal = proposal_advisor.consider(snapshots)
            if proposal and proposal["status"] == "pending":
                self._deliver(self._parameter_proposal_event(proposal), notify)

        # If a monitor starts late, absorption evaluation may become ready at
        # the same time as model-deviation statistics. Keep the conceptual
        # workflow order: absorption notifications precede exploration stats.
        if not self._stop_requested():
            for event in statistics_events:
                self._deliver(event, notify)

        if (
            evaluate
            and self.evaluator
            and self.config.evaluation.blind_spot_enabled
            and not self._stop_requested()
        ):
            evaluations_complete += self._process_evaluation_phase(
                snapshots,
                "blind_spot",
                self.config.evaluation.blind_spot_ready_task,
                notify,
            )

        list_proposals = getattr(self.state, "list_parameter_proposals", None)
        pending_proposals = (
            len(list_proposals("pending")) if list_proposals is not None else 0
        )
        summary = {
            "iterations": len(snapshots),
            "statistics_ready": ready_stats,
            "evaluations_complete": evaluations_complete,
            "committee_replays_complete": replays_complete,
            "parameter_proposals_pending": pending_proposals,
        }
        self._log_changed(
            "scan-summary",
            "[监控总览] "
            f"发现迭代 {summary['iterations']}；"
            f"完整探索统计 {summary['statistics_ready']} 轮；"
            f"本轮新增模型结果 {summary['evaluations_complete']} 个；"
            f"本轮新增委员会回放 {summary['committee_replays_complete']} 个；"
            f"待审批参数建议 {summary['parameter_proposals_pending']} 个",
        )
        return summary

    def _process_committee_replays(
        self,
        snapshots: list[IterationSnapshot],
        notify: bool,
    ) -> int:
        if self.replay_evaluator is None:
            return 0
        completed_count = 0
        by_iteration = {snapshot.iteration: snapshot for snapshot in snapshots}
        replay_config = self.config.committee_replay
        for model in snapshots:
            if self._stop_requested():
                break
            if model.iteration < replay_config.start_iteration:
                continue
            for offset in replay_config.source_offsets:
                source_iteration = model.iteration - offset
                source = by_iteration.get(source_iteration)
                if source is None:
                    continue
                event_key = self._committee_replay_event_key(
                    model.iteration, source.iteration
                )
                row = self.state.get_committee_replay(
                    model.iteration, source.iteration
                )
                delivery_complete = all(
                    self.state.is_delivered(event_key, notifier.name)
                    for notifier in self.notifiers
                )
                if row and row["status"] == "complete" and delivery_complete:
                    continue
                status_key = (
                    f"committee-replay:model.{model.iteration:06d}:"
                    f"source.{source.iteration:06d}"
                )
                if model.train_dir is None:
                    self._log_changed(status_key, "[委员会回放] 等待训练目录生成")
                    continue
                if model.task is None or model.task < replay_config.ready_task:
                    current = "未知" if model.task is None else f"task {model.task:02d}"
                    self._log_changed(
                        status_key,
                        f"[委员会回放] iter.{model.iteration:06d} 当前 {current}；"
                        f"等待 task {replay_config.ready_task:02d}",
                    )
                    continue

                self._log_changed(
                    status_key,
                    f"[委员会回放] iter.{model.iteration:06d} 委员会回放 "
                    f"iter.{source.iteration:06d} holdout",
                )
                result = self.replay_evaluator.evaluate(model, source)
                if result.status == "complete":
                    completed_count += 1
                    self._deliver(self._committee_replay_event(result), notify)
                elif result.status == "running":
                    self._log_changed(
                        status_key,
                        "[委员会回放] 已由 DPDispatcher 提交；下轮扫描继续查询",
                    )
                    # The local profile owns one GPU. Do not enqueue another
                    # replay until this request reaches a terminal state.
                    return completed_count
                elif result.status == "failed":
                    self._deliver(self._committee_replay_error_event(result), notify)
                elif result.status == "cancelled":
                    self._log_changed(status_key, "[委员会回放] 已取消，保留已有中间文件")
                    return completed_count
                elif result.error:
                    self._log_changed(status_key, f"[委员会回放] {result.error}")
        return completed_count

    def _process_evaluation_phase(
        self,
        snapshots: list[IterationSnapshot],
        phase: str,
        required_task: int,
        notify: bool,
    ) -> int:
        completed_count = 0
        for snapshot in snapshots:
            if self._stop_requested():
                break
            if snapshot.iteration < self.config.evaluation.start_iteration:
                continue
            event_key = self._evaluation_event_key(snapshot.iteration, phase)
            results_complete = self.state.evaluations_complete(
                snapshot.iteration,
                phase,
                self.config.evaluation.model_ids,
            )
            delivery_complete = all(
                self.state.is_delivered(event_key, notifier.name)
                for notifier in self.notifiers
            )
            if results_complete and delivery_complete:
                continue
            ready, reason = self._phase_readiness(snapshot, phase, required_task)
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
            completed_count += len(complete)
            cancelled = [item for item in results if item.status == "cancelled"]
            if cancelled or self._stop_requested():
                self._log_changed(
                    status_key,
                    f"[模型监控] iter.{snapshot.iteration:06d} "
                    f"{self._phase_label(phase)}已取消；"
                    f"保留已完成结果 {len(complete)}/"
                    f"{len(self.config.evaluation.model_ids)}；未发送失败通知",
                )
                break
            failures = [item for item in results if item.status == "failed"]
            for failure in failures:
                self._deliver(
                    self._evaluation_error_event(snapshot, failure), notify
                )
            if len(complete) == len(self.config.evaluation.model_ids):
                self._deliver(
                    self._evaluation_event(snapshot, phase, results), notify
                )
                continue
            waiting = [
                item.model_id for item in results if item.status == "waiting"
            ]
            self._log_changed(
                status_key,
                f"[模型监控] iter.{snapshot.iteration:06d} "
                f"{self._phase_label(phase)}完成 "
                f"{len(complete)}/{len(self.config.evaluation.model_ids)}"
                + (f"；等待模型 {waiting}" if waiting else ""),
            )
        return completed_count

    def _stop_requested(self) -> bool:
        return bool(self._stop_event and self._stop_event.is_set())

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
        message = (
            f"candidate {stats['candidate_count']}/{stats['candidate_total']} "
            f"({format_percentage(stats['candidate_percent'])}), "
            f"failed {stats['failed_count']}/{stats['failed_total']} "
            f"({format_percentage(stats['failed_percent'])}), "
            f"accurate {format_percentage(stats['accurate_percent'])}"
        )
        event = MonitorEvent(
            key=event_key,
            event_type="exploration_stats_ready",
            title=f"DP-GEN iter.{iteration:06d} 探索统计",
            message=message,
            iteration=iteration,
            payload=dict(stats),
        )
        if self._has_pending_delivery(event):
            image = render_statistics_trend(
                self.state.list_statistics(),
                self.config.project.output_dir / "artifacts" / "dpgen_stats_trend.png",
            )
            event = replace(event, image_paths=(image,))
        return event

    def _evaluation_event(
        self,
        snapshot: IterationSnapshot,
        phase: str,
        results: list[EvaluationResult],
    ) -> MonitorEvent:
        event_key = self._evaluation_event_key(snapshot.iteration, phase)
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
        event = MonitorEvent(
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
        if self._has_pending_delivery(event):
            images = render_evaluation(
                snapshot.iteration,
                results,
                self.config.project.output_dir
                / "artifacts"
                / f"iter.{snapshot.iteration:06d}"
                / phase,
                phase=phase,
            )
            event = replace(event, image_paths=images)
        return event

    @staticmethod
    def _committee_replay_event_key(
        model_iteration: int, source_iteration: int
    ) -> str:
        return (
            f"committee-replay:model.iter.{model_iteration:06d}:"
            f"source.iter.{source_iteration:06d}"
        )

    def _committee_replay_event(
        self, result: CommitteeReplayResult
    ) -> MonitorEvent:
        summary = result.summary
        if summary is None:
            raise ValueError(f"委员会回放摘要不可读: {result.summary_file}")
        tracked = int(summary["tracked_candidates"])
        absorbed = int(summary["absorbed"])
        remaining = int(summary["remaining_candidate"])
        worsened = int(summary["worsened_to_failed"])
        return MonitorEvent(
            key=self._committee_replay_event_key(
                result.model_iteration, result.source_iteration
            ),
            event_type="committee_replay_ready",
            title=(
                f"DP-GEN iter.{result.model_iteration:06d} 委员会回放 "
                f"iter.{result.source_iteration:06d}"
            ),
            message=(
                f"固定 holdout {int(summary['frame_count'])} 帧；"
                f"跟踪候选 {tracked} 个，其中 {absorbed} 个已吸收 "
                f"({format_percentage(float(summary['absorption_percent']))})，"
                f"{remaining} 个仍为 candidate，{worsened} 个升为 failed。"
                f" holdout 全原子 candidate="
                f"{format_percentage(float(summary['candidate_percent']))}，"
                f"failed={format_percentage(float(summary['failed_percent']))}。"
            ),
            iteration=result.model_iteration,
            payload=summary,
        )

    @classmethod
    def _committee_replay_error_event(
        cls, result: CommitteeReplayResult
    ) -> MonitorEvent:
        return MonitorEvent(
            key=(
                "committee-replay-error:"
                f"model.iter.{result.model_iteration:06d}:"
                f"source.iter.{result.source_iteration:06d}"
            ),
            event_type="committee_replay_failed",
            title=(
                f"DP-GEN iter.{result.model_iteration:06d} 委员会回放失败"
            ),
            message=(
                f"来源 iter.{result.source_iteration:06d}: {result.error}"
            ),
            iteration=result.model_iteration,
            payload={
                "model_iteration": result.model_iteration,
                "source_iteration": result.source_iteration,
                "error": result.error,
            },
        )

    @staticmethod
    def _parameter_proposal_event(proposal: dict) -> MonitorEvent:
        job = proposal["proposed_job"]
        target = int(proposal["target_iteration"])
        return MonitorEvent(
            key=f"parameter-proposal:{proposal['proposal_id']}",
            event_type="parameter_proposal_ready",
            title=f"DP-GEN iter.{target:06d} model_devi 参数待审批",
            message=(
                "DP-GEN 已停在 post_train gate；repeat_last 建议为 "
                f"{job['ensemble'].upper()}、nsteps={job['nsteps']}、"
                f"trj_freq={job['trj_freq']}、temps={job['temps']}。"
                f"建议 ID: {proposal['proposal_id']}。"
                "批准和应用是两个独立的人工操作；不会自动恢复 DP-GEN。"
            ),
            iteration=target,
            payload={
                "proposal_id": proposal["proposal_id"],
                "target_iteration": target,
                "status": proposal["status"],
                "proposed_job": job,
                "evidence": proposal["evidence"],
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

    def _has_pending_delivery(self, event: MonitorEvent) -> bool:
        return any(
            not self.state.has_delivered_content(
                event.key, notifier.name, event.content_hash
            )
            for notifier in self.notifiers
        )

    def _deliver(self, event: MonitorEvent, enabled: bool) -> None:
        content_hash = event.content_hash
        pending = [
            notifier
            for notifier in self.notifiers
            if not self.state.has_delivered_content(
                event.key, notifier.name, content_hash
            )
        ]
        if not pending:
            return
        if not enabled:
            print(f"[检查模式] 待发送事件: {event.key} — {event.title}")
            return
        for notifier in pending:
            try:
                notifier.send(event)
                self.state.record_delivery(
                    event.key,
                    notifier.name,
                    True,
                    content_hash=content_hash,
                )
                print(f"[通知] {event.key} -> {notifier.name} 成功")
            except Exception as exc:
                self.state.record_delivery(
                    event.key,
                    notifier.name,
                    False,
                    str(exc),
                    content_hash=content_hash,
                )
                print(f"[通知][错误] {event.key} -> {notifier.name} 失败: {exc}")

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .config import DpgenConfig


ITER_RE = re.compile(r"iter\.(\d{6})")
TASK_RE = re.compile(r"task\s+(\d{2})")
SYSTEM_RE = re.compile(
    r"system\s+(\d+)\s+(candidate|failed|accurate)\s*:\s*"
    r"([0-9,]+)\s+in\s+([0-9,]+)\s+([\d.]+)\s*%"
)
RATIO_RE = re.compile(r"system\s+(\d+)\s+accurate_ratio\s*:\s*([\d.]+)")


@dataclass(frozen=True)
class IterationSnapshot:
    iteration: int
    iteration_dir: Path
    train_dir: Path | None
    task: int | None
    source_identity: str = ""
    generation: int = 0


@dataclass(frozen=True)
class StatisticsInspection:
    iteration: int
    status: str
    task: int | None
    statistics_task: int
    stats: dict[str, int | float] | None = None
    missing_metrics: tuple[str, ...] = ()
    detail: str | None = None

    def message(self, check_interval: int | None = None) -> str:
        name = f"iter.{self.iteration:06d}"
        if self.status == "pending":
            if self.task is None:
                progress = "已启动，尚未检测到明确的任务阶段"
            elif self.task < self.statistics_task:
                progress = (
                    f"当前到 task {self.task:02d}，探索统计通常在 "
                    f"task {self.statistics_task:02d} 生成"
                )
            else:
                progress = f"当前到 task {self.task:02d}，统计行可能仍在写入"
            suffix = (
                f"；{check_interval}s 后再次检查"
                if check_interval is not None
                else ""
            )
            return f"{name} {progress}{suffix}"
        if self.status == "partial":
            missing = ", ".join(self.missing_metrics)
            return f"{name} 统计正在写入，当前缺少 {missing}；稍后重试"
        if self.status == "unrecognized":
            return f"{name} 已出现疑似统计行，但格式无法识别；稍后重试"
        if self.status == "log_missing":
            return f"DP-GEN 日志不存在: {self.detail}"
        if self.status == "read_error":
            return f"读取 {name} 统计失败: {self.detail}"
        return f"{name} 统计尚未就绪"


class DpgenObserver:
    """Read DP-GEN checkpoints, log stages, and exploration statistics."""

    def __init__(self, run_dir: Path, config: DpgenConfig):
        self.run_dir = run_dir
        self.config = config
        self.log_path = run_dir / config.log_file
        self.record_path = run_dir / config.record_file

    def scan(self) -> tuple[list[IterationSnapshot], dict[int, StatisticsInspection]]:
        record_tasks = self._read_record_tasks()
        log_tasks, inspections = self._read_log()
        snapshots = []
        for iteration_dir in sorted(self.run_dir.glob("iter.[0-9][0-9][0-9][0-9][0-9][0-9]")):
            match = ITER_RE.fullmatch(iteration_dir.name)
            if not match or not iteration_dir.is_dir():
                continue
            iteration = int(match.group(1))
            task_values = [
                task for task in (record_tasks.get(iteration), log_tasks.get(iteration))
                if task is not None
            ]
            task = max(task_values) if task_values else None
            train_dir = iteration_dir / "00.train"
            stat = iteration_dir.stat()
            snapshots.append(
                IterationSnapshot(
                    iteration=iteration,
                    iteration_dir=iteration_dir,
                    train_dir=train_dir if train_dir.is_dir() else None,
                    task=task,
                    source_identity=f"{stat.st_dev}:{stat.st_ino}",
                )
            )
            if iteration not in inspections:
                inspections[iteration] = StatisticsInspection(
                    iteration,
                    "pending",
                    task,
                    self.config.statistics_task,
                )
            elif inspections[iteration].task != task:
                old = inspections[iteration]
                inspections[iteration] = StatisticsInspection(
                    iteration=old.iteration,
                    status=old.status,
                    task=task,
                    statistics_task=old.statistics_task,
                    stats=old.stats,
                    missing_metrics=old.missing_metrics,
                    detail=old.detail,
                )
        return snapshots, inspections

    def _read_record_tasks(self) -> dict[int, int]:
        tasks: dict[int, int] = {}
        if not self.record_path.exists():
            return tasks
        try:
            with self.record_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    try:
                        iteration, task = map(int, parts)
                    except ValueError:
                        continue
                    # record.dpgen may be truncated and restarted from an older
                    # stage during a recovery.  The last entry is authoritative;
                    # taking max() would keep the superseded run forever.
                    tasks[iteration] = task
        except OSError:
            return {}
        return tasks

    def _read_log(
        self,
    ) -> tuple[dict[int, int], dict[int, StatisticsInspection]]:
        if not self.log_path.exists():
            return {}, {}
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return {}, {
                -1: StatisticsInspection(
                    -1,
                    "read_error",
                    None,
                    self.config.statistics_task,
                    detail=str(exc),
                )
            }

        tasks: dict[int, int] = {}
        stats_by_iter: dict[int, dict[str, dict[str, int | float]]] = {}
        ratios_by_iter: dict[int, dict[str, float]] = {}
        suspicious: set[int] = set()
        seen: set[int] = set()
        current_iter: int | None = None
        latest_epoch_iter: int | None = None

        for line in lines:
            iter_match = ITER_RE.search(line)
            task_match = TASK_RE.search(line)
            # 只把 DP-GEN 的阶段标题当作上下文切换，避免命令或文件路径中的
            # iter.XXXXXX 把后续统计误归到其他迭代。
            is_stage_header = bool(task_match or re.search(r"[=-]{3,}", line))
            if iter_match and is_stage_header:
                next_iter = int(iter_match.group(1))
                if latest_epoch_iter is not None and next_iter < latest_epoch_iter:
                    # A lower iteration after a later one marks a run rollback.
                    # Forget parsed data from the superseded tail of the log.
                    for mapping in (tasks, stats_by_iter, ratios_by_iter):
                        for iteration in tuple(mapping):
                            if iteration >= next_iter:
                                mapping.pop(iteration, None)
                    suspicious.difference_update(
                        {
                            iteration
                            for iteration in suspicious
                            if iteration >= next_iter
                        }
                    )
                    seen.difference_update(
                        {iteration for iteration in seen if iteration >= next_iter}
                    )
                    latest_epoch_iter = next_iter
                elif latest_epoch_iter is None or next_iter > latest_epoch_iter:
                    latest_epoch_iter = next_iter
                current_iter = next_iter
                seen.add(current_iter)
            if current_iter is None:
                continue

            if task_match:
                task = int(task_match.group(1))
                tasks[current_iter] = task

            clean = line.replace(",", "")
            system_match = SYSTEM_RE.search(clean)
            if system_match:
                system_id = system_match.group(1).zfill(3)
                metric = system_match.group(2)
                entry = stats_by_iter.setdefault(current_iter, {}).setdefault(system_id, {})
                entry[f"{metric}_count"] = int(system_match.group(3))
                entry[f"{metric}_total"] = int(system_match.group(4))
                entry[f"{metric}_percent"] = float(system_match.group(5))
                continue

            ratio_match = RATIO_RE.search(clean)
            if ratio_match:
                system_id = ratio_match.group(1).zfill(3)
                ratios_by_iter.setdefault(current_iter, {})[system_id] = (
                    float(ratio_match.group(2)) * 100.0
                )
                continue

            if "system" in clean and any(
                keyword in clean
                for keyword in ("candidate", "failed", "accurate", "accurate_ratio")
            ):
                suspicious.add(current_iter)

        inspections: dict[int, StatisticsInspection] = {}
        for iteration in seen:
            task = tasks.get(iteration)
            systems = stats_by_iter.get(iteration, {})
            ratios = ratios_by_iter.get(iteration, {})
            target = "000" if "000" in systems or "000" in ratios else None
            if target is None and systems:
                target = next(iter(systems))
            if target is None and ratios:
                target = next(iter(ratios))
            if target is None:
                status = "unrecognized" if iteration in suspicious else "pending"
                inspections[iteration] = StatisticsInspection(
                    iteration, status, task, self.config.statistics_task
                )
                continue

            selected = dict(systems.get(target, {}))
            if "accurate_percent" not in selected and target in ratios:
                selected["accurate_percent"] = ratios[target]
            required = {"candidate_percent", "failed_percent", "accurate_percent"}
            missing = tuple(sorted(required.difference(selected)))
            if missing:
                inspections[iteration] = StatisticsInspection(
                    iteration,
                    "partial",
                    task,
                    self.config.statistics_task,
                    missing_metrics=missing,
                )
                continue

            stats = {
                "candidate_count": int(selected.get("candidate_count", 0)),
                "candidate_total": int(selected.get("candidate_total", 0)),
                "candidate_percent": float(selected["candidate_percent"]),
                "failed_count": int(selected.get("failed_count", 0)),
                "failed_total": int(selected.get("failed_total", 0)),
                "failed_percent": float(selected["failed_percent"]),
                "accurate_count": int(selected.get("accurate_count", 0)),
                "accurate_total": int(selected.get("accurate_total", 0)),
                "accurate_percent": float(selected["accurate_percent"]),
            }
            # 日志中的百分比通常仅有两位小数。优先使用精确计数恢复比例，
            # 避免把极小非零比例显示为 0%，或把 99.9987% 显示为 100%。
            for prefix in ("candidate", "failed", "accurate"):
                total = int(stats[f"{prefix}_total"])
                if total > 0:
                    stats[f"{prefix}_percent"] = (
                        int(stats[f"{prefix}_count"]) / total * 100.0
                    )
            inspections[iteration] = StatisticsInspection(
                iteration,
                "ready",
                task,
                self.config.statistics_task,
                stats=stats,
            )
        return tasks, inspections

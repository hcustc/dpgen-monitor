from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterator

import yaml

from .config import CommitteeReplayConfig, DpgenConfig, ParameterProposalConfig
from .dpgen import IterationSnapshot
from .state import StateStore


JOB_FIELDS = ("sys_idx", "temps", "trj_freq", "nsteps", "ensemble", "_idx")
REQUIRED_JOB_FIELDS = {"sys_idx", "temps", "trj_freq", "nsteps", "ensemble"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_parameter_data(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取 DP-GEN 参数文件: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"DP-GEN 参数文件顶层必须是映射: {path}")
    jobs = data.get("model_devi_jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"DP-GEN 参数文件缺少非空 model_devi_jobs: {path}")
    if not all(isinstance(job, dict) for job in jobs):
        raise ValueError("model_devi_jobs 的每一项都必须是映射")
    return data, jobs


def _validated_repeat_job(
    source: dict[str, Any],
    target_iteration: int,
    config: ParameterProposalConfig,
) -> dict[str, Any]:
    unknown = set(source) - set(JOB_FIELDS)
    missing = REQUIRED_JOB_FIELDS - set(source)
    if unknown:
        raise ValueError(
            "repeat_last 暂不支持上一任务中的字段: "
            + ", ".join(sorted(unknown))
        )
    if missing:
        raise ValueError(
            "上一 model_devi job 缺少字段: " + ", ".join(sorted(missing))
        )

    sys_idx = source["sys_idx"]
    temps = source["temps"]
    if (
        not isinstance(sys_idx, list)
        or not sys_idx
        or len(sys_idx) > 64
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in sys_idx
        )
        or len(set(sys_idx)) != len(sys_idx)
    ):
        raise ValueError("repeat_last 需要 1--64 个不重复的非负 sys_idx")
    if (
        not isinstance(temps, list)
        or not temps
        or len(temps) > 32
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            or float(value) > 10000
            for value in temps
        )
    ):
        raise ValueError("repeat_last 需要 1--32 个 0--10000 K 的有限温度")

    nsteps = source["nsteps"]
    trj_freq = source["trj_freq"]
    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps <= 0:
        raise ValueError("model_devi nsteps 必须是正整数")
    if nsteps > config.max_nsteps:
        raise ValueError(
            f"上一任务 nsteps={nsteps} 超过批准上限 {config.max_nsteps}"
        )
    if (
        isinstance(trj_freq, bool)
        or not isinstance(trj_freq, int)
        or trj_freq <= 0
        or trj_freq > nsteps
        or nsteps % trj_freq != 0
    ):
        raise ValueError("model_devi trj_freq 必须整除 nsteps")
    if source["ensemble"] != "nvt":
        raise ValueError("repeat_last 第一版仅允许 NVT model_devi job")

    return {
        "sys_idx": list(sys_idx),
        "temps": list(temps),
        "trj_freq": trj_freq,
        "nsteps": nsteps,
        "ensemble": "nvt",
        "_idx": target_iteration,
    }


class ParameterProposalAdvisor:
    """Create a deterministic draft only after the natural post-train gate."""

    def __init__(
        self,
        config: ParameterProposalConfig,
        replay_config: CommitteeReplayConfig,
        output_dir: Path,
        state: StateStore,
    ):
        if config.parameter_file is None:
            raise ValueError("parameter_proposals.parameter_file 不能为空")
        self.config = config
        self.replay_config = replay_config
        self.parameter_file = config.parameter_file.resolve()
        self.output_dir = output_dir.resolve()
        self.state = state

    def _replay_evidence(self, target_iteration: int) -> list[dict[str, Any]] | None:
        evidence: list[dict[str, Any]] = []
        for offset in self.replay_config.source_offsets:
            source_iteration = target_iteration - offset
            row = self.state.get_committee_replay(target_iteration, source_iteration)
            if not row or row["status"] != "complete" or not row["summary_file"]:
                return None
            summary_file = Path(str(row["summary_file"])).resolve()
            try:
                summary_file.relative_to(self.output_dir)
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
            except (ValueError, OSError, json.JSONDecodeError):
                return None
            fingerprint = summary.get("fingerprint")
            model_generation = summary.get("model_generation")
            source_generation = summary.get("source_generation")
            frame_count = summary.get("frame_count")
            tracked_candidates = summary.get("tracked_candidates")
            percentages = {
                key: summary.get(key)
                for key in (
                    "absorption_percent",
                    "candidate_percent",
                    "failed_percent",
                )
            }
            if (
                not isinstance(fingerprint, str)
                or not fingerprint
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in (model_generation, source_generation)
                )
                or isinstance(frame_count, bool)
                or not isinstance(frame_count, int)
                or frame_count <= 0
                or isinstance(tracked_candidates, bool)
                or not isinstance(tracked_candidates, int)
                or tracked_candidates < 0
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0 <= float(value) <= 100
                    for value in percentages.values()
                )
            ):
                return None

            current_model_generation = self.state.get_iteration_generation(
                target_iteration
            )
            current_source_generation = self.state.get_iteration_generation(
                source_iteration
            )
            expected_file = (
                self.output_dir
                / "evaluations"
                / f"iter.{target_iteration:06d}"
            )
            if current_model_generation > 0:
                expected_file /= f"generation.{current_model_generation:06d}"
            expected_file /= (
                Path("committee_replay")
                / f"source.iter.{source_iteration:06d}"
            )
            if current_source_generation > 0:
                expected_file /= f"generation.{current_source_generation:06d}"
            expected_file = (expected_file / "summary.json").resolve()
            if (
                summary_file != expected_file
                or summary.get("model_iteration") != target_iteration
                or summary.get("source_iteration") != source_iteration
                or model_generation != current_model_generation
                or source_generation != current_source_generation
            ):
                return None
            evidence.append(
                {
                    "source_iteration": source_iteration,
                    "summary_file": str(summary_file),
                    "fingerprint": fingerprint,
                    "model_generation": model_generation,
                    "source_generation": source_generation,
                    "frame_count": frame_count,
                    "tracked_candidates": tracked_candidates,
                    **percentages,
                }
            )
        return evidence

    def consider(
        self, snapshots: list[IterationSnapshot]
    ) -> dict[str, Any] | None:
        _, jobs = _load_parameter_data(self.parameter_file)
        target_iteration = len(jobs)
        if target_iteration < self.config.start_iteration:
            return None
        snapshot = next(
            (item for item in snapshots if item.iteration == target_iteration),
            None,
        )
        if snapshot is None or snapshot.task != self.config.required_task:
            return None
        evidence = self._replay_evidence(target_iteration)
        if evidence is None:
            return None

        job = _validated_repeat_job(jobs[-1], target_iteration, self.config)
        parameter_sha256 = _sha256_file(self.parameter_file)
        identity = {
            "target_iteration": target_iteration,
            "parameter_sha256": parameter_sha256,
            "job": job,
            "evidence": evidence,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        proposal_id = f"model-devi-{target_iteration:06d}-{digest[:12]}"
        self.state.create_parameter_proposal(
            proposal_id=proposal_id,
            target_iteration=target_iteration,
            status="pending",
            parameter_file=str(self.parameter_file),
            parameter_sha256=parameter_sha256,
            strategy=self.config.strategy,
            proposed_job=job,
            evidence=evidence,
        )
        return self.state.get_parameter_proposal(proposal_id)


class ParameterFileController:
    """Apply an approved repeat-last proposal without starting DP-GEN."""

    def __init__(
        self,
        config: ParameterProposalConfig,
        dpgen_config: DpgenConfig,
        run_dir: Path,
        state: StateStore,
    ):
        if config.parameter_file is None:
            raise ValueError("parameter_proposals.parameter_file 不能为空")
        self.config = config
        self.parameter_file = config.parameter_file.resolve()
        self.record_file = run_dir.resolve() / dpgen_config.record_file
        self.run_dir = run_dir.resolve()
        self.state = state

    @contextmanager
    def _parameter_lock(self) -> Iterator[None]:
        lock_file = self.parameter_file.with_name(
            f".{self.parameter_file.name}.dpgen-monitor.lock"
        )
        handle = lock_file.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _assert_natural_gate(self, target_iteration: int) -> None:
        latest: tuple[int, int] | None = None
        try:
            for line in self.record_file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                fields = line.split()
                if len(fields) != 2:
                    continue
                try:
                    latest = int(fields[0]), int(fields[1])
                except ValueError:
                    continue
        except OSError as exc:
            raise ValueError(f"无法读取 DP-GEN record: {self.record_file}") from exc
        expected = (target_iteration, self.config.required_task)
        if latest != expected:
            raise ValueError(
                f"DP-GEN 不在自然停止点；record 最新为 {latest}，期待 {expected}"
            )
        iteration_dir = self.run_dir / f"iter.{target_iteration:06d}"
        if not (iteration_dir / "00.train").is_dir():
            raise ValueError("自然停止点缺少已完成的训练目录")
        if (iteration_dir / "01.model_devi").exists():
            raise ValueError("目标迭代已经存在 01.model_devi，拒绝追加参数")

    @staticmethod
    def _render_job(job: dict[str, Any]) -> str:
        return (
            f"  - sys_idx: {json.dumps(job['sys_idx'])}\n"
            f"    temps: {json.dumps(job['temps'])}\n"
            f"    trj_freq: {job['trj_freq']}\n"
            f"    nsteps: {job['nsteps']}\n"
            f"    ensemble: {job['ensemble']}\n"
            f"    _idx: {job['_idx']}\n\n"
        )

    @classmethod
    def _append_job_text(cls, source: str, job: dict[str, Any]) -> str:
        lines = source.splitlines(keepends=True)
        start = next(
            (
                index
                for index, line in enumerate(lines)
                if re.match(r"^model_devi_jobs\s*:", line)
            ),
            None,
        )
        if start is None:
            raise ValueError("参数文件缺少顶层 model_devi_jobs")
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if re.match(r"^[A-Za-z_][A-Za-z0-9_.-]*\s*:", lines[index])
            ),
            len(lines),
        )
        insert_at = end
        while insert_at > start + 1:
            stripped = lines[insert_at - 1].strip()
            if stripped and not stripped.startswith("#"):
                break
            insert_at -= 1
        lines.insert(insert_at, cls._render_job(job))
        return "".join(lines)

    def apply(self, proposal_id: str) -> tuple[Path, Path | None, bool]:
        proposal = self.state.get_parameter_proposal(proposal_id)
        if proposal is None:
            raise ValueError(f"参数建议不存在: {proposal_id}")
        if proposal["status"] == "applied":
            backup = proposal.get("backup_file")
            return self.parameter_file, Path(backup) if backup else None, False
        if proposal["status"] != "approved":
            raise ValueError("只有 approved 参数建议可以应用")
        if Path(proposal["parameter_file"]).resolve() != self.parameter_file:
            raise ValueError("建议中的参数文件与当前配置不一致")
        if proposal["strategy"] != self.config.strategy:
            raise ValueError("建议策略与当前配置不一致")
        target_iteration = int(proposal["target_iteration"])
        proposed_job = proposal["proposed_job"]

        with self._parameter_lock():
            source = self.parameter_file.read_text(encoding="utf-8")
            data, jobs = _load_parameter_data(self.parameter_file)
            if len(jobs) == target_iteration + 1 and jobs[-1] == proposed_job:
                self.state.transition_parameter_proposal(
                    proposal_id,
                    expected_statuses=("approved",),
                    status="applied",
                )
                return self.parameter_file, None, False
            if len(jobs) != target_iteration:
                raise ValueError(
                    "model_devi_jobs 长度已变化；拒绝覆盖或插入历史任务"
                )
            if _sha256_file(self.parameter_file) != proposal["parameter_sha256"]:
                raise ValueError("DP-GEN 参数文件在建议生成后已变化")
            self._assert_natural_gate(target_iteration)
            expected_job = _validated_repeat_job(
                jobs[-1], target_iteration, self.config
            )
            if proposed_job != expected_job:
                raise ValueError("建议内容不再符合 repeat_last 白名单策略")

            rendered = self._append_job_text(source, proposed_job)
            try:
                rendered_data = yaml.safe_load(rendered)
            except yaml.YAMLError as exc:
                raise ValueError("追加后的 DP-GEN YAML 无效") from exc
            rendered_jobs = rendered_data.get("model_devi_jobs")
            if (
                not isinstance(rendered_jobs, list)
                or rendered_jobs[:-1] != jobs
                or rendered_jobs[-1] != proposed_job
                or {
                    key: value
                    for key, value in rendered_data.items()
                    if key != "model_devi_jobs"
                }
                != {
                    key: value
                    for key, value in data.items()
                    if key != "model_devi_jobs"
                }
            ):
                raise ValueError("追加验证失败：参数文件出现白名单之外的变化")

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = self.parameter_file.with_name(
                f"{self.parameter_file.name}.bak.{timestamp}.{os.getpid()}"
            )
            shutil.copy2(self.parameter_file, backup)
            temporary = self.parameter_file.with_name(
                f".{self.parameter_file.name}.tmp-{os.getpid()}"
            )
            try:
                temporary.write_text(rendered, encoding="utf-8")
                os.chmod(temporary, self.parameter_file.stat().st_mode)
                os.replace(temporary, self.parameter_file)
            finally:
                temporary.unlink(missing_ok=True)

            self.state.transition_parameter_proposal(
                proposal_id,
                expected_statuses=("approved",),
                status="applied",
                backup_file=str(backup),
            )
            return self.parameter_file, backup, True

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib
from typing import Any

from .evaluation_data import INITIAL_TEST_DATA, PREVIOUS_ITERATION_TEST_DATA


@dataclass(frozen=True)
class ProjectConfig:
    run_dir: Path
    output_dir: Path
    check_interval: int = 300
    heartbeat_interval: int = 21600


@dataclass(frozen=True)
class DpgenConfig:
    log_file: str = "dpgen.log"
    record_file: str = "record.dpgen"
    statistics_task: int = 6
    statistics_start_iteration: int = 0


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool = True
    backend: str = "deepmd"
    command: tuple[str, ...] = ("conda", "run", "-n", "deepmd", "dp")
    model_ids: tuple[str, ...] = ("000", "001", "002", "003")
    model_file: str = "frozen_model.pb"
    test_data: str = PREVIOUS_ITERATION_TEST_DATA
    initial_test_data: str = INITIAL_TEST_DATA
    compare_previous_model: bool = True
    num_test: int = 30
    start_iteration: int = 0
    absorption_ready_task: int = 2
    blind_spot_enabled: bool = True
    blind_spot_ready_task: int = 8


@dataclass(frozen=True)
class CommitteeReplayConfig:
    """Post-training committee replay on unselected model-deviation frames."""

    enabled: bool = False
    command: tuple[str, ...] = ("conda", "run", "-n", "deepmd", "dp")
    executor_profile: str = "local_gpu"
    dispatcher_check_interval: int = 30
    cpu_per_node: int = 4
    gpu_device: int = 0
    max_total_frames: int = 2048
    model_ids: tuple[str, ...] = ("000", "001", "002", "003")
    model_pattern: str = "graph.{model_id}.pb"
    source_offsets: tuple[int, ...] = (1,)
    candidate_manifest: str = "02.fp/candidate_selection.000.jsonl"
    exclude_selected: bool = True
    max_frames_per_task: int = 256
    time_bins: int = 5
    seed: int = 20260905
    type_map: tuple[str, ...] = ()
    relative: float | None = 1.0
    f_trust_lo: float = 0.15
    f_trust_hi: float = 0.30
    start_iteration: int = 1
    ready_task: int = 2


@dataclass(frozen=True)
class ParameterProposalConfig:
    """Human-approved additions to a gated DP-GEN model_devi_jobs list."""

    enabled: bool = False
    parameter_file: Path | None = None
    strategy: str = "repeat_last"
    start_iteration: int = 1
    required_task: int = 2
    max_nsteps: int = 25_000_000


@dataclass(frozen=True)
class NotificationConfig:
    name: str
    type: str
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MonitorConfig:
    project: ProjectConfig
    dpgen: DpgenConfig = field(default_factory=DpgenConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    notifications: tuple[NotificationConfig, ...] = ()
    committee_replay: CommitteeReplayConfig = field(
        default_factory=CommitteeReplayConfig
    )
    parameter_proposals: ParameterProposalConfig = field(
        default_factory=ParameterProposalConfig
    )


def _load_raw_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("rb") as handle:
        if suffix == ".toml":
            return tomllib.load(handle)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "YAML 配置需要额外安装 PyYAML；当前环境可直接使用 TOML 配置"
            ) from exc
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
            return loaded or {}

    raise ValueError(f"不支持的配置格式: {path.suffix}；请使用 .toml、.yaml 或 .yml")


def _resolve_path(value: str, base_dir: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    return expanded.resolve()


def load_config(config_path: str | Path) -> MonitorConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"配置文件不存在: {path}")
    raw = _load_raw_config(path)
    base_dir = path.parent

    project_raw = raw.get("project", {})
    if not project_raw.get("run_dir") or not project_raw.get("output_dir"):
        raise ValueError("配置缺少 project.run_dir 或 project.output_dir")
    project = ProjectConfig(
        run_dir=_resolve_path(str(project_raw["run_dir"]), base_dir),
        output_dir=_resolve_path(str(project_raw["output_dir"]), base_dir),
        check_interval=max(1, int(project_raw.get("check_interval", 300))),
        heartbeat_interval=max(
            0, int(project_raw.get("heartbeat_interval", 21600))
        ),
    )

    dpgen_raw = raw.get("dpgen", {})
    dpgen = DpgenConfig(
        log_file=str(dpgen_raw.get("log_file", "dpgen.log")),
        record_file=str(dpgen_raw.get("record_file", "record.dpgen")),
        statistics_task=int(dpgen_raw.get("statistics_task", 6)),
        statistics_start_iteration=max(
            0, int(dpgen_raw.get("statistics_start_iteration", 0))
        ),
    )

    evaluation_raw = raw.get("evaluation", {})
    evaluation = EvaluationConfig(
        enabled=bool(evaluation_raw.get("enabled", True)),
        backend=str(evaluation_raw.get("backend", "deepmd")),
        command=tuple(str(item) for item in evaluation_raw.get(
            "command", ["conda", "run", "-n", "deepmd", "dp"]
        )),
        model_ids=tuple(str(item) for item in evaluation_raw.get(
            "model_ids", ["000", "001", "002", "003"]
        )),
        model_file=str(evaluation_raw.get("model_file", "frozen_model.pb")),
        test_data=str(evaluation_raw.get("test_data", PREVIOUS_ITERATION_TEST_DATA)),
        initial_test_data=str(
            evaluation_raw.get("initial_test_data", INITIAL_TEST_DATA)
        ),
        compare_previous_model=bool(
            evaluation_raw.get("compare_previous_model", True)
        ),
        num_test=max(0, int(evaluation_raw.get("num_test", 30))),
        start_iteration=max(0, int(evaluation_raw.get("start_iteration", 0))),
        absorption_ready_task=max(
            0, int(evaluation_raw.get("absorption_ready_task", 2))
        ),
        blind_spot_enabled=bool(
            evaluation_raw.get("blind_spot_enabled", True)
        ),
        blind_spot_ready_task=max(
            0, int(evaluation_raw.get("blind_spot_ready_task", 8))
        ),
    )

    replay_raw = raw.get("committee_replay", {})
    removed_runner_keys = {
        "execution",
        "remote_target",
        "ssh_command",
        "copy_command",
        "remote_root",
        "remote_command",
    }.intersection(replay_raw)
    if removed_runner_keys:
        names = ", ".join(sorted(removed_runner_keys))
        raise ValueError(
            "committee_replay 已移除原始 SSH runner 配置: "
            f"{names}；请改用 executor_profile = \"local_gpu\""
        )
    relative_raw = replay_raw.get("relative", 1.0)
    relative = None if relative_raw is None else float(relative_raw)
    source_offsets = tuple(
        sorted({int(value) for value in replay_raw.get("source_offsets", [1])})
    )
    if not source_offsets or any(value <= 0 for value in source_offsets):
        raise ValueError("committee_replay.source_offsets 必须是正整数列表")
    f_trust_lo = float(replay_raw.get("f_trust_lo", 0.15))
    f_trust_hi = float(replay_raw.get("f_trust_hi", 0.30))
    if not 0 <= f_trust_lo < f_trust_hi:
        raise ValueError("committee_replay 需要 0 <= f_trust_lo < f_trust_hi")
    committee_replay = CommitteeReplayConfig(
        enabled=bool(replay_raw.get("enabled", False)),
        command=tuple(
            str(item)
            for item in replay_raw.get("command", list(evaluation.command))
        ),
        executor_profile=str(
            replay_raw.get("executor_profile", "local_gpu")
        ),
        dispatcher_check_interval=max(
            1, int(replay_raw.get("dispatcher_check_interval", 30))
        ),
        cpu_per_node=max(1, int(replay_raw.get("cpu_per_node", 4))),
        gpu_device=max(0, int(replay_raw.get("gpu_device", 0))),
        max_total_frames=max(
            1, int(replay_raw.get("max_total_frames", 2048))
        ),
        model_ids=tuple(
            str(item)
            for item in replay_raw.get("model_ids", list(evaluation.model_ids))
        ),
        model_pattern=str(
            replay_raw.get("model_pattern", "graph.{model_id}.pb")
        ),
        source_offsets=source_offsets,
        candidate_manifest=str(
            replay_raw.get(
                "candidate_manifest", "02.fp/candidate_selection.000.jsonl"
            )
        ),
        exclude_selected=bool(replay_raw.get("exclude_selected", True)),
        max_frames_per_task=max(
            1, int(replay_raw.get("max_frames_per_task", 256))
        ),
        time_bins=max(1, int(replay_raw.get("time_bins", 5))),
        seed=int(replay_raw.get("seed", 20260905)),
        type_map=tuple(str(item) for item in replay_raw.get("type_map", [])),
        relative=relative,
        f_trust_lo=f_trust_lo,
        f_trust_hi=f_trust_hi,
        start_iteration=max(1, int(replay_raw.get("start_iteration", 1))),
        ready_task=max(0, int(replay_raw.get("ready_task", 2))),
    )

    proposal_raw = raw.get("parameter_proposals", {})
    parameter_file_raw = proposal_raw.get("parameter_file")
    parameter_proposals = ParameterProposalConfig(
        enabled=bool(proposal_raw.get("enabled", False)),
        parameter_file=(
            _resolve_path(str(parameter_file_raw), base_dir)
            if parameter_file_raw
            else None
        ),
        strategy=str(proposal_raw.get("strategy", "repeat_last")),
        start_iteration=max(0, int(proposal_raw.get("start_iteration", 1))),
        required_task=max(0, int(proposal_raw.get("required_task", 2))),
        max_nsteps=max(1, int(proposal_raw.get("max_nsteps", 25_000_000))),
    )

    notification_entries = raw.get("notifications", [])
    notifications = []
    for index, entry in enumerate(notification_entries):
        entry = dict(entry)
        notifier_type = str(entry.pop("type", "console"))
        name = str(entry.pop("name", f"{notifier_type}-{index + 1}"))
        enabled = bool(entry.pop("enabled", True))
        notifications.append(NotificationConfig(name, notifier_type, enabled, entry))

    if not notifications:
        notifications.append(NotificationConfig("console", "console"))

    config = MonitorConfig(
        project=project,
        dpgen=dpgen,
        evaluation=evaluation,
        notifications=tuple(notifications),
        committee_replay=committee_replay,
        parameter_proposals=parameter_proposals,
    )
    validate_config(config)
    return config


def validate_config(config: MonitorConfig) -> None:
    if not config.project.run_dir.is_dir():
        raise ValueError(f"DP-GEN 运行目录不存在: {config.project.run_dir}")
    if config.evaluation.enabled:
        if config.evaluation.backend != "deepmd":
            raise ValueError(f"尚未注册评估后端: {config.evaluation.backend}")
        if not config.evaluation.command:
            raise ValueError("evaluation.command 不能为空")
        if not config.evaluation.model_ids:
            raise ValueError("evaluation.model_ids 不能为空")
    if config.committee_replay.enabled:
        if config.committee_replay.executor_profile != "local_gpu":
            raise ValueError(
                "committee_replay.executor_profile 当前仅支持 local_gpu"
            )
        if not config.committee_replay.command:
            raise ValueError("committee_replay.command 不能为空")
        if len(config.committee_replay.model_ids) < 2:
            raise ValueError("committee_replay.model_ids 至少需要两个模型")
        if len(set(config.committee_replay.model_ids)) != len(
            config.committee_replay.model_ids
        ):
            raise ValueError("committee_replay.model_ids 不能重复")
        if "{model_id}" not in config.committee_replay.model_pattern:
            raise ValueError("committee_replay.model_pattern 必须包含 {model_id}")
        try:
            rendered_models = [
                Path(
                    config.committee_replay.model_pattern.format(
                        model_id=model_id
                    )
                )
                for model_id in config.committee_replay.model_ids
            ]
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "committee_replay.model_pattern 包含无效格式字段"
            ) from exc
        if any(
            model.is_absolute() or ".." in model.parts or model == Path(".")
            for model in rendered_models
        ):
            raise ValueError(
                "committee_replay.model_pattern 和 model_ids 必须生成 "
                "00.train 下的安全相对文件路径"
            )
        candidate_manifest = Path(config.committee_replay.candidate_manifest)
        if candidate_manifest.is_absolute() or ".." in candidate_manifest.parts:
            raise ValueError(
                "committee_replay.candidate_manifest 必须是迭代目录下的安全相对路径"
            )
        if (
            config.committee_replay.relative is not None
            and config.committee_replay.relative <= 0
        ):
            raise ValueError("committee_replay.relative 必须大于 0 或设为 null")
        if config.committee_replay.cpu_per_node > 64:
            raise ValueError("committee_replay.cpu_per_node 不能超过 64")
        if (
            config.committee_replay.max_total_frames
            < config.committee_replay.max_frames_per_task
        ):
            raise ValueError(
                "committee_replay.max_total_frames 不能小于 "
                "max_frames_per_task"
            )
    if config.parameter_proposals.enabled:
        proposal = config.parameter_proposals
        if not config.committee_replay.enabled:
            raise ValueError(
                "parameter_proposals 需要先启用 committee_replay"
            )
        if proposal.parameter_file is None:
            raise ValueError("parameter_proposals.parameter_file 不能为空")
        if proposal.parameter_file.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("parameter_proposals 当前只支持 YAML 参数文件")
        try:
            proposal.parameter_file.resolve().relative_to(
                config.project.run_dir.resolve()
            )
        except ValueError as exc:
            raise ValueError(
                "parameter_proposals.parameter_file 必须位于 DP-GEN run_dir"
            ) from exc
        if not proposal.parameter_file.is_file():
            raise ValueError(
                f"DP-GEN 参数文件不存在: {proposal.parameter_file}"
            )
        if proposal.strategy != "repeat_last":
            raise ValueError(
                "parameter_proposals.strategy 当前仅支持 repeat_last"
            )
        if proposal.required_task != config.committee_replay.ready_task:
            raise ValueError(
                "parameter_proposals.required_task 必须与 "
                "committee_replay.ready_task 一致"
            )
        if proposal.start_iteration < config.committee_replay.start_iteration:
            raise ValueError(
                "parameter_proposals.start_iteration 不能早于委员会回放"
            )
        if config.committee_replay.source_offsets != (1,):
            raise ValueError(
                "parameter_proposals 只允许 committee_replay.source_offsets = [1]；"
                "参数建议必须只使用上一轮轨迹"
            )

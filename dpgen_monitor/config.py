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

    config = MonitorConfig(project, dpgen, evaluation, tuple(notifications))
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

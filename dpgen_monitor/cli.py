from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading

from . import __version__
from .config import load_config
from .dpgen import DpgenObserver
from .service import MonitorService
from .state import StateStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dpgen-monitor",
        description="通用 DP-GEN 流程监控、模型评估与通知工具",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("check", "只读检查配置和 DP-GEN 目录，不评估、不发送"),
        ("sync", "同步阶段和统计到状态库，不评估、不发送"),
        ("once", "执行一次完整扫描、评估和通知"),
        ("run", "持续监控"),
        ("status", "查看 SQLite 持久化状态摘要"),
    ):
        item = subparsers.add_parser(command, help=help_text)
        item.add_argument("config", type=Path, help="TOML/YAML 配置文件")
    return parser


def command_check(config_path: Path) -> int:
    config = load_config(config_path)
    observer = DpgenObserver(config.project.run_dir, config.dpgen)
    snapshots, inspections = observer.scan()
    ready = sum(item.status == "ready" for item in inspections.values())
    latest = snapshots[-1] if snapshots else None
    print(f"配置有效: {config_path.resolve()}")
    print(f"运行目录: {config.project.run_dir}")
    print(f"输出目录: {config.project.output_dir}")
    print(f"发现迭代: {len(snapshots)}；含完整探索统计: {ready}")
    if latest:
        print(
            f"最新迭代: iter.{latest.iteration:06d}，"
            f"task={latest.task if latest.task is not None else '未知'}"
        )
        inspection = inspections.get(latest.iteration)
        if inspection and inspection.status != "ready":
            print(f"统计状态: {inspection.message(config.project.check_interval)}")
    print("检查模式未运行 dp test，也未发送任何通知。")
    return 0


def command_once(config_path: Path) -> int:
    service = MonitorService(load_config(config_path))
    try:
        service.scan_once(evaluate=True, notify=True)
    finally:
        service.close()
    return 0


def command_sync(config_path: Path) -> int:
    service = MonitorService(load_config(config_path), enable_notifiers=False)
    try:
        summary = service.scan_once(evaluate=False, notify=False)
        print(
            "状态同步完成："
            f"迭代 {summary['iterations']}，"
            f"完整探索统计 {summary['statistics_ready']}"
        )
    finally:
        service.close()
    return 0


def command_run(config_path: Path) -> int:
    service = MonitorService(load_config(config_path))
    stop_event = threading.Event()

    def request_stop(_signum, _frame):
        print("\n收到停止请求，正在安全退出...")
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        service.run(stop_event)
    finally:
        service.close()
    return 0


def command_status(config_path: Path) -> int:
    config = load_config(config_path)
    state_path = config.project.output_dir / "monitor.sqlite3"
    if not state_path.exists():
        print(f"尚未创建状态库: {state_path}")
        return 0
    store = StateStore(state_path)
    try:
        print(json.dumps(store.status_summary(), ensure_ascii=False, indent=2))
    finally:
        store.close()
    return 0


def main() -> int:
    args = build_parser().parse_args()
    commands = {
        "check": command_check,
        "sync": command_sync,
        "once": command_once,
        "run": command_run,
        "status": command_status,
    }
    try:
        return commands[args.command](args.config)
    except (ValueError, RuntimeError) as exc:
        print(f"[配置错误] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

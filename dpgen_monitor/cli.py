from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
from typing import Iterator

from . import __version__
from .config import load_config
from .dpgen import DpgenObserver
from .proposals import ParameterFileController
from .service import MonitorService
from .state import StateStore


@contextmanager
def _monitor_lock(output_dir: Path) -> Iterator[Path]:
    """Prevent concurrent evaluators from sharing one state/output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "monitor.run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "未知进程"
            raise RuntimeError(
                f"已有监控/评估进程正在运行（{owner}）；"
                "请等待其显示‘程序已安全停止’后再启动"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield lock_path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


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
        ("tui", "交互式只读总览、事件与参数设置"),
        ("proposals", "查看 model_devi 参数建议"),
    ):
        item = subparsers.add_parser(command, help=help_text)
        item.add_argument("config", type=Path, help="TOML/YAML 配置文件")
    for command, help_text in (
        ("approve", "批准一个参数建议，但不修改 DP-GEN 参数"),
        ("reject", "拒绝一个参数建议"),
        ("apply", "将已批准建议安全追加到 model_devi_jobs"),
    ):
        item = subparsers.add_parser(command, help=help_text)
        item.add_argument("config", type=Path, help="TOML/YAML 配置文件")
        item.add_argument("proposal_id", help="参数建议 ID")
        if command in {"approve", "reject"}:
            item.add_argument("--note", default=None, help="审批备注")
    return parser


def command_check(config_path: Path) -> int:
    config = load_config(config_path)
    observer = DpgenObserver(config.project.run_dir, config.dpgen)
    snapshots, inspections = observer.scan()
    active_iterations = {snapshot.iteration for snapshot in snapshots}
    ready = sum(
        item.status == "ready"
        for iteration, item in inspections.items()
        if iteration in active_iterations
    )
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
    config = load_config(config_path)
    with _monitor_lock(config.project.output_dir):
        service = MonitorService(config)
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
    config = load_config(config_path)
    stop_event = threading.Event()

    def request_stop(_signum, _frame):
        if not stop_event.is_set():
            print("\n收到停止请求，正在取消当前评估并安全退出...")
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    with _monitor_lock(config.project.output_dir):
        service = MonitorService(config)
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


def command_tui(config_path: Path) -> int:
    from .tui import run_tui

    return run_tui(config_path)


def command_proposals(config_path: Path) -> int:
    config = load_config(config_path)
    state_path = config.project.output_dir / "monitor.sqlite3"
    if not state_path.exists():
        print("[]")
        return 0
    store = StateStore(state_path)
    try:
        print(
            json.dumps(
                store.list_parameter_proposals(),
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        store.close()
    return 0


def _transition_proposal(
    config_path: Path,
    proposal_id: str,
    *,
    status: str,
    note: str | None,
) -> int:
    config = load_config(config_path)
    if not config.parameter_proposals.enabled:
        raise ValueError("parameter_proposals 未启用")
    store = StateStore(config.project.output_dir / "monitor.sqlite3")
    try:
        expected = ("pending",) if status == "approved" else ("pending", "approved")
        proposal = store.transition_parameter_proposal(
            proposal_id,
            expected_statuses=expected,
            status=status,
            review_note=note,
        )
        print(
            f"参数建议 {proposal['proposal_id']} 已{('批准' if status == 'approved' else '拒绝')}；"
            "尚未修改 DP-GEN 参数，也未启动 DP-GEN。"
        )
    finally:
        store.close()
    return 0


def command_approve(config_path: Path, proposal_id: str, note: str | None) -> int:
    return _transition_proposal(
        config_path,
        proposal_id,
        status="approved",
        note=note,
    )


def command_reject(config_path: Path, proposal_id: str, note: str | None) -> int:
    return _transition_proposal(
        config_path,
        proposal_id,
        status="rejected",
        note=note,
    )


def command_apply(config_path: Path, proposal_id: str) -> int:
    config = load_config(config_path)
    if not config.parameter_proposals.enabled:
        raise ValueError("parameter_proposals 未启用")
    store = StateStore(config.project.output_dir / "monitor.sqlite3")
    try:
        controller = ParameterFileController(
            config.parameter_proposals,
            config.dpgen,
            config.project.run_dir,
            store,
        )
        parameter_file, backup, changed = controller.apply(proposal_id)
        if changed:
            print(f"已追加 model_devi_jobs: {parameter_file}")
            print(f"原文件备份: {backup}")
        else:
            print(f"参数建议已经应用，无需重复修改: {parameter_file}")
        print("未启动 DP-GEN；请检查参数后再由人工恢复流程。")
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
        "tui": command_tui,
        "proposals": command_proposals,
    }
    try:
        if args.command == "approve":
            return command_approve(args.config, args.proposal_id, args.note)
        if args.command == "reject":
            return command_reject(args.config, args.proposal_id, args.note)
        if args.command == "apply":
            return command_apply(args.config, args.proposal_id)
        return commands[args.command](args.config)
    except (ValueError, RuntimeError) as exc:
        print(f"[配置错误] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

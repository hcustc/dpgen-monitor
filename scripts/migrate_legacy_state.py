#!/usr/bin/env python3
"""Import completion markers from dptest_feishu into dpgen-monitor SQLite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dpgen_monitor.evaluation import is_valid_force_file
from dpgen_monitor.state import StateStore


def read_iterations(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    return {
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().isdigit()
    }


def force_path(legacy_dir: Path, iteration: int, phase: str, model_id: str) -> Path:
    iteration_dir = legacy_dir / f"iter.{iteration:06d}"
    phased = iteration_dir / phase / model_id / f"{model_id}.f.out"
    if phased.is_file():
        return phased
    return iteration_dir / model_id / f"{model_id}.f.out"


def migrate(
    legacy_dir: Path,
    state_path: Path,
    notifier: str,
    model_ids: tuple[str, ...],
) -> dict[str, int]:
    store = StateStore(state_path)
    imported = {"statistics": 0, "absorption": 0, "blind_spot": 0, "models": 0}
    try:
        statistics = read_iterations(legacy_dir / "processed_stats.log")
        old_absorption = read_iterations(legacy_dir / "processed_models.log")
        absorption = old_absorption | read_iterations(
            legacy_dir / "processed_absorption_evaluations.log"
        )
        blind_spot = read_iterations(
            legacy_dir / "processed_blind_spot_evaluations.log"
        )

        for iteration in sorted(statistics):
            event_key = f"statistics:iter.{iteration:06d}"
            if not store.is_delivered(event_key, notifier):
                store.record_delivery(event_key, notifier, True)
            imported["statistics"] += 1

        for phase, iterations in (
            ("absorption", absorption),
            ("blind_spot", blind_spot),
        ):
            for iteration in sorted(iterations):
                event_key = f"evaluation:{phase}:iter.{iteration:06d}"
                if not store.is_delivered(event_key, notifier):
                    store.record_delivery(event_key, notifier, True)
                imported[phase] += 1
                for model_id in model_ids:
                    path = force_path(legacy_dir, iteration, phase, model_id)
                    if not is_valid_force_file(path):
                        continue
                    store.set_evaluation(
                        iteration,
                        phase,
                        model_id,
                        "complete",
                        str(path.resolve()),
                    )
                    imported["models"] += 1
    finally:
        store.close()
    return imported


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_dir", type=Path)
    parser.add_argument("state_path", type=Path)
    parser.add_argument("--notifier", default="feishu")
    parser.add_argument(
        "--model-ids", nargs="+", default=["000", "001", "002", "003"]
    )
    args = parser.parse_args()
    result = migrate(
        args.legacy_dir.resolve(),
        args.state_path.resolve(),
        args.notifier,
        tuple(args.model_ids),
    )
    print(
        "迁移完成："
        f"统计通知 {result['statistics']}，"
        f"吸收评估通知 {result['absorption']}，"
        f"盲区评估通知 {result['blind_spot']}，"
        f"有效模型结果 {result['models']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

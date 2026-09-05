import json
from pathlib import Path
import sqlite3
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dpgen_monitor.committee_replay import CommitteeReplayResult
from dpgen_monitor.config import DpgenConfig, EvaluationConfig, load_config
from dpgen_monitor.cli import _monitor_lock, build_parser
from dpgen_monitor.dpgen import DpgenObserver, IterationSnapshot
from dpgen_monitor.evaluation import DeepMDEvaluator, EvaluationResult
from dpgen_monitor.evaluation_data import (
    current_iteration_fp_evaluation_description,
    resolve_current_iteration_fp_data,
    resolve_evaluation_data,
)
from dpgen_monitor.evaluation_plots import (
    best_force_model,
    load_force_parity,
    render_absorption_gain,
    render_force_density_comparison,
)
from dpgen_monitor.events import MonitorEvent
from dpgen_monitor.render import format_percentage
from dpgen_monitor.service import MonitorService
from dpgen_monitor.state import StateStore
from dpgen_monitor.tui import (
    config_setting_values,
    display_width,
    fit_cells,
    save_toml_settings,
)


LOG = """2026-01-01 - INFO : =====iter.000000=====\n
2026-01-01 - INFO : -----iter.000000 task 06-----\n
2026-01-01 - INFO : system 000 candidate : 10 in 100 10.00 %\n
2026-01-01 - INFO : system 000 failed : 5 in 100 5.00 %\n
2026-01-01 - INFO : system 000 accurate : 85 in 100 85.00 %\n
2026-01-01 - INFO : =====iter.000001=====\n
2026-01-01 - INFO : -----iter.000001 task 04-----\n
"""


class ConfigTests(unittest.TestCase):
    def test_monitor_lock_rejects_a_second_evaluator(self):
        with tempfile.TemporaryDirectory() as root:
            output_dir = Path(root)
            with _monitor_lock(output_dir):
                with self.assertRaisesRegex(RuntimeError, "已有监控/评估进程"):
                    with _monitor_lock(output_dir):
                        self.fail("第二个评估器不应获得锁")

    def test_tui_cell_layout_handles_chinese_and_clipping(self):
        self.assertEqual(display_width("总览 A"), 6)
        self.assertEqual(display_width(fit_cells("总览", 8)), 8)
        self.assertEqual(fit_cells("generation", 6), "gener…")

    def test_missing_config_has_a_concise_error(self):
        with tempfile.TemporaryDirectory() as root:
            missing = Path(root) / "missing.toml"
            with self.assertRaisesRegex(ValueError, "配置文件不存在"):
                load_config(missing)

    def test_cli_accepts_tui_command(self):
        args = build_parser().parse_args(["tui", "configs/local.toml"])
        self.assertEqual(args.command, "tui")

    def test_tui_settings_are_saved_atomically_without_touching_secrets(self):
        source = """[project]
run_dir = "run"
output_dir = "output"
check_interval = 600 # polling
heartbeat_interval = 21600

[dpgen]
statistics_task = 6
statistics_start_iteration = 25

[evaluation]
start_iteration = 30
absorption_ready_task = 2
blind_spot_ready_task = 8

[[notifications]]
name = "feishu"
type = "feishu"
bot_url_env = "DO_NOT_CHANGE_ME"
"""
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            (root_path / "run").mkdir()
            path = root_path / "local.toml"
            path.write_text(source, encoding="utf-8")
            config = load_config(path)
            values = config_setting_values(config)
            values["project.check_interval"] = 120
            values["evaluation.start_iteration"] = 27

            backup = save_toml_settings(path, values)

            rendered = path.read_text(encoding="utf-8")
            self.assertIn("check_interval = 120 # polling", rendered)
            self.assertIn("start_iteration = 27", rendered)
            self.assertIn('bot_url_env = "DO_NOT_CHANGE_ME"', rendered)
            self.assertEqual(backup.read_text(encoding="utf-8"), source)
            self.assertEqual(load_config(path).project.check_interval, 120)

    def test_tui_settings_insert_fields_that_previously_used_defaults(self):
        source = """[project]
run_dir = "run"
output_dir = "output"

[[notifications]]
name = "feishu"
type = "feishu"
bot_url_env = "DO_NOT_CHANGE_ME"
"""
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            (root_path / "run").mkdir()
            path = root_path / "minimal.toml"
            path.write_text(source, encoding="utf-8")
            values = config_setting_values(load_config(path))
            values["project.check_interval"] = 120

            backup = save_toml_settings(path, values)

            rendered = path.read_text(encoding="utf-8")
            self.assertIn("check_interval = 120", rendered)
            self.assertIn("heartbeat_interval = 21600", rendered)
            self.assertIn("[dpgen]", rendered)
            self.assertIn("[evaluation]", rendered)
            self.assertIn('bot_url_env = "DO_NOT_CHANGE_ME"', rendered)
            self.assertEqual(backup.read_text(encoding="utf-8"), source)
            self.assertEqual(load_config(path).project.check_interval, 120)


class ObserverTests(unittest.TestCase):
    def test_ready_and_pending_statistics_are_distinguished(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root)
            (run_dir / "iter.000000" / "00.train").mkdir(parents=True)
            (run_dir / "iter.000001" / "00.train").mkdir(parents=True)
            (run_dir / "dpgen.log").write_text(LOG, encoding="utf-8")
            (run_dir / "record.dpgen").write_text("0 6\n1 3\n", encoding="utf-8")

            snapshots, inspections = DpgenObserver(run_dir, DpgenConfig()).scan()

            self.assertEqual([item.iteration for item in snapshots], [0, 1])
            self.assertEqual(inspections[0].status, "ready")
            self.assertEqual(inspections[0].stats["candidate_count"], 10)
            self.assertEqual(inspections[1].status, "pending")
            self.assertEqual(inspections[1].task, 4)
            self.assertIn("task 06", inspections[1].message(300))

    def test_percentages_are_recomputed_from_exact_counts(self):
        log = """=====iter.000027=====\n
-----iter.000027 task 06-----\n
system 000 candidate : 3316 in 265000000 0.00 %\n
system 000 failed : 202 in 265000000 0.00 %\n
system 000 accurate : 264996482 in 265000000 100.00 %\n
"""
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root)
            (run_dir / "dpgen.log").write_text(log, encoding="utf-8")
            _, inspections = DpgenObserver(run_dir, DpgenConfig()).scan()
            stats = inspections[27].stats
            self.assertAlmostEqual(
                stats["candidate_percent"], 3316 / 265_000_000 * 100
            )
            self.assertEqual(format_percentage(stats["candidate_percent"]), "0.0013%")
            self.assertEqual(format_percentage(stats["failed_percent"]), "0.0001%")
            self.assertEqual(format_percentage(stats["accurate_percent"]), "99.9987%")

    def test_reused_iteration_uses_latest_stage_and_statistics_generation(self):
        log = """=====iter.000027=====\n
-----iter.000027 task 08-----\n
system 000 candidate : 10 in 100 10.00 %\n
system 000 failed : 5 in 100 5.00 %\n
system 000 accurate : 85 in 100 85.00 %\n
=====iter.000030=====\n
-----iter.000030 task 04-----\n
=====iter.000027=====\n
-----iter.000027 task 03-----\n
-----iter.000027 task 04-----\n
"""
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root)
            (run_dir / "iter.000027" / "00.train").mkdir(parents=True)
            (run_dir / "dpgen.log").write_text(log, encoding="utf-8")
            (run_dir / "record.dpgen").write_text(
                "27 8\n27 2\n27 3\n", encoding="utf-8"
            )

            snapshots, inspections = DpgenObserver(run_dir, DpgenConfig()).scan()

            self.assertEqual(snapshots[0].task, 4)
            self.assertEqual(inspections[27].status, "pending")
            self.assertEqual(inspections[27].task, 4)


class StateTests(unittest.TestCase):
    def test_delivery_content_hash_suppresses_only_identical_content(self):
        with tempfile.TemporaryDirectory() as root:
            store = StateStore(Path(root) / "state.sqlite3")
            try:
                store.record_delivery(
                    "event", "channel", True, content_hash="content-a"
                )
                self.assertTrue(
                    store.has_delivered_content("event", "channel", "content-a")
                )
                self.assertFalse(
                    store.has_delivered_content("event", "channel", "content-b")
                )
            finally:
                store.close()

    def test_legacy_delivery_is_adopted_without_resending(self):
        with tempfile.TemporaryDirectory() as root:
            store = StateStore(Path(root) / "state.sqlite3")
            try:
                store.record_delivery("event", "channel", True)
                self.assertTrue(
                    store.has_delivered_content("event", "channel", "first-hash")
                )
                self.assertFalse(
                    store.has_delivered_content("event", "channel", "changed-hash")
                )
            finally:
                store.close()

    def test_failed_delivery_is_retryable(self):
        with tempfile.TemporaryDirectory() as root:
            store = StateStore(Path(root) / "state.sqlite3")
            try:
                store.record_delivery("event", "channel", False, "temporary")
                self.assertFalse(store.is_delivered("event", "channel"))
                store.record_delivery("event", "channel", True)
                self.assertTrue(store.is_delivered("event", "channel"))
            finally:
                store.close()

    def test_evaluation_phases_are_stored_independently(self):
        with tempfile.TemporaryDirectory() as root:
            store = StateStore(Path(root) / "state.sqlite3")
            try:
                store.set_evaluation(30, "absorption", "000", "complete", "a")
                store.set_evaluation(30, "blind_spot", "000", "waiting")
                self.assertEqual(
                    store.get_evaluation(30, "absorption", "000")["force_file"],
                    "a",
                )
                self.assertEqual(
                    store.get_evaluation(30, "blind_spot", "000")["status"],
                    "waiting",
                )
            finally:
                store.close()

    def test_legacy_evaluation_schema_is_migrated_to_absorption_phase(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "state.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute(
                """
                CREATE TABLE evaluations (
                    iteration INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    force_file TEXT,
                    last_error TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (iteration, model_id)
                )
                """
            )
            connection.execute(
                "INSERT INTO evaluations VALUES (7, '000', 'complete', 'f', NULL, 'now')"
            )
            connection.commit()
            connection.close()

            store = StateStore(path)
            try:
                row = store.get_evaluation(7, "absorption", "000")
                self.assertEqual(row["status"], "complete")
            finally:
                store.close()

    def test_stage_regression_invalidates_superseded_iteration_results(self):
        with tempfile.TemporaryDirectory() as root:
            store = StateStore(Path(root) / "state.sqlite3")
            try:
                store.upsert_stage(27, 8)
                store.upsert_statistics(27, {"candidate_count": 1})
                store.set_evaluation(27, "absorption", "000", "complete", "old")
                store.record_delivery("statistics:iter.000027", "feishu", True)
                snapshot = IterationSnapshot(
                    27,
                    Path(root) / "iter.000027",
                    None,
                    4,
                    source_identity="1:2",
                )

                generations, changes = store.reconcile_iterations([snapshot])

                self.assertEqual(generations[27], 1)
                self.assertTrue(any("回退" in change for change in changes))
                self.assertTrue(
                    store.is_delivered("statistics:iter.000027", "feishu")
                )
                self.assertIsNone(store.get_evaluation(27, "absorption", "000"))
                self.assertEqual(store.list_statistics(), [])
            finally:
                store.close()

    def test_removed_later_iterations_are_deactivated(self):
        with tempfile.TemporaryDirectory() as root:
            store = StateStore(Path(root) / "state.sqlite3")
            try:
                for iteration in (27, 28, 29, 30):
                    store.upsert_stage(iteration, 8)
                    store.record_delivery(
                        f"statistics:iter.{iteration:06d}", "feishu", True
                    )
                snapshot = IterationSnapshot(
                    27,
                    Path(root) / "iter.000027",
                    None,
                    8,
                    source_identity="1:2",
                )

                _, changes = store.reconcile_iterations([snapshot])

                self.assertEqual(store.status_summary()["iterations"], 1)
                self.assertEqual(len(changes), 3)
                self.assertTrue(
                    store.is_delivered("statistics:iter.000030", "feishu")
                )
            finally:
                store.close()


class EvaluatorTests(unittest.TestCase):
    def test_keyboard_interrupt_is_cancelled_and_stops_remaining_models(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            train_dir = root_path / "run" / "iter.000000" / "00.train"
            for model_id in ("000", "001"):
                model = train_dir / model_id / "frozen_model.pb"
                model.parent.mkdir(parents=True)
                model.write_text("model", encoding="utf-8")
            (train_dir / "data.init" / "03.data").mkdir(parents=True)
            state = StateStore(root_path / "output" / "state.sqlite3")
            stop_event = threading.Event()
            calls = []

            def cancelled_run(command, capture_output, text):
                calls.append(command)
                stop_event.set()
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="CondaError: KeyboardInterrupt",
                )

            try:
                evaluator = DeepMDEvaluator(
                    EvaluationConfig(
                        command=("dp",),
                        model_ids=("000", "001"),
                        compare_previous_model=False,
                    ),
                    root_path / "output",
                    state,
                    stop_event,
                )
                snapshot = IterationSnapshot(0, train_dir.parent, train_dir, 2)
                with patch(
                    "dpgen_monitor.evaluation.subprocess.run",
                    side_effect=cancelled_run,
                ):
                    results = evaluator.evaluate_iteration(snapshot)

                self.assertEqual(len(calls), 1)
                self.assertEqual(results[0].status, "cancelled")
                self.assertEqual(
                    state.get_evaluation(0, "absorption", "000")["status"],
                    "cancelled",
                )
            finally:
                state.close()

    def test_cancelled_wrapper_keeps_a_valid_force_output(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            train_dir = root_path / "run" / "iter.000000" / "00.train"
            model = train_dir / "000" / "frozen_model.pb"
            model.parent.mkdir(parents=True)
            model.write_text("model", encoding="utf-8")
            (train_dir / "data.init" / "03.data").mkdir(parents=True)
            state = StateStore(root_path / "output" / "state.sqlite3")
            stop_event = threading.Event()

            def interrupted_after_output(command, capture_output, text):
                detail_prefix = Path(command[command.index("-d") + 1])
                Path(f"{detail_prefix}.f.out").write_text(
                    "1 2 3 1 2 3\n", encoding="utf-8"
                )
                stop_event.set()
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="CondaError: KeyboardInterrupt",
                )

            try:
                evaluator = DeepMDEvaluator(
                    EvaluationConfig(
                        command=("dp",),
                        model_ids=("000",),
                        compare_previous_model=False,
                    ),
                    root_path / "output",
                    state,
                    stop_event,
                )
                snapshot = IterationSnapshot(0, train_dir.parent, train_dir, 2)
                with patch(
                    "dpgen_monitor.evaluation.subprocess.run",
                    side_effect=interrupted_after_output,
                ):
                    results = evaluator.evaluate_iteration(snapshot)

                self.assertEqual(results[0].status, "complete")
                self.assertTrue(results[0].force_file.is_file())
            finally:
                state.close()

    def test_current_iteration_fp_data_is_selected_for_blind_spot_evaluation(self):
        train_dir = Path("/project/run/iter.000029/00.train")
        resolved = resolve_current_iteration_fp_data(train_dir, 29)
        self.assertEqual(
            resolved,
            Path("/project/run/iter.000029/02.fp/data.000"),
        )
        self.assertIn(
            "iter.000029 FP 数据",
            current_iteration_fp_evaluation_description(29),
        )

    def test_previous_iteration_fp_data_is_selected(self):
        train_dir = Path("/project/run/iter.000025/00.train")
        resolved = resolve_evaluation_data(train_dir, 25)
        self.assertEqual(
            resolved,
            train_dir / "data.iters" / "iter.000024" / "02.fp" / "data.000",
        )

    def test_iteration_zero_uses_initial_data(self):
        train_dir = Path("/project/run/iter.000000/00.train")
        resolved = resolve_evaluation_data(train_dir, 0)
        self.assertEqual(resolved, train_dir / "data.init" / "03.data")

    def test_literal_test_data_path_remains_supported(self):
        train_dir = Path("/project/run/iter.000025/00.train")
        resolved = resolve_evaluation_data(
            train_dir,
            25,
            test_data="benchmarks/fixed-test",
        )
        self.assertEqual(resolved, train_dir / "benchmarks" / "fixed-test")

    def test_evaluator_passes_previous_iteration_data_to_dp_test(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            train_dir = root_path / "run" / "iter.000007" / "00.train"
            model_path = train_dir / "000" / "frozen_model.pb"
            model_path.parent.mkdir(parents=True)
            model_path.write_text("model", encoding="utf-8")
            previous_model_path = (
                root_path
                / "run"
                / "iter.000006"
                / "00.train"
                / "000"
                / "frozen_model.pb"
            )
            previous_model_path.parent.mkdir(parents=True)
            previous_model_path.write_text("previous model", encoding="utf-8")
            previous_data = (
                train_dir
                / "data.iters"
                / "iter.000006"
                / "02.fp"
                / "data.000"
            )
            previous_data.mkdir(parents=True)

            output_dir = root_path / "output"
            state = StateStore(output_dir / "state.sqlite3")
            commands = []

            def fake_run(command, capture_output, text):
                commands.append(command)
                detail_prefix = Path(command[command.index("-d") + 1])
                Path(f"{detail_prefix}.f.out").write_text(
                    "1 2 3 1 2 3\n", encoding="utf-8"
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            try:
                config = EvaluationConfig(command=("dp",), model_ids=("000",))
                evaluator = DeepMDEvaluator(config, output_dir, state)
                snapshot = IterationSnapshot(7, train_dir.parent, train_dir, 6)

                with patch(
                    "dpgen_monitor.evaluation.subprocess.run", side_effect=fake_run
                ):
                    results = evaluator.evaluate_iteration(snapshot)

                self.assertEqual(results[0].status, "complete")
                self.assertEqual(results[0].test_data, previous_data)
                self.assertIsNotNone(results[0].baseline_force_file)
                self.assertEqual(len(commands), 2)
                self.assertEqual(
                    Path(commands[0][commands[0].index("-s") + 1]),
                    previous_data,
                )
                self.assertEqual(
                    Path(commands[1][commands[1].index("-m") + 1]),
                    previous_model_path,
                )
            finally:
                state.close()

    def test_existing_valid_force_output_is_reused(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            train_dir = root_path / "run" / "iter.000007" / "00.train"
            (train_dir / "data.init" / "03.data").mkdir(parents=True)
            output_dir = root_path / "output"
            state = StateStore(output_dir / "state.sqlite3")
            try:
                config = EvaluationConfig(model_ids=("000",))
                force_file = (
                    output_dir
                    / "evaluations"
                    / "iter.000007"
                    / "absorption"
                    / "000"
                    / "000.f.out"
                )
                force_file.parent.mkdir(parents=True)
                force_file.write_text("1 2 3 1 2 3\n", encoding="utf-8")
                evaluator = DeepMDEvaluator(config, output_dir, state)
                snapshot = IterationSnapshot(7, train_dir.parent, train_dir, 6)

                with patch("dpgen_monitor.evaluation.subprocess.run") as run:
                    results = evaluator.evaluate_iteration(snapshot)

                run.assert_not_called()
                self.assertEqual(results[0].status, "complete")
            finally:
                state.close()

    def test_rebuilt_iteration_uses_generation_specific_output(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            train_dir = root_path / "run" / "iter.000007" / "00.train"
            model_path = train_dir / "000" / "frozen_model.pb"
            model_path.parent.mkdir(parents=True)
            model_path.write_text("model", encoding="utf-8")
            test_data = (
                train_dir / "data.iters" / "iter.000006" / "02.fp" / "data.000"
            )
            test_data.mkdir(parents=True)
            output_dir = root_path / "output"
            state = StateStore(output_dir / "state.sqlite3")
            legacy_force = (
                output_dir
                / "evaluations"
                / "iter.000007"
                / "absorption"
                / "000"
                / "000.f.out"
            )
            legacy_force.parent.mkdir(parents=True)
            legacy_force.write_text("1 2 3 1 2 3\n", encoding="utf-8")

            def fake_run(command, capture_output, text):
                detail_prefix = Path(command[command.index("-d") + 1])
                Path(f"{detail_prefix}.f.out").write_text(
                    "1 2 3 1 2 3\n", encoding="utf-8"
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            try:
                config = EvaluationConfig(
                    command=("dp",), model_ids=("000",), compare_previous_model=False
                )
                evaluator = DeepMDEvaluator(config, output_dir, state)
                snapshot = IterationSnapshot(
                    7, train_dir.parent, train_dir, 6, generation=1
                )

                with patch(
                    "dpgen_monitor.evaluation.subprocess.run", side_effect=fake_run
                ) as run:
                    results = evaluator.evaluate_iteration(snapshot)

                run.assert_called_once()
                self.assertIn("generation.000001", str(results[0].force_file))
            finally:
                state.close()


class EvaluationPlotTests(unittest.TestCase):
    @staticmethod
    def write_force_file(path, offset):
        x = np.linspace(-3.0, 3.0, 60)
        reference = np.column_stack((x, 0.5 * x, -0.25 * x))
        predicted = reference + offset
        np.savetxt(path, np.column_stack((reference, predicted)))

    def test_density_metrics_and_absorption_plot(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            current_path = root_path / "current.f.out"
            previous_path = root_path / "previous.f.out"
            self.write_force_file(current_path, 0.1)
            self.write_force_file(previous_path, 0.4)

            current = load_force_parity("000", current_path)
            previous = load_force_parity("000", previous_path)
            self.assertAlmostEqual(current.mae, 0.1)
            self.assertAlmostEqual(current.rmse, 0.1)
            self.assertEqual(best_force_model([previous, current]), current)

            density_path = render_force_density_comparison(
                [current], root_path / "density.png", iteration=7, bins=20
            )
            gain_path = render_absorption_gain(
                [current], [previous], root_path / "gain.png", iteration=7
            )
            self.assertGreater(density_path.stat().st_size, 0)
            self.assertGreater(gain_path.stat().st_size, 0)


class ServiceLoggingTests(unittest.TestCase):
    def test_changed_committee_replay_content_is_delivered_again(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            summary_file = root_path / "summary.json"
            state = StateStore(root_path / "state.sqlite3")
            service = object.__new__(MonitorService)
            service.config = SimpleNamespace(
                committee_replay=SimpleNamespace(
                    start_iteration=2,
                    source_offsets=(1,),
                    ready_task=2,
                )
            )
            service.state = state
            service.notifiers = [SimpleNamespace(name="channel")]
            service.replay_evaluator = SimpleNamespace(
                evaluate=lambda model, source: self.fail(
                    "a complete replay should be restored without rerunning"
                )
            )
            service._stop_event = None
            service._last_status = {}
            model = IterationSnapshot(
                2,
                Path("/run/iter.000002"),
                Path("/run/iter.000002/00.train"),
                2,
            )
            source = IterationSnapshot(1, Path("/run/iter.000001"), None, 8)
            summary = {
                "frame_count": 1,
                "tracked_candidates": 2,
                "absorbed": 1,
                "remaining_candidate": 1,
                "worsened_to_failed": 0,
                "absorption_percent": 50.0,
                "candidate_percent": 25.0,
                "failed_percent": 0.0,
            }
            try:
                summary_file.write_text(json.dumps(summary), encoding="utf-8")
                old_result = CommitteeReplayResult(
                    2, 1, "complete", summary_file
                )
                old_event = service._committee_replay_event(old_result)
                state.record_delivery(
                    old_event.key,
                    "channel",
                    True,
                    content_hash=old_event.content_hash,
                )
                state.set_committee_replay(2, 1, "complete", str(summary_file))

                with patch.object(service, "_deliver") as unchanged_delivery:
                    self.assertEqual(
                        service._process_committee_replays(
                            [source, model], notify=True
                        ),
                        0,
                    )
                unchanged_delivery.assert_not_called()

                summary["absorbed"] = 2
                summary["remaining_candidate"] = 0
                summary["absorption_percent"] = 100.0
                summary_file.write_text(json.dumps(summary), encoding="utf-8")
                state.set_committee_replay(2, 1, "complete", str(summary_file))

                with patch.object(service, "_deliver") as deliver:
                    completed = service._process_committee_replays(
                        [source, model], notify=True
                    )

                self.assertEqual(completed, 1)
                delivered_event = deliver.call_args.args[0]
                self.assertNotEqual(
                    delivered_event.content_hash, old_event.content_hash
                )
                deliver.assert_called_once_with(delivered_event, True)
            finally:
                state.close()

    def test_cancelled_evaluation_does_not_send_failure_notification(self):
        service = object.__new__(MonitorService)
        service.config = SimpleNamespace(
            evaluation=SimpleNamespace(
                start_iteration=0,
                model_ids=("000",),
            )
        )
        service.state = SimpleNamespace(
            evaluations_complete=lambda iteration, phase, models: False,
            is_delivered=lambda event_key, notifier: False,
        )
        service.notifiers = [SimpleNamespace(name="channel")]
        service.evaluator = SimpleNamespace(
            evaluate_iteration=lambda snapshot, phase: [
                EvaluationResult("000", "cancelled", phase=phase)
            ]
        )
        service._stop_event = None
        service._last_status = {}
        snapshot = IterationSnapshot(
            0, Path("/run/iter.000000"), Path("/run/iter.000000/00.train"), 2
        )

        with (
            patch.object(service, "_phase_readiness", return_value=(True, "")),
            patch.object(service, "_deliver") as deliver,
        ):
            service._process_evaluation_phase(
                [snapshot], "absorption", 2, notify=True
            )

        deliver.assert_not_called()

    def test_event_content_hash_is_stable_and_content_sensitive(self):
        first = MonitorEvent("key", "type", "title", "message", payload={"b": 2})
        same = MonitorEvent("key", "type", "title", "message", payload={"b": 2})
        changed = MonitorEvent(
            "key", "type", "title", "different", payload={"b": 2}
        )
        self.assertEqual(first.content_hash, same.content_hash)
        self.assertNotEqual(first.content_hash, changed.content_hash)

    def test_catch_up_delivers_evaluation_before_statistics(self):
        service = object.__new__(MonitorService)
        snapshot = IterationSnapshot(
            27,
            Path("/run/iter.000027"),
            Path("/run/iter.000027/00.train"),
            7,
            source_identity="1:2",
        )
        inspection = SimpleNamespace(status="ready", stats={"candidate_count": 1})
        service.observer = SimpleNamespace(
            scan=lambda: ([snapshot], {27: inspection})
        )
        service.state = SimpleNamespace(
            reconcile_iterations=lambda snapshots: ({27: 0}, []),
            upsert_stage=lambda iteration, task: None,
            upsert_statistics=lambda iteration, stats: None,
            evaluations_complete=lambda iteration, phase, models: False,
            is_delivered=lambda event_key, notifier: False,
        )
        service.config = SimpleNamespace(
            dpgen=SimpleNamespace(statistics_start_iteration=0),
            evaluation=SimpleNamespace(
                start_iteration=27,
                absorption_ready_task=2,
                blind_spot_enabled=True,
                blind_spot_ready_task=8,
                model_ids=("000",),
            ),
        )
        service.evaluator = SimpleNamespace(
            evaluate_iteration=lambda snapshot, phase: [
                EvaluationResult("000", "complete", phase=phase)
            ]
        )
        service.notifiers = [SimpleNamespace(name="channel")]
        service._last_status = {}
        service._stop_event = None
        delivered: list[str] = []
        statistics_event = MonitorEvent("statistics", "statistics", "s", "s")

        with (
            patch.object(service, "_phase_readiness", return_value=(True, "")),
            patch.object(
                service,
                "_evaluation_event",
                side_effect=lambda snapshot, phase, results: MonitorEvent(
                    phase, "evaluation", "e", "e"
                ),
            ),
            patch.object(service, "_statistics_event", return_value=statistics_event),
            patch.object(
                service,
                "_deliver",
                side_effect=lambda event, notify: delivered.append(event.key),
            ),
        ):
            service.scan_once(evaluate=True, notify=True)

        self.assertEqual(delivered, ["absorption", "statistics", "blind_spot"])

    def test_unchanged_status_is_logged_only_once(self):
        service = object.__new__(MonitorService)
        service._last_status = {}
        with patch("builtins.print") as printer:
            service._log_changed("phase", "same status")
            service._log_changed("phase", "same status")
        printer.assert_called_once_with("same status")


if __name__ == "__main__":
    unittest.main()

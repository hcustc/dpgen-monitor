from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dpgen_monitor.config import DpgenConfig, EvaluationConfig
from dpgen_monitor.dpgen import DpgenObserver, IterationSnapshot
from dpgen_monitor.evaluation import DeepMDEvaluator
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
from dpgen_monitor.render import format_percentage
from dpgen_monitor.service import MonitorService
from dpgen_monitor.state import StateStore


LOG = """2026-01-01 - INFO : =====iter.000000=====\n
2026-01-01 - INFO : -----iter.000000 task 06-----\n
2026-01-01 - INFO : system 000 candidate : 10 in 100 10.00 %\n
2026-01-01 - INFO : system 000 failed : 5 in 100 5.00 %\n
2026-01-01 - INFO : system 000 accurate : 85 in 100 85.00 %\n
2026-01-01 - INFO : =====iter.000001=====\n
2026-01-01 - INFO : -----iter.000001 task 04-----\n
"""


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


class StateTests(unittest.TestCase):
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


class EvaluatorTests(unittest.TestCase):
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
    def test_unchanged_status_is_logged_only_once(self):
        service = object.__new__(MonitorService)
        service._last_status = {}
        with patch("builtins.print") as printer:
            service._log_changed("phase", "same status")
            service._log_changed("phase", "same status")
        printer.assert_called_once_with("same status")


if __name__ == "__main__":
    unittest.main()

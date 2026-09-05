import json
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dpgen_monitor.committee_replay import (
    CommitteeReplayEvaluator,
    CommitteeReplayResult,
    ReplayFrame,
    build_replay_dataset,
    read_lammps_dump_frame,
    select_replay_frames,
    summarize_replay_output,
)
from dpgen_monitor.config import CommitteeReplayConfig, load_config
from dpgen_monitor.dpgen import IterationSnapshot
from dpgen_monitor.execution import (
    CommitteeReplayRequest,
    ExecutionResult,
    ReplaySubmissionController,
)
from dpgen_monitor.service import MonitorService
from dpgen_monitor.state import StateStore


def write_dump(path: Path, step: int, *, shifted: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ITEM: TIMESTEP\n"
        f"{step}\n"
        "ITEM: NUMBER OF ATOMS\n"
        "2\n"
        "ITEM: BOX BOUNDS pp pp pp\n"
        "1 11\n"
        "2 12\n"
        "3 13\n"
        "ITEM: ATOMS id type x y z\n"
        f"2 2 {3 + shifted} 4 5\n"
        f"1 1 {2 + shifted} 3 4\n",
        encoding="utf-8",
    )


def write_candidate_manifest(
    run_dir: Path, source_iteration: int, rows: list[dict]
) -> Path:
    path = (
        run_dir
        / f"iter.{source_iteration:06d}"
        / "02.fp"
        / "candidate_selection.000.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


class CommitteeReplayTests(unittest.TestCase):
    def test_example_places_runner_settings_in_replay_section(self):
        example = Path(__file__).parents[1] / "configs" / "example.toml"
        with example.open("rb") as handle:
            raw = tomllib.load(handle)

        self.assertEqual(
            raw["committee_replay"]["executor_profile"], "local_gpu"
        )
        self.assertNotIn("remote_command", raw["committee_replay"])
        self.assertNotIn("executor_profile", raw["evaluation"])

    def test_selection_excludes_fp_rows_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root)
            source = run_dir / "iter.000001"
            rows = []
            for step in range(0, 1000, 100):
                task = "task.000.000000"
                write_dump(source / "01.model_devi" / task / "traj" / f"{step}.lammpstrj", step)
                rows.append(
                    {
                        "task": f"iter.000001/01.model_devi/{task}",
                        "step": step,
                        "atom_index": step // 100 % 2,
                        "model_deviation": 0.2,
                        "selected": step == 500,
                    }
                )
                if step == 500:
                    rows.append(
                        {
                            "task": f"iter.000001/01.model_devi/{task}",
                            "step": step,
                            "atom_index": 1,
                            "model_deviation": 0.21,
                            "selected": False,
                        }
                    )
            write_candidate_manifest(run_dir, 1, rows)
            config = CommitteeReplayConfig(
                max_frames_per_task=5, time_bins=5, seed=17
            )

            first, _ = select_replay_frames(run_dir, 1, config)
            second, _ = select_replay_frames(run_dir, 1, config)

            self.assertEqual(first, second)
            self.assertEqual(len(first), 5)
            self.assertNotIn(500, [frame.step for frame in first])
            self.assertLess(min(frame.step for frame in first), 200)
            self.assertGreater(max(frame.step for frame in first), 700)
            all_frames, _ = select_replay_frames(
                run_dir,
                1,
                CommitteeReplayConfig(max_frames_per_task=20),
            )
            self.assertNotIn(500, [frame.step for frame in all_frames])

    def test_selection_applies_global_budget_fairly_across_tasks(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root)
            source = run_dir / "iter.000001"
            rows = []
            for task_index in range(3):
                task = f"task.000.{task_index:06d}"
                for step in (0, 100, 200):
                    write_dump(
                        source
                        / "01.model_devi"
                        / task
                        / "traj"
                        / f"{step}.lammpstrj",
                        step,
                    )
                    rows.append(
                        {
                            "task": task,
                            "step": step,
                            "atom_index": 0,
                            "model_deviation": 0.2,
                            "selected": False,
                        }
                    )
            write_candidate_manifest(run_dir, 1, rows)
            config = CommitteeReplayConfig(
                max_frames_per_task=3,
                max_total_frames=5,
                time_bins=3,
                seed=23,
            )

            first, _ = select_replay_frames(run_dir, 1, config)
            second, _ = select_replay_frames(run_dir, 1, config)

            counts = {
                task: sum(frame.task == task for frame in first)
                for task in {frame.task for frame in first}
            }
            self.assertEqual(first, second)
            self.assertEqual(len(first), 5)
            self.assertEqual(sorted(counts.values()), [1, 2, 2])

    def test_selection_rejects_frame_level_candidate_reports(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root)
            write_candidate_manifest(
                run_dir,
                1,
                [
                    {
                        "task": "task.000.000000",
                        "step": 100,
                        "atom_index": None,
                        "model_deviation": 0.2,
                        "selected": False,
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "不支持帧级候选"):
                select_replay_frames(run_dir, 1, CommitteeReplayConfig())

    def test_dump_parser_and_dataset_builder_preserve_atom_order(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            dump = root_path / "traj" / "100.lammpstrj"
            write_dump(dump, 100)
            step, box, coordinates, atom_types = read_lammps_dump_frame(dump)
            self.assertEqual(step, 100)
            np.testing.assert_allclose(box, np.diag([10.0, 10.0, 10.0]))
            np.testing.assert_allclose(coordinates[0], [1.0, 1.0, 1.0])
            np.testing.assert_array_equal(atom_types, [0, 1])

            frame = ReplayFrame(
                "task.000.000000", 100, dump, 2500.0, ((0, 0.2),)
            )
            dataset = root_path / "holdout.deepmd"
            manifest_path = root_path / "holdout.json"
            manifest = build_replay_dataset(
                [frame],
                dataset,
                manifest_path,
                type_map=("C", "H"),
                fingerprint="fingerprint",
            )
            self.assertEqual(manifest["frame_count"], 1)
            self.assertEqual(manifest["atom_count"], 2)
            self.assertEqual(np.load(dataset / "set.000" / "coord.npy").shape, (1, 6))
            self.assertEqual((dataset / "type.raw").read_text().strip(), "0 1")

    def test_summary_tracks_candidate_absorption_and_all_atom_rates(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "model_devi.out"
            output.write_text(
                "# header\n"
                "0 0 0 0 0 0 0 0.10 0.20\n"
                "1 0 0 0 0 0 0 0 0.31 0.16\n",
                encoding="utf-8",
            )
            holdout = {
                "atom_count": 2,
                "frames": [
                    {
                        "task": "task.0",
                        "temperature": 2500,
                        "candidates": [{"atom_index": 0, "old_deviation": 0.2}],
                    },
                    {
                        "task": "task.1",
                        "temperature": 3000,
                        "candidates": [{"atom_index": 0, "old_deviation": 0.2}],
                    },
                ],
            }

            summary = summarize_replay_output(
                output, holdout, f_trust_lo=0.15, f_trust_hi=0.30
            )

            self.assertEqual(summary["tracked_candidates"], 2)
            self.assertEqual(summary["absorbed"], 1)
            self.assertEqual(summary["worsened_to_failed"], 1)
            self.assertEqual(summary["all_atom_candidate"], 2)
            self.assertEqual(summary["all_atom_failed"], 1)
            self.assertAlmostEqual(summary["absorption_percent"], 50.0)

    def test_evaluator_runs_one_committee_command_and_persists_result(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            run_dir = root_path / "run"
            source_dir = run_dir / "iter.000001"
            task = "task.000.000000"
            write_dump(source_dir / "01.model_devi" / task / "traj" / "100.lammpstrj", 100)
            write_candidate_manifest(
                run_dir,
                1,
                [
                    {
                        "task": f"iter.000001/01.model_devi/{task}",
                        "step": 100,
                        "atom_index": 0,
                        "model_deviation": 0.2,
                        "selected": False,
                    }
                ],
            )
            train_dir = run_dir / "iter.000002" / "00.train"
            train_dir.mkdir(parents=True)
            for model_id in ("000", "001"):
                (train_dir / f"graph.{model_id}.pb").write_text("model")
            output_dir = root_path / "output"
            state = StateStore(output_dir / "monitor.sqlite3")
            config = CommitteeReplayConfig(
                enabled=True,
                command=("dp",),
                model_ids=("000", "001"),
                model_pattern="graph.{model_id}.pb",
                max_frames_per_task=10,
                type_map=("C", "H"),
            )
            evaluator = CommitteeReplayEvaluator(config, run_dir, output_dir, state)
            model = IterationSnapshot(2, train_dir.parent, train_dir, 2)
            source = IterationSnapshot(1, source_dir, None, 8)
            requests = []

            def fake_submit(request):
                requests.append(request)
                output = request.deviation_file
                output.write_text("0 0 0 0 0 0 0 0.10 0.20\n", encoding="utf-8")
                return ExecutionResult(("dp", "model-devi"))

            try:
                with patch.object(
                    evaluator.submission_controller,
                    "submit",
                    side_effect=fake_submit,
                ):
                    result = evaluator.evaluate(model, source)

                self.assertEqual(result.status, "complete")
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0].model_iteration, 2)
                self.assertEqual(requests[0].source_iteration, 1)
                self.assertEqual(result.summary["absorbed"], 1)
                row = state.get_committee_replay(2, 1)
                self.assertEqual(row["status"], "complete")
            finally:
                state.close()

    def test_submission_controller_uses_bounded_dpdispatcher_task(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            run_dir = root_path / "run"
            output_root = root_path / "output" / "evaluations"
            work_dir = output_root / "iter.000002" / "committee_replay"
            dataset = work_dir / "holdout.deepmd"
            dataset.mkdir(parents=True)
            fingerprint = "a" * 64
            (work_dir / "holdout_manifest.json").write_text(
                json.dumps({"fingerprint": fingerprint, "frame_count": 16})
            )
            train_dir = run_dir / "iter.000002" / "00.train"
            train_dir.mkdir(parents=True)
            models = tuple(
                train_dir / f"graph.{model_id}.pt2"
                for model_id in ("000", "001")
            )
            for model in models:
                model.write_text("model")
            deviation = work_dir / "model_devi.out"
            config = CommitteeReplayConfig(
                enabled=True,
                command=("dp",),
                model_ids=("000", "001"),
                model_pattern="graph.{model_id}.pt2",
                max_frames_per_task=16,
                max_total_frames=32,
                cpu_per_node=3,
                gpu_device=1,
            )
            controller = ReplaySubmissionController(
                config, run_dir, output_root
            )
            request = CommitteeReplayRequest(
                model_iteration=2,
                source_iteration=1,
                plan_fingerprint=fingerprint,
                work_dir=work_dir,
                dataset=dataset,
                models=models,
                deviation_file=deviation,
                frame_count=16,
                task_count=1,
            )
            observed = {}

            class FakeMachine:
                @staticmethod
                def load_from_dict(value):
                    observed["machine"] = value
                    return value

            class FakeResources:
                def __init__(self, **kwargs):
                    observed["resources"] = kwargs

            class FakeTask:
                def __init__(self, **kwargs):
                    observed["task"] = kwargs

            class FakeSubmission:
                def __init__(self, **kwargs):
                    observed["submission"] = kwargs

                def run_submission(self, **kwargs):
                    observed["run"] = kwargs
                    deviation.write_text("result", encoding="utf-8")

                def check_all_finished(self):
                    return True

            fake_module = SimpleNamespace(
                Machine=FakeMachine,
                Resources=FakeResources,
                Submission=FakeSubmission,
                Task=FakeTask,
            )

            with patch.dict("sys.modules", {"dpdispatcher": fake_module}):
                result = controller.submit(request)

            self.assertIn("model-devi", result.command)
            self.assertNotIn("ssh", result.command)
            self.assertEqual(observed["machine"]["batch_type"], "Shell")
            self.assertEqual(
                observed["machine"]["context_type"], "LazyLocalContext"
            )
            self.assertEqual(observed["resources"]["gpu_per_node"], 1)
            self.assertEqual(
                observed["resources"]["envs"]["CUDA_VISIBLE_DEVICES"], "1"
            )
            self.assertFalse(observed["run"]["clean"])
            self.assertTrue(observed["run"]["exit_on_submit"])
            audit = json.loads(
                (work_dir / "execution_request.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("command", audit)
            self.assertEqual(
                audit["request_id"],
                f"committee-replay-000002-000001-{fingerprint[:20]}",
            )
            self.assertEqual(audit["executor_profile"], "local_gpu")
            self.assertEqual(audit["dispatch_status"], "complete")

    def test_submission_controller_returns_without_waiting_for_running_job(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            run_dir = root_path / "run"
            output_root = root_path / "output" / "evaluations"
            work_dir = output_root / "work"
            dataset = work_dir / "holdout.deepmd"
            dataset.mkdir(parents=True)
            fingerprint = "b" * 64
            (work_dir / "holdout_manifest.json").write_text(
                json.dumps({"fingerprint": fingerprint, "frame_count": 1})
            )
            train_dir = run_dir / "iter.000002" / "00.train"
            train_dir.mkdir(parents=True)
            models = tuple(
                train_dir / f"graph.{model_id}.pt2"
                for model_id in ("000", "001")
            )
            for model in models:
                model.write_text("model")
            config = CommitteeReplayConfig(
                enabled=True,
                command=("dp",),
                model_ids=("000", "001"),
                model_pattern="graph.{model_id}.pt2",
            )
            controller = ReplaySubmissionController(config, run_dir, output_root)
            request = CommitteeReplayRequest(
                2,
                1,
                fingerprint,
                work_dir,
                dataset,
                models,
                work_dir / "model_devi.out",
                1,
                1,
            )

            class FakeMachine:
                @staticmethod
                def load_from_dict(value):
                    return value

            class FakeResources:
                def __init__(self, **kwargs):
                    pass

            class FakeTask:
                def __init__(self, **kwargs):
                    pass

            class FakeSubmission:
                def __init__(self, **kwargs):
                    pass

                def run_submission(self, **kwargs):
                    self.exit_on_submit = kwargs["exit_on_submit"]

                def check_all_finished(self):
                    return False

            fake_module = SimpleNamespace(
                Machine=FakeMachine,
                Resources=FakeResources,
                Submission=FakeSubmission,
                Task=FakeTask,
            )
            with patch.dict("sys.modules", {"dpdispatcher": fake_module}):
                result = controller.submit(request)

            self.assertEqual(result.status, "running")
            audit = json.loads(
                (work_dir / "execution_request.json").read_text(encoding="utf-8")
            )
            self.assertEqual(audit["dispatch_status"], "running")

    def test_submission_controller_rejects_unbounded_intent(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            run_dir = root_path / "run"
            output_root = root_path / "output" / "evaluations"
            work_dir = output_root / "work"
            dataset = work_dir / "holdout.deepmd"
            dataset.mkdir(parents=True)
            fingerprint = "c" * 64
            (work_dir / "holdout_manifest.json").write_text(
                json.dumps({"fingerprint": fingerprint, "frame_count": 9})
            )
            train_dir = run_dir / "iter.000002" / "00.train"
            train_dir.mkdir(parents=True)
            expected = train_dir / "graph.000.pt2"
            expected.write_text("model")
            outside = root_path / "untrusted.pt2"
            outside.write_text("model")
            config = CommitteeReplayConfig(
                enabled=True,
                model_ids=("000", "001"),
                model_pattern="graph.{model_id}.pt2",
                max_frames_per_task=8,
                max_total_frames=8,
            )
            controller = ReplaySubmissionController(
                config, run_dir, output_root
            )
            request = CommitteeReplayRequest(
                2,
                1,
                fingerprint,
                work_dir,
                dataset,
                (expected, outside),
                work_dir / "model_devi.out",
                9,
                1,
            )

            with self.assertRaisesRegex(ValueError, "超过预算"):
                controller.validate(request)

            wrong_models_request = CommitteeReplayRequest(
                2,
                1,
                fingerprint,
                work_dir,
                dataset,
                (expected, outside),
                work_dir / "model_devi.out",
                8,
                1,
            )
            with self.assertRaisesRegex(ValueError, "模型清单"):
                controller.validate(wrong_models_request)

    def test_model_ids_cannot_escape_iteration_train_directory(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            run_dir = root_path / "run"
            run_dir.mkdir()
            config_path = root_path / "monitor.toml"
            config_path.write_text(
                "[project]\n"
                'run_dir = "run"\n'
                'output_dir = "output"\n'
                "[committee_replay]\n"
                "enabled = true\n"
                'command = ["dp"]\n'
                f'model_ids = ["000", "{root_path / "outside.pt2"}"]\n'
                'model_pattern = "{model_id}"\n'
                "max_frames_per_task = 1\n"
                "max_total_frames = 1\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "安全相对文件路径"):
                load_config(config_path)

    def test_config_and_event_are_separate_from_per_model_evaluation(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            (root_path / "run").mkdir()
            config_path = root_path / "monitor.toml"
            config_path.write_text(
                "[project]\n"
                'run_dir = "run"\n'
                'output_dir = "output"\n'
                "[committee_replay]\n"
                "enabled = true\n"
                'command = ["dp"]\n'
                'model_ids = ["000", "001"]\n'
                'model_pattern = "graph.{model_id}.pt2"\n'
                "source_offsets = [1, 2]\n"
                "max_frames_per_task = 12\n",
                encoding="utf-8",
            )
            config = load_config(config_path)
            self.assertTrue(config.committee_replay.enabled)
            self.assertEqual(config.committee_replay.executor_profile, "local_gpu")
            self.assertEqual(config.committee_replay.source_offsets, (1, 2))
            self.assertEqual(config.committee_replay.max_frames_per_task, 12)

            summary_file = root_path / "summary.json"
            summary_file.write_text(
                json.dumps(
                    {
                        "frame_count": 10,
                        "tracked_candidates": 20,
                        "absorbed": 15,
                        "remaining_candidate": 4,
                        "worsened_to_failed": 1,
                        "absorption_percent": 75.0,
                        "candidate_percent": 0.2,
                        "failed_percent": 0.01,
                    }
                )
            )
            service = object.__new__(MonitorService)
            event = service._committee_replay_event(
                CommitteeReplayResult(2, 1, "complete", summary_file)
            )
            self.assertEqual(event.event_type, "committee_replay_ready")
            self.assertIn("75.00%", event.message)
            self.assertIn("source.iter.000001", event.key)

    def test_legacy_raw_ssh_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            (root_path / "run").mkdir()
            config_path = root_path / "monitor.toml"
            config_path.write_text(
                "[project]\n"
                'run_dir = "run"\n'
                'output_dir = "output"\n'
                "[committee_replay]\n"
                "enabled = true\n"
                'execution = "ssh"\n'
                'remote_command = ["srun", "dp"]\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "已移除原始 SSH runner"):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()

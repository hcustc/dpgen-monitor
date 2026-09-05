import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from dpgen_monitor.cli import build_parser
from dpgen_monitor.config import (
    CommitteeReplayConfig,
    DpgenConfig,
    ParameterProposalConfig,
    load_config,
)
from dpgen_monitor.dpgen import IterationSnapshot
from dpgen_monitor.proposals import ParameterFileController, ParameterProposalAdvisor
from dpgen_monitor.state import StateStore


PARAMETER_TEXT = """# preserve this header
model_devi_jobs:
  - sys_idx: [0]
    temps: [2500, 3000]
    trj_freq: 1000
    nsteps: 1000000
    ensemble: nvt
    _idx: 0
  - sys_idx: [0]
    temps: [2500, 3000]
    trj_freq: 1000
    nsteps: 2000000
    ensemble: nvt
    _idx: 1

  # Stage gate: append only after review.

fp_style: gaussian
fp_task_max: 10
"""


def write_replay_summary(
    output_dir: Path,
    state: StateStore,
    model_iteration: int,
    source_iteration: int,
    *,
    model_generation: int = 0,
    source_generation: int = 0,
) -> None:
    path = (
        output_dir
        / "evaluations"
        / f"iter.{model_iteration:06d}"
    )
    if model_generation > 0:
        path /= f"generation.{model_generation:06d}"
    path /= Path("committee_replay") / f"source.iter.{source_iteration:06d}"
    if source_generation > 0:
        path /= f"generation.{source_generation:06d}"
    path /= "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "model_iteration": model_iteration,
                "source_iteration": source_iteration,
                "model_generation": model_generation,
                "source_generation": source_generation,
                "fingerprint": f"fingerprint-{source_iteration}",
                "frame_count": 10,
                "tracked_candidates": 20,
                "absorption_percent": 95.0,
                "candidate_percent": 0.01,
                "failed_percent": 0.001,
            }
        ),
        encoding="utf-8",
    )
    state.set_committee_replay(
        model_iteration,
        source_iteration,
        "complete",
        str(path),
    )


class ParameterProposalTests(unittest.TestCase):
    def test_parameter_proposal_config_requires_explicit_run_parameter_file(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            run_dir = root_path / "run"
            run_dir.mkdir()
            parameter_file = run_dir / "param.yaml"
            parameter_file.write_text(PARAMETER_TEXT, encoding="utf-8")
            config_file = root_path / "monitor.toml"
            config_file.write_text(
                "[project]\n"
                'run_dir = "run"\n'
                'output_dir = "output"\n'
                "[committee_replay]\n"
                "enabled = true\n"
                'command = ["dp"]\n'
                'model_ids = ["000", "001"]\n'
                'model_pattern = "graph.{model_id}.pt2"\n'
                "start_iteration = 2\n"
                "ready_task = 2\n"
                "[parameter_proposals]\n"
                "enabled = true\n"
                'parameter_file = "run/param.yaml"\n'
                'strategy = "repeat_last"\n'
                "start_iteration = 2\n"
                "required_task = 2\n"
                "max_nsteps = 2000000\n",
                encoding="utf-8",
            )

            config = load_config(config_file)

            self.assertTrue(config.parameter_proposals.enabled)
            self.assertEqual(
                config.parameter_proposals.parameter_file,
                parameter_file.resolve(),
            )

            config_file.write_text(
                config_file.read_text(encoding="utf-8").replace(
                    'model_pattern = "graph.{model_id}.pt2"\n',
                    'model_pattern = "graph.{model_id}.pt2"\n'
                    "source_offsets = [1, 2]\n",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "只使用上一轮轨迹"):
                load_config(config_file)

    def test_cli_exposes_separate_review_and_apply_commands(self):
        parser = build_parser()
        approve = parser.parse_args(["approve", "config.toml", "proposal-1"])
        apply = parser.parse_args(["apply", "config.toml", "proposal-1"])
        self.assertEqual(approve.command, "approve")
        self.assertEqual(apply.command, "apply")

    def test_advisor_waits_for_gate_and_previous_replay_then_repeats_last_job(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            parameter_file = root_path / "run" / "param.yaml"
            parameter_file.parent.mkdir()
            parameter_file.write_text(PARAMETER_TEXT, encoding="utf-8")
            output_dir = root_path / "output"
            state = StateStore(output_dir / "monitor.sqlite3")
            proposal_config = ParameterProposalConfig(
                enabled=True,
                parameter_file=parameter_file,
                start_iteration=2,
                required_task=2,
                max_nsteps=2_000_000,
            )
            replay_config = CommitteeReplayConfig(
                enabled=True,
                source_offsets=(1,),
                start_iteration=2,
                ready_task=2,
            )
            advisor = ParameterProposalAdvisor(
                proposal_config,
                replay_config,
                output_dir,
                state,
            )
            snapshot = IterationSnapshot(
                2,
                root_path / "run" / "iter.000002",
                root_path / "run" / "iter.000002" / "00.train",
                2,
            )
            try:
                self.assertIsNone(advisor.consider([snapshot]))
                write_replay_summary(output_dir, state, 2, 1)

                proposal = advisor.consider([snapshot])
                repeated = advisor.consider([snapshot])

                self.assertEqual(proposal["status"], "pending")
                self.assertEqual(proposal["proposal_id"], repeated["proposal_id"])
                self.assertEqual(proposal["target_iteration"], 2)
                self.assertEqual(proposal["proposed_job"]["nsteps"], 2_000_000)
                self.assertEqual(proposal["proposed_job"]["_idx"], 2)
                self.assertEqual(len(proposal["evidence"]), 1)
                self.assertEqual(len(state.list_parameter_proposals()), 1)
                self.assertEqual(parameter_file.read_text(), PARAMETER_TEXT)
                wrong_stage = IterationSnapshot(
                    2, snapshot.iteration_dir, snapshot.train_dir, 3
                )
                self.assertIsNone(advisor.consider([wrong_stage]))
            finally:
                state.close()

    def test_advisor_accepts_current_generation_replay_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            run_dir = root_path / "run"
            run_dir.mkdir()
            parameter_file = run_dir / "param.yaml"
            parameter_file.write_text(PARAMETER_TEXT, encoding="utf-8")
            output_dir = root_path / "output"
            state = StateStore(output_dir / "monitor.sqlite3")
            source = IterationSnapshot(
                1,
                run_dir / "iter.000001",
                None,
                2,
                source_identity="source-new",
            )
            model = IterationSnapshot(
                2,
                run_dir / "iter.000002",
                run_dir / "iter.000002" / "00.train",
                2,
                source_identity="model-new",
            )
            try:
                state.upsert_stage(1, 8)
                state.reconcile_iterations([source])
                state.upsert_stage(2, 8)
                generations, _ = state.reconcile_iterations([source, model])
                self.assertEqual(generations, {1: 1, 2: 1})
                write_replay_summary(
                    output_dir,
                    state,
                    2,
                    1,
                    model_generation=1,
                    source_generation=1,
                )
                advisor = ParameterProposalAdvisor(
                    ParameterProposalConfig(
                        enabled=True,
                        parameter_file=parameter_file,
                        start_iteration=2,
                        required_task=2,
                        max_nsteps=2_000_000,
                    ),
                    CommitteeReplayConfig(
                        enabled=True,
                        source_offsets=(1,),
                        start_iteration=2,
                        ready_task=2,
                    ),
                    output_dir,
                    state,
                )

                current_model = IterationSnapshot(
                    model.iteration,
                    model.iteration_dir,
                    model.train_dir,
                    model.task,
                    source_identity=model.source_identity,
                    generation=1,
                )
                proposal = advisor.consider([source, current_model])

                self.assertIsNotNone(proposal)
                self.assertEqual(proposal["evidence"][0]["model_generation"], 1)
                self.assertEqual(proposal["evidence"][0]["source_generation"], 1)
            finally:
                state.close()

    def test_apply_requires_approval_and_natural_gate_then_only_appends(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            run_dir = root_path / "run"
            run_dir.mkdir()
            parameter_file = run_dir / "param.yaml"
            parameter_file.write_text(PARAMETER_TEXT, encoding="utf-8")
            original = parameter_file.read_text(encoding="utf-8")
            (run_dir / "record.dpgen").write_text("1 8\n2 2\n", encoding="utf-8")
            (run_dir / "iter.000002" / "00.train").mkdir(parents=True)
            state = StateStore(root_path / "output" / "monitor.sqlite3")
            config = ParameterProposalConfig(
                enabled=True,
                parameter_file=parameter_file,
                start_iteration=2,
                required_task=2,
                max_nsteps=2_000_000,
            )
            proposed_job = {
                "sys_idx": [0],
                "temps": [2500, 3000],
                "trj_freq": 1000,
                "nsteps": 2_000_000,
                "ensemble": "nvt",
                "_idx": 2,
            }
            proposal_id = "model-devi-000002-test"
            state.create_parameter_proposal(
                proposal_id=proposal_id,
                target_iteration=2,
                status="pending",
                parameter_file=str(parameter_file),
                parameter_sha256=hashlib.sha256(original.encode()).hexdigest(),
                strategy="repeat_last",
                proposed_job=proposed_job,
                evidence=[],
            )
            controller = ParameterFileController(
                config,
                DpgenConfig(record_file="record.dpgen"),
                run_dir,
                state,
            )
            try:
                with self.assertRaisesRegex(ValueError, "approved"):
                    controller.apply(proposal_id)
                state.transition_parameter_proposal(
                    proposal_id,
                    expected_statuses=("pending",),
                    status="approved",
                )

                path, backup, changed = controller.apply(proposal_id)

                self.assertTrue(changed)
                self.assertEqual(path, parameter_file)
                self.assertIsNotNone(backup)
                self.assertEqual(backup.read_text(encoding="utf-8"), original)
                rendered = parameter_file.read_text(encoding="utf-8")
                self.assertIn("# preserve this header", rendered)
                self.assertIn("# Stage gate: append only after review.", rendered)
                jobs = yaml.safe_load(rendered)["model_devi_jobs"]
                self.assertEqual(len(jobs), 3)
                self.assertEqual(jobs[-1], proposed_job)
                self.assertEqual(
                    state.get_parameter_proposal(proposal_id)["status"],
                    "applied",
                )

                _, same_backup, changed_again = controller.apply(proposal_id)
                self.assertFalse(changed_again)
                self.assertEqual(same_backup, backup)
                self.assertEqual(
                    len(yaml.safe_load(parameter_file.read_text())["model_devi_jobs"]),
                    3,
                )
            finally:
                state.close()

    def test_apply_rejects_parameter_file_changed_after_approval(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            run_dir = root_path / "run"
            run_dir.mkdir()
            parameter_file = run_dir / "param.yaml"
            parameter_file.write_text(PARAMETER_TEXT, encoding="utf-8")
            original_hash = hashlib.sha256(parameter_file.read_bytes()).hexdigest()
            (run_dir / "record.dpgen").write_text("2 2\n", encoding="utf-8")
            (run_dir / "iter.000002" / "00.train").mkdir(parents=True)
            state = StateStore(root_path / "output" / "monitor.sqlite3")
            job = {
                "sys_idx": [0],
                "temps": [2500, 3000],
                "trj_freq": 1000,
                "nsteps": 2_000_000,
                "ensemble": "nvt",
                "_idx": 2,
            }
            state.create_parameter_proposal(
                proposal_id="changed",
                target_iteration=2,
                status="approved",
                parameter_file=str(parameter_file),
                parameter_sha256=original_hash,
                strategy="repeat_last",
                proposed_job=job,
                evidence=[],
            )
            parameter_file.write_text(PARAMETER_TEXT + "# external edit\n")
            controller = ParameterFileController(
                ParameterProposalConfig(
                    enabled=True,
                    parameter_file=parameter_file,
                    required_task=2,
                    max_nsteps=2_000_000,
                ),
                DpgenConfig(),
                run_dir,
                state,
            )
            try:
                with self.assertRaisesRegex(ValueError, "已变化"):
                    controller.apply("changed")
            finally:
                state.close()

    def test_apply_rejects_when_dpgen_is_not_at_natural_gate(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            run_dir = root_path / "run"
            run_dir.mkdir()
            parameter_file = run_dir / "param.yaml"
            parameter_file.write_text(PARAMETER_TEXT, encoding="utf-8")
            (run_dir / "record.dpgen").write_text("2 3\n", encoding="utf-8")
            (run_dir / "iter.000002" / "00.train").mkdir(parents=True)
            state = StateStore(root_path / "output" / "monitor.sqlite3")
            state.create_parameter_proposal(
                proposal_id="wrong-gate",
                target_iteration=2,
                status="approved",
                parameter_file=str(parameter_file),
                parameter_sha256=hashlib.sha256(
                    parameter_file.read_bytes()
                ).hexdigest(),
                strategy="repeat_last",
                proposed_job={
                    "sys_idx": [0],
                    "temps": [2500, 3000],
                    "trj_freq": 1000,
                    "nsteps": 2_000_000,
                    "ensemble": "nvt",
                    "_idx": 2,
                },
                evidence=[],
            )
            controller = ParameterFileController(
                ParameterProposalConfig(
                    enabled=True,
                    parameter_file=parameter_file,
                    required_task=2,
                    max_nsteps=2_000_000,
                ),
                DpgenConfig(),
                run_dir,
                state,
            )
            try:
                with self.assertRaisesRegex(ValueError, "自然停止点"):
                    controller.apply("wrong-gate")
                self.assertEqual(
                    parameter_file.read_text(encoding="utf-8"),
                    PARAMETER_TEXT,
                )
            finally:
                state.close()


if __name__ == "__main__":
    unittest.main()

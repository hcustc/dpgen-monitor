from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .config import EvaluationConfig
from .dpgen import IterationSnapshot
from .evaluation_data import resolve_evaluation_data
from .evaluation_data import resolve_current_iteration_fp_data
from .state import StateStore


@dataclass(frozen=True)
class EvaluationResult:
    model_id: str
    status: str
    force_file: Path | None = None
    lcurve_file: Path | None = None
    error: str | None = None
    test_data: Path | None = None
    baseline_force_file: Path | None = None
    phase: str = "absorption"


def is_valid_force_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        values = np.atleast_2d(np.loadtxt(path))
        return values.shape[0] > 0 and values.shape[1] >= 6
    except Exception:
        return False


class DeepMDEvaluator:
    """DeepMD implementation of the model evaluation adapter."""

    def __init__(
        self,
        config: EvaluationConfig,
        output_dir: Path,
        state: StateStore,
    ):
        self.config = config
        self.output_dir = output_dir / "evaluations"
        self.state = state

    def evaluate_iteration(
        self,
        snapshot: IterationSnapshot,
        phase: str = "absorption",
    ) -> list[EvaluationResult]:
        if phase not in {"absorption", "blind_spot"}:
            raise ValueError(f"未知评估阶段: {phase}")
        if snapshot.train_dir is None:
            return []
        train_dir = snapshot.train_dir
        if phase == "blind_spot":
            test_data = resolve_current_iteration_fp_data(
                train_dir, snapshot.iteration
            )
        else:
            test_data = resolve_evaluation_data(
                train_dir,
                snapshot.iteration,
                self.config.test_data,
                self.config.initial_test_data,
            )
        print(
            f"[模型监控][{phase}] iter.{snapshot.iteration:06d} "
            f"评估数据: {test_data}"
        )
        results = []
        for model_id in self.config.model_ids:
            result = self._evaluate_model(
                snapshot.iteration, phase, train_dir, test_data, model_id
            )
            if (
                result.status == "complete"
                and phase == "absorption"
                and snapshot.iteration > 0
                and self.config.compare_previous_model
            ):
                baseline_force_file = self._resolve_absorption_baseline(
                    snapshot.iteration,
                    train_dir,
                    test_data,
                    model_id,
                )
                result = replace(
                    result,
                    baseline_force_file=baseline_force_file,
                )
            results.append(result)
        return results

    def _resolve_absorption_baseline(
        self,
        iteration: int,
        train_dir: Path,
        test_data: Path,
        model_id: str,
    ) -> Path | None:
        blind_spot_force = (
            self.output_dir
            / f"iter.{iteration - 1:06d}"
            / "blind_spot"
            / model_id
            / f"{model_id}.f.out"
        )
        if is_valid_force_file(blind_spot_force):
            return blind_spot_force
        return self._evaluate_previous_model(
            iteration, train_dir, test_data, model_id
        )

    def _evaluate_previous_model(
        self,
        iteration: int,
        train_dir: Path,
        test_data: Path,
        model_id: str,
    ) -> Path | None:
        """Evaluate the previous model on the same batch, before it was trained in."""
        previous_train_dir = (
            train_dir.parent.parent
            / f"iter.{iteration - 1:06d}"
            / "00.train"
        )
        model_path = previous_train_dir / model_id / self.config.model_file
        iteration_output = (
            self.output_dir / f"iter.{iteration:06d}" / "absorption"
        )
        model_output = iteration_output / "baseline" / model_id
        force_output = model_output / f"{model_id}.f.out"

        if is_valid_force_file(force_output):
            return force_output
        if not model_path.is_file():
            print(f"[模型监控] 缺少上一轮模型，跳过吸收前基线: {model_path}")
            return None
        if not test_data.exists():
            return None

        iteration_output.mkdir(parents=True, exist_ok=True)
        model_output.mkdir(parents=True, exist_ok=True)
        detail_prefix = iteration_output / f"baseline_{model_id}"
        command = [
            *self.config.command,
            "test",
            "-m", str(model_path),
            "-s", str(test_data),
            "-n", str(self.config.num_test),
            "-d", str(detail_prefix),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True)
        except Exception as exc:
            print(f"[模型监控] 上一轮模型基线测试无法启动: {exc}")
            return None

        log_text = (
            f"command: {' '.join(command)}\n"
            f"returncode: {completed.returncode}\n\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}\n"
        )
        (model_output / "dp_test.log").write_text(
            log_text, encoding="utf-8", errors="replace"
        )
        self._archive_outputs(detail_prefix, model_output, model_id)

        if completed.returncode != 0 or not is_valid_force_file(force_output):
            error_text = (completed.stderr or completed.stdout or "无错误输出").strip()
            print(
                f"[模型监控] iter.{iteration - 1:06d}/{model_id} 吸收前基线失败: "
                f"{error_text[-1000:]}"
            )
            return None
        return force_output

    def _evaluate_model(
        self,
        iteration: int,
        phase: str,
        train_dir: Path,
        test_data: Path,
        model_id: str,
    ) -> EvaluationResult:
        model_dir = train_dir / model_id
        model_path = model_dir / self.config.model_file
        lcurve_path = model_dir / "lcurve.out"
        iteration_output = self.output_dir / f"iter.{iteration:06d}" / phase
        model_output = iteration_output / model_id
        force_output = model_output / f"{model_id}.f.out"

        if is_valid_force_file(force_output):
            self.state.set_evaluation(
                iteration, phase, model_id, "complete", str(force_output), None
            )
            return EvaluationResult(
                model_id,
                "complete",
                force_output,
                lcurve_path
                if phase == "absorption" and lcurve_path.is_file()
                else None,
                test_data=test_data,
                phase=phase,
            )

        if not model_path.is_file():
            return EvaluationResult(
                model_id, "waiting", error=f"等待模型 {model_path}",
                test_data=test_data, phase=phase,
            )
        if not test_data.exists():
            error = f"测试数据不存在: {test_data}"
            self.state.set_evaluation(
                iteration, phase, model_id, "failed", last_error=error
            )
            return EvaluationResult(
                model_id, "failed", error=error,
                test_data=test_data, phase=phase,
            )

        iteration_output.mkdir(parents=True, exist_ok=True)
        model_output.mkdir(parents=True, exist_ok=True)
        detail_prefix = iteration_output / model_id
        command = [
            *self.config.command,
            "test",
            "-m", str(model_path),
            "-s", str(test_data),
            "-n", str(self.config.num_test),
            "-d", str(detail_prefix),
        ]
        self.state.set_evaluation(iteration, phase, model_id, "running")
        try:
            completed = subprocess.run(command, capture_output=True, text=True)
        except Exception as exc:
            error = f"无法启动 dp test: {exc}"
            self.state.set_evaluation(
                iteration, phase, model_id, "failed", last_error=error
            )
            return EvaluationResult(
                model_id, "failed", error=error,
                test_data=test_data, phase=phase,
            )

        log_text = (
            f"command: {' '.join(command)}\n"
            f"returncode: {completed.returncode}\n\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}\n"
        )
        (model_output / "dp_test.log").write_text(
            log_text, encoding="utf-8", errors="replace"
        )
        self._archive_outputs(detail_prefix, model_output, model_id)

        if completed.returncode != 0:
            error_text = (completed.stderr or completed.stdout or "无错误输出").strip()
            error = f"dp test 退出码 {completed.returncode}: {error_text[-2000:]}"
            self.state.set_evaluation(
                iteration, phase, model_id, "failed", last_error=error
            )
            return EvaluationResult(
                model_id, "failed", error=error,
                test_data=test_data, phase=phase,
            )

        if not is_valid_force_file(force_output):
            error = "dp test 已结束，但没有生成有效的六列力场输出"
            self.state.set_evaluation(
                iteration, phase, model_id, "failed", last_error=error
            )
            return EvaluationResult(
                model_id, "failed", error=error,
                test_data=test_data, phase=phase,
            )

        self.state.set_evaluation(
            iteration, phase, model_id, "complete", str(force_output), None
        )
        return EvaluationResult(
            model_id,
            "complete",
            force_output,
            lcurve_path if phase == "absorption" and lcurve_path.is_file() else None,
            test_data=test_data,
            phase=phase,
        )

    @staticmethod
    def _archive_outputs(prefix: Path, model_output: Path, model_id: str) -> None:
        suffixes = (
            ".f.out", ".e.out", ".v.out", ".e_peratom.out", ".v_peratom.out",
            ".f.npy", ".e.npy", ".v.npy",
        )
        for suffix in suffixes:
            source = Path(f"{prefix}{suffix}")
            if source.is_file():
                destination = model_output / f"{model_id}{suffix}"
                source.replace(destination)

        legacy_names = {
            "force.out": f"{model_id}.f.out",
            "energy.out": f"{model_id}.e.out",
            "virial.out": f"{model_id}.v.out",
            "energy_peratom.out": f"{model_id}.e_peratom.out",
            "virial_peratom.out": f"{model_id}.v_peratom.out",
        }
        for old_name, new_name in legacy_names.items():
            source = model_output / old_name
            if source.is_file():
                destination = model_output / new_name
                if destination.exists():
                    destination.unlink()
                shutil.move(str(source), str(destination))

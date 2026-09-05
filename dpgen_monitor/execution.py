from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shlex
from typing import Any
from uuid import uuid4

from .config import CommitteeReplayConfig


@dataclass(frozen=True)
class CommitteeReplayRequest:
    """A bounded semantic request; it deliberately contains no shell command."""

    model_iteration: int
    source_iteration: int
    plan_fingerprint: str
    work_dir: Path
    dataset: Path
    models: tuple[Path, ...]
    deviation_file: Path
    frame_count: int
    task_count: int


@dataclass(frozen=True)
class ExecutionResult:
    command: tuple[str, ...]
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    status: str = "complete"


class ReplaySubmissionController:
    """Validate replay intent and submit only a fixed local-GPU task shape."""

    def __init__(
        self,
        config: CommitteeReplayConfig,
        run_dir: Path,
        output_dir: Path,
    ):
        self.config = config
        self.run_dir = run_dir.resolve()
        self.output_dir = output_dir.resolve()

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            return False
        return True

    def _expected_models(self, iteration: int) -> tuple[Path, ...]:
        train_dir = (
            self.run_dir / f"iter.{iteration:06d}" / "00.train"
        ).resolve()
        models = tuple(
            (
                train_dir
                / self.config.model_pattern.format(model_id=model_id)
            ).resolve()
            for model_id in self.config.model_ids
        )
        if any(
            model == train_dir or not self._inside(model, train_dir)
            for model in models
        ):
            raise ValueError("委员会回放模型必须位于目标迭代的 00.train 下")
        return models

    @staticmethod
    def _request_id(request: CommitteeReplayRequest) -> str:
        return (
            f"committee-replay-{request.model_iteration:06d}-"
            f"{request.source_iteration:06d}-{request.plan_fingerprint[:20]}"
        )

    def validate(self, request: CommitteeReplayRequest) -> None:
        if self.config.executor_profile != "local_gpu":
            raise ValueError("未授权的委员会回放执行 profile")
        if not re.fullmatch(r"[0-9a-f]{64}", request.plan_fingerprint):
            raise ValueError("委员会回放 plan_fingerprint 无效")
        offset = request.model_iteration - request.source_iteration
        if offset not in self.config.source_offsets:
            raise ValueError("委员会回放请求了未授权的源迭代")
        if request.model_iteration < self.config.start_iteration:
            raise ValueError("委员会回放请求早于 start_iteration")
        if request.frame_count <= 0:
            raise ValueError("委员会回放必须至少包含一帧")
        if request.frame_count > self.config.max_total_frames:
            raise ValueError(
                f"委员会回放帧数 {request.frame_count} 超过预算 "
                f"{self.config.max_total_frames}"
            )
        if request.task_count <= 0:
            raise ValueError("委员会回放 task_count 无效")
        if request.frame_count > request.task_count * self.config.max_frames_per_task:
            raise ValueError("委员会回放超过每个 task 的抽样上限")

        work_dir = request.work_dir.resolve()
        if work_dir != request.deviation_file.resolve().parent:
            raise ValueError("委员会回放输出必须位于任务工作目录")
        if not self._inside(work_dir, self.output_dir):
            raise ValueError("委员会回放工作目录越界")
        if not self._inside(request.dataset, work_dir):
            raise ValueError("委员会回放数据集必须位于任务工作目录")
        if request.dataset.resolve() != (work_dir / "holdout.deepmd").resolve():
            raise ValueError("委员会回放数据集路径未授权")
        if request.deviation_file.name != "model_devi.out":
            raise ValueError("委员会回放输出文件名未授权")

        models = tuple(path.resolve() for path in request.models)
        if models != self._expected_models(request.model_iteration):
            raise ValueError("委员会回放模型清单与授权委员会不一致")
        missing = [path for path in models if not path.is_file()]
        if not request.dataset.is_dir():
            missing.append(request.dataset)
        if missing:
            raise FileNotFoundError(f"委员会回放输入不存在: {missing[0]}")
        holdout_manifest = work_dir / "holdout_manifest.json"
        try:
            holdout = json.loads(holdout_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("委员会回放 holdout manifest 无效") from exc
        if holdout.get("fingerprint") != request.plan_fingerprint:
            raise ValueError("委员会回放指纹与 holdout manifest 不一致")
        if holdout.get("frame_count") != request.frame_count:
            raise ValueError("委员会回放帧数与 holdout manifest 不一致")

    def _arguments(self, request: CommitteeReplayRequest) -> tuple[str, ...]:
        arguments = (
            "model-devi",
            "-m",
            *(str(path.resolve()) for path in request.models),
            "-s",
            str(request.dataset.resolve()),
            "-o",
            str(request.deviation_file.resolve()),
            "-f",
            "1",
            "--atomic",
        )
        if self.config.relative is not None:
            arguments = (*arguments, "--relative", str(self.config.relative))
        return arguments

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def submit(self, request: CommitteeReplayRequest) -> ExecutionResult:
        self.validate(request)
        request.work_dir.mkdir(parents=True, exist_ok=True)
        command = (*self.config.command, *self._arguments(request))
        request_id = self._request_id(request)
        audit_file = request.work_dir / "execution_request.json"
        try:
            previous_request = json.loads(audit_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_request = {}
        if previous_request.get("request_id") != request_id:
            request.deviation_file.unlink(missing_ok=True)
            for name in ("dispatcher.stdout.log", "dispatcher.stderr.log"):
                (request.work_dir / name).unlink(missing_ok=True)

        audit_payload = {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(request).items()
                if key != "models"
            },
            "models": [str(path) for path in request.models],
            "request_id": request_id,
            "executor_profile": self.config.executor_profile,
            "resources": {
                "number_node": 1,
                "cpu_per_node": self.config.cpu_per_node,
                "gpu_per_node": 1,
                "gpu_device": self.config.gpu_device,
            },
            "dispatch_status": "submitting",
        }
        self._write_json(audit_file, audit_payload)

        try:
            from dpdispatcher import Machine, Resources, Submission, Task
        except ImportError as exc:
            raise RuntimeError(
                "committee replay 需要 dpdispatcher>=1.0.3；"
                "请重新安装 dpgen-monitor"
            ) from exc

        machine = Machine.load_from_dict(
            {
                "batch_type": "Shell",
                "context_type": "LazyLocalContext",
                "local_root": str(request.work_dir),
            }
        )
        resources = Resources(
            number_node=1,
            cpu_per_node=self.config.cpu_per_node,
            gpu_per_node=1,
            queue_name="",
            group_size=1,
            envs={"CUDA_VISIBLE_DEVICES": str(self.config.gpu_device)},
        )
        task = Task(
            # Include the validated request ID in the fixed command shape so a
            # changed replay fingerprint receives a distinct recovery hash.
            command=shlex.join(
                ("env", f"DPGEN_REPLAY_REQUEST_ID={request_id}", *command)
            ),
            task_work_path=".",
            forward_files=[],
            backward_files=[],
            outlog="dispatcher.stdout.log",
            errlog="dispatcher.stderr.log",
        )
        submission = Submission(
            work_base=".",
            machine=machine,
            resources=resources,
            task_list=[task],
        )
        try:
            submission.run_submission(
                exit_on_submit=True,
                clean=False,
                check_interval=self.config.dispatcher_check_interval,
            )
        except Exception as exc:
            stderr_file = request.work_dir / "dispatcher.stderr.log"
            detail = ""
            try:
                detail = stderr_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
            self._write_json(
                audit_file,
                {**audit_payload, "dispatch_status": "failed"},
            )
            suffix = f": {detail[-2000:].strip()}" if detail.strip() else ""
            raise RuntimeError(f"DPDispatcher 委员会回放失败: {exc}{suffix}") from exc

        stdout_file = request.work_dir / "dispatcher.stdout.log"
        stderr_file = request.work_dir / "dispatcher.stderr.log"
        try:
            stdout = stdout_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            stdout = ""
        try:
            stderr = stderr_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            stderr = ""
        if not submission.check_all_finished():
            self._write_json(
                audit_file,
                {**audit_payload, "dispatch_status": "running"},
            )
            return ExecutionResult(
                command=command,
                stdout=stdout,
                stderr=stderr,
                status="running",
            )
        if (
            not request.deviation_file.is_file()
            or request.deviation_file.stat().st_size == 0
        ):
            self._write_json(
                audit_file,
                {**audit_payload, "dispatch_status": "failed"},
            )
            raise RuntimeError("DPDispatcher 任务未生成有效 model_devi.out")
        self._write_json(
            audit_file,
            {**audit_payload, "dispatch_status": "complete"},
        )
        return ExecutionResult(command=command, stdout=stdout, stderr=stderr)

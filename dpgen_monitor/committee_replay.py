from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
from typing import Any, Iterable

import numpy as np

from .config import CommitteeReplayConfig
from .dpgen import IterationSnapshot
from .execution import CommitteeReplayRequest, ReplaySubmissionController
from .state import StateStore


@dataclass(frozen=True)
class ReplayFrame:
    task: str
    step: int
    trajectory: Path
    temperature: float | None
    candidates: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class CommitteeReplayResult:
    model_iteration: int
    source_iteration: int
    status: str
    summary_file: Path | None = None
    error: str | None = None

    @property
    def summary(self) -> dict[str, Any] | None:
        if not self.summary_file or not self.summary_file.is_file():
            return None
        try:
            value = json.loads(self.summary_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None


def _stable_order(seed: int, task: str, step: int) -> str:
    return hashlib.sha256(f"{seed}:{task}:{step}".encode()).hexdigest()


def _task_temperature(task_dir: Path) -> float | None:
    path = task_dir / "job.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("temps")
        return float(value) if value is not None else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _select_task_frames(
    frames: list[ReplayFrame], *, limit: int, bins: int, seed: int
) -> list[ReplayFrame]:
    if len(frames) <= limit:
        return sorted(frames, key=lambda item: item.step)
    frames = sorted(frames, key=lambda item: item.step)
    low, high = frames[0].step, frames[-1].step
    grouped: list[list[ReplayFrame]] = [[] for _ in range(bins)]
    for frame in frames:
        if high == low:
            index = 0
        else:
            index = min(bins - 1, int((frame.step - low) * bins / (high - low + 1)))
        grouped[index].append(frame)

    selected: list[ReplayFrame] = []
    selected_keys: set[tuple[str, int]] = set()
    base, extra = divmod(limit, bins)
    for index, group in enumerate(grouped):
        quota = base + (1 if index < extra else 0)
        ordered = sorted(
            group,
            key=lambda item: _stable_order(seed, item.task, item.step),
        )
        for frame in ordered[:quota]:
            selected.append(frame)
            selected_keys.add((frame.task, frame.step))

    if len(selected) < limit:
        remaining = [
            frame
            for frame in frames
            if (frame.task, frame.step) not in selected_keys
        ]
        remaining.sort(key=lambda item: _stable_order(seed, item.task, item.step))
        selected.extend(remaining[: limit - len(selected)])
    return sorted(selected, key=lambda item: item.step)


def select_replay_frames(
    run_dir: Path,
    source_iteration: int,
    config: CommitteeReplayConfig,
) -> tuple[list[ReplayFrame], Path]:
    source_dir = run_dir / f"iter.{source_iteration:06d}"
    manifest = source_dir / config.candidate_manifest
    if not manifest.is_file():
        raise FileNotFoundError(
            f"等待候选清单 {manifest}；committee replay 需要 DP-GEN "
            "输出原子级 candidate_selection JSONL"
        )

    grouped: dict[tuple[str, int], dict[int, float]] = {}
    selected_frames: set[tuple[str, int]] = set()
    with manifest.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                raw_task = row["task"]
                raw_step = row["step"]
                raw_atom_index = row["atom_index"]
                raw_deviation = row["model_deviation"]
                selected = row["selected"]
                if not isinstance(raw_task, str):
                    raise TypeError("task 必须是字符串")
                if isinstance(raw_step, bool) or not isinstance(raw_step, int):
                    raise TypeError("step 必须是非负整数")
                if isinstance(raw_atom_index, bool) or not isinstance(
                    raw_atom_index, int
                ):
                    raise TypeError(
                        "atom_index 必须是非负整数；当前不支持帧级候选"
                    )
                if isinstance(raw_deviation, bool) or not isinstance(
                    raw_deviation, (int, float)
                ):
                    raise TypeError("model_deviation 必须是有限数值")
                if not isinstance(selected, bool):
                    raise TypeError("selected 必须是布尔值")
                task = Path(raw_task).name
                step = raw_step
                atom_index = raw_atom_index
                deviation = float(raw_deviation)
                if task in {"", ".", ".."}:
                    raise ValueError("task 必须包含有效的 model_devi task 名称")
                if step < 0 or atom_index < 0:
                    raise ValueError("step 和 atom_index 必须是非负整数")
                if not math.isfinite(deviation):
                    raise ValueError("model_deviation 必须是有限数值")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"候选清单格式错误: {manifest}:{line_number}: {exc}"
                ) from exc
            frame_key = (task, step)
            if selected:
                selected_frames.add(frame_key)
            atoms = grouped.setdefault(frame_key, {})
            atoms[atom_index] = max(atoms.get(atom_index, -math.inf), deviation)

    # FP selection is frame-level even though the manifest is atom-level. If
    # any candidate atom selected a frame for labeling, exclude the entire
    # frame so the holdout cannot leak labeled/training configurations.
    if config.exclude_selected:
        for frame_key in selected_frames:
            grouped.pop(frame_key, None)

    by_task: dict[str, list[ReplayFrame]] = {}
    for (task, step), atoms in grouped.items():
        task_dir = source_dir / "01.model_devi" / task
        trajectory = task_dir / "traj" / f"{step}.lammpstrj"
        frame = ReplayFrame(
            task=task,
            step=step,
            trajectory=trajectory,
            temperature=_task_temperature(task_dir),
            candidates=tuple(sorted(atoms.items())),
        )
        by_task.setdefault(task, []).append(frame)

    # Allocate the global budget fairly before selecting within each task.
    # Applying only the per-task limit can exceed max_total_frames on ordinary
    # runs with many temperature/system tasks and leave the controller with no
    # choice but to reject an otherwise valid replay request.
    capacities = {
        task: min(len(frames), config.max_frames_per_task)
        for task, frames in by_task.items()
    }
    task_order = sorted(
        capacities,
        key=lambda task: hashlib.sha256(
            f"{config.seed}:{task}".encode()
        ).hexdigest(),
    )
    quotas = {task: 0 for task in task_order}
    remaining = min(config.max_total_frames, sum(capacities.values()))
    while remaining:
        progressed = False
        for task in task_order:
            if quotas[task] >= capacities[task]:
                continue
            quotas[task] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break

    selected: list[ReplayFrame] = []
    for task in sorted(by_task):
        quota = quotas.get(task, 0)
        if quota == 0:
            continue
        selected.extend(
            _select_task_frames(
                by_task[task],
                limit=quota,
                bins=min(config.time_bins, quota),
                seed=config.seed,
            )
        )
    selected.sort(key=lambda item: (item.task, item.step))
    if not selected:
        reason = "未找到未入选候选" if config.exclude_selected else "候选清单为空"
        raise ValueError(f"{manifest}: {reason}")
    missing = [frame.trajectory for frame in selected if not frame.trajectory.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(f"等待回放轨迹帧（缺少 {len(missing)} 个）: {preview}")
    return selected, manifest


def _parse_box(header: str, rows: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    if any(len(row) < 2 for row in rows):
        raise ValueError("LAMMPS dump 的 BOX BOUNDS 行至少需要两列")
    triclinic = "xy" in header.split()
    if triclinic:
        if any(len(row) < 3 for row in rows):
            raise ValueError("三斜晶胞 BOX BOUNDS 缺少 xy/xz/yz")
        xlo_bound, xhi_bound, xy = rows[0][:3]
        ylo_bound, yhi_bound, xz = rows[1][:3]
        zlo, zhi, yz = rows[2][:3]
        xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
        xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max(0.0, yz)
        cell = np.asarray(
            [[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi - zlo]],
            dtype=float,
        )
        origin = np.asarray([xlo, ylo, zlo], dtype=float)
    else:
        lows = np.asarray([row[0] for row in rows], dtype=float)
        highs = np.asarray([row[1] for row in rows], dtype=float)
        cell = np.diag(highs - lows)
        origin = lows
    if not np.all(np.isfinite(cell)) or abs(float(np.linalg.det(cell))) < 1.0e-12:
        raise ValueError("LAMMPS dump 包含无效晶胞")
    return cell, origin


def read_lammps_dump_frame(path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        def line(expected: str | None = None) -> str:
            value = handle.readline()
            if not value:
                raise ValueError(f"LAMMPS dump 截断: {path}")
            value = value.rstrip("\n")
            if expected is not None and value != expected:
                raise ValueError(f"LAMMPS dump 格式错误: {path}: 期待 {expected!r}")
            return value

        line("ITEM: TIMESTEP")
        step = int(line())
        line("ITEM: NUMBER OF ATOMS")
        natoms = int(line())
        box_header = line()
        if not box_header.startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"LAMMPS dump 缺少 BOX BOUNDS: {path}")
        box_rows = [[float(value) for value in line().split()] for _ in range(3)]
        cell, origin = _parse_box(box_header, box_rows)
        atoms_header = line()
        if not atoms_header.startswith("ITEM: ATOMS "):
            raise ValueError(f"LAMMPS dump 缺少 ATOMS 列: {path}")
        columns = atoms_header.split()[2:]
        try:
            id_index = columns.index("id")
            type_index = columns.index("type")
        except ValueError as exc:
            raise ValueError(f"LAMMPS dump 必须包含 id/type 列: {path}") from exc

        coordinate_names: tuple[str, str, str] | None = None
        scaled = False
        for names, is_scaled in (
            (("x", "y", "z"), False),
            (("xu", "yu", "zu"), False),
            (("xs", "ys", "zs"), True),
            (("xsu", "ysu", "zsu"), True),
        ):
            if all(name in columns for name in names):
                coordinate_names, scaled = names, is_scaled
                break
        if coordinate_names is None:
            raise ValueError(f"LAMMPS dump 缺少受支持的坐标列: {path}")
        coordinate_indices = [columns.index(name) for name in coordinate_names]

        records: list[tuple[int, int, list[float]]] = []
        for _ in range(natoms):
            fields = line().split()
            try:
                atom_id = int(fields[id_index])
                atom_type = int(fields[type_index])
                coordinate = [float(fields[index]) for index in coordinate_indices]
            except (IndexError, ValueError) as exc:
                raise ValueError(f"LAMMPS dump 原子行格式错误: {path}") from exc
            records.append((atom_id, atom_type, coordinate))
        records.sort(key=lambda item: item[0])
        ids = [item[0] for item in records]
        if ids != list(range(1, natoms + 1)):
            raise ValueError(f"LAMMPS dump 原子 id 必须连续且从 1 开始: {path}")
        types = np.asarray([item[1] - 1 for item in records], dtype=np.int32)
        if np.any(types < 0):
            raise ValueError(f"LAMMPS dump 原子 type 必须从 1 开始: {path}")
        coordinates = np.asarray([item[2] for item in records], dtype=float)
        coordinates = coordinates @ cell if scaled else coordinates - origin
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(f"LAMMPS dump 包含非有限坐标: {path}")
        return step, cell, coordinates, types


def _infer_type_map(frames: Iterable[ReplayFrame], configured: tuple[str, ...]) -> list[str]:
    if configured:
        return list(configured)
    for frame in frames:
        path = frame.trajectory.parent.parent / "dpa4c_offline.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("type_map")
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            return value
    return []


def _plan_fingerprint(
    frames: list[ReplayFrame], models: list[Path], config: CommitteeReplayConfig
) -> str:
    payload = {
        "frames": [
            [
                frame.task,
                frame.step,
                list(frame.candidates),
                str(frame.trajectory),
                frame.trajectory.stat().st_size,
                frame.trajectory.stat().st_mtime_ns,
            ]
            for frame in frames
        ],
        "models": [
            [str(path), path.stat().st_size, path.stat().st_mtime_ns] for path in models
        ],
        "relative": config.relative,
        "f_trust_lo": config.f_trust_lo,
        "f_trust_hi": config.f_trust_hi,
        "type_map": config.type_map,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_replay_dataset(
    frames: list[ReplayFrame],
    dataset: Path,
    manifest_path: Path,
    *,
    type_map: tuple[str, ...] = (),
    fingerprint: str,
) -> dict[str, Any]:
    existing: dict[str, Any] | None = None
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    if (
        existing
        and existing.get("fingerprint") == fingerprint
        and (dataset / "set.000" / "coord.npy").is_file()
        and (dataset / "set.000" / "box.npy").is_file()
        and (dataset / "type.raw").is_file()
    ):
        return existing

    cells: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    atom_types: np.ndarray | None = None
    rendered_frames: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        step, cell, coords, types = read_lammps_dump_frame(frame.trajectory)
        if step != frame.step:
            raise ValueError(
                f"轨迹文件步号不一致: {frame.trajectory}: {step} != {frame.step}"
            )
        if atom_types is None:
            atom_types = types
        elif not np.array_equal(atom_types, types):
            raise ValueError("回放 holdout 中的原子数或类型顺序不一致")
        cells.append(cell.reshape(-1))
        coordinates.append(coords.reshape(-1))
        rendered_frames.append(
            {
                "index": index,
                "task": frame.task,
                "step": frame.step,
                "temperature": frame.temperature,
                "trajectory": str(frame.trajectory),
                "candidates": [
                    {"atom_index": atom, "old_deviation": deviation}
                    for atom, deviation in frame.candidates
                ],
            }
        )
    if atom_types is None:
        raise ValueError("回放 holdout 为空")

    names = _infer_type_map(frames, type_map)
    if names and int(np.max(atom_types)) >= len(names):
        raise ValueError("type_map 无法覆盖轨迹中的原子类型")
    temporary = dataset.with_name(f".{dataset.name}.tmp-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    try:
        set_dir = temporary / "set.000"
        set_dir.mkdir(parents=True)
        np.save(set_dir / "coord.npy", np.asarray(coordinates, dtype=np.float64))
        np.save(set_dir / "box.npy", np.asarray(cells, dtype=np.float64))
        (temporary / "type.raw").write_text(
            " ".join(str(int(value)) for value in atom_types) + "\n",
            encoding="utf-8",
        )
        if names:
            (temporary / "type_map.raw").write_text(
                "\n".join(names) + "\n", encoding="utf-8"
            )
        if dataset.exists():
            shutil.rmtree(dataset)
        os.replace(temporary, dataset)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    manifest = {
        "fingerprint": fingerprint,
        "frame_count": len(frames),
        "atom_count": int(atom_types.size),
        "type_map": names,
        "frames": rendered_frames,
    }
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.tmp-{os.getpid()}"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def summarize_replay_output(
    output: Path,
    holdout: dict[str, Any],
    *,
    f_trust_lo: float,
    f_trust_hi: float,
) -> dict[str, Any]:
    frames = holdout["frames"]
    natoms = int(holdout["atom_count"])
    rows: list[list[float]] = []
    with output.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                values = [float(value) for value in line.split()]
            except ValueError as exc:
                raise ValueError(f"model deviation 输出包含非数值: {output}:{line_number}") from exc
            if len(values) not in {7 + natoms, 8 + natoms}:
                raise ValueError(
                    f"model deviation 列数错误: {output}:{line_number}: "
                    f"{len(values)}，期待 {7 + natoms} 或 {8 + natoms}"
                )
            rows.append(values)
    if len(rows) != len(frames):
        raise ValueError(
            f"model deviation 帧数错误: {len(rows)}，期待 {len(frames)}"
        )

    tracked = absorbed = remaining = worsened = 0
    all_accurate = all_candidate = all_failed = 0
    per_task: dict[str, dict[str, Any]] = {}
    for metadata, values in zip(frames, rows):
        atomic = np.asarray(values[-natoms:], dtype=float)
        if not np.all(np.isfinite(atomic)):
            raise ValueError("model deviation 输出包含非有限原子偏差")
        accurate_mask = atomic < f_trust_lo
        candidate_mask = (atomic >= f_trust_lo) & (atomic < f_trust_hi)
        failed_mask = atomic >= f_trust_hi
        all_accurate += int(np.count_nonzero(accurate_mask))
        all_candidate += int(np.count_nonzero(candidate_mask))
        all_failed += int(np.count_nonzero(failed_mask))

        task = str(metadata["task"])
        task_row = per_task.setdefault(
            task,
            {
                "temperature": metadata.get("temperature"),
                "frames": 0,
                "tracked_candidates": 0,
                "absorbed": 0,
                "remaining_candidate": 0,
                "worsened_to_failed": 0,
                "all_atom_candidate": 0,
                "all_atom_failed": 0,
                "all_atom_total": 0,
            },
        )
        task_row["frames"] += 1
        task_row["all_atom_candidate"] += int(np.count_nonzero(candidate_mask))
        task_row["all_atom_failed"] += int(np.count_nonzero(failed_mask))
        task_row["all_atom_total"] += natoms
        for candidate in metadata["candidates"]:
            atom_index = int(candidate["atom_index"])
            if atom_index < 0 or atom_index >= natoms:
                raise ValueError(f"candidate atom_index 越界: {atom_index}/{natoms}")
            tracked += 1
            task_row["tracked_candidates"] += 1
            if accurate_mask[atom_index]:
                absorbed += 1
                task_row["absorbed"] += 1
            elif candidate_mask[atom_index]:
                remaining += 1
                task_row["remaining_candidate"] += 1
            else:
                worsened += 1
                task_row["worsened_to_failed"] += 1

    total = all_accurate + all_candidate + all_failed
    return {
        "frame_count": len(frames),
        "atom_count": natoms,
        "tracked_candidates": tracked,
        "absorbed": absorbed,
        "remaining_candidate": remaining,
        "worsened_to_failed": worsened,
        "absorption_percent": 100.0 * absorbed / tracked if tracked else 0.0,
        "all_atom_accurate": all_accurate,
        "all_atom_candidate": all_candidate,
        "all_atom_failed": all_failed,
        "all_atom_total": total,
        "candidate_percent": 100.0 * all_candidate / total if total else 0.0,
        "failed_percent": 100.0 * all_failed / total if total else 0.0,
        "per_task": per_task,
    }


class CommitteeReplayEvaluator:
    def __init__(
        self,
        config: CommitteeReplayConfig,
        run_dir: Path,
        output_dir: Path,
        state: StateStore,
        stop_event: threading.Event | None = None,
    ):
        self.config = config
        self.run_dir = run_dir
        self.output_dir = output_dir / "evaluations"
        self.state = state
        self.stop_event = stop_event
        self.submission_controller = ReplaySubmissionController(
            config, run_dir, self.output_dir
        )

    def set_stop_event(self, stop_event: threading.Event | None) -> None:
        self.stop_event = stop_event

    def _stop_requested(self) -> bool:
        return bool(self.stop_event and self.stop_event.is_set())

    def _output_dir(
        self, model: IterationSnapshot, source: IterationSnapshot
    ) -> Path:
        result = self.output_dir / f"iter.{model.iteration:06d}"
        if model.generation > 0:
            result = result / f"generation.{model.generation:06d}"
        result = result / "committee_replay" / f"source.iter.{source.iteration:06d}"
        if source.generation > 0:
            result = result / f"generation.{source.generation:06d}"
        return result

    def _models(self, snapshot: IterationSnapshot) -> list[Path]:
        if snapshot.train_dir is None:
            return []
        return [
            snapshot.train_dir
            / self.config.model_pattern.format(model_id=model_id)
            for model_id in self.config.model_ids
        ]

    def evaluate(
        self, model: IterationSnapshot, source: IterationSnapshot
    ) -> CommitteeReplayResult:
        identity = (model.iteration, source.iteration)
        output_dir = self._output_dir(model, source)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_file = output_dir / "summary.json"
        try:
            frames, candidate_manifest = select_replay_frames(
                self.run_dir, source.iteration, self.config
            )
            models = self._models(model)
            missing_models = [path for path in models if not path.is_file()]
            if missing_models:
                error = f"等待委员会模型 {missing_models[0]}"
                self.state.set_committee_replay(
                    *identity, "waiting", last_error=error
                )
                return CommitteeReplayResult(
                    *identity,
                    "waiting",
                    error=error,
                )
            fingerprint = _plan_fingerprint(frames, models, self.config)
            try:
                existing = json.loads(summary_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if existing and existing.get("fingerprint") == fingerprint:
                self.state.set_committee_replay(
                    *identity, "complete", str(summary_file), None
                )
                return CommitteeReplayResult(*identity, "complete", summary_file)

            dataset = output_dir / "holdout.deepmd"
            holdout_file = output_dir / "holdout_manifest.json"
            holdout = build_replay_dataset(
                frames,
                dataset,
                holdout_file,
                type_map=self.config.type_map,
                fingerprint=fingerprint,
            )
            if self._stop_requested():
                self.state.set_committee_replay(*identity, "cancelled")
                return CommitteeReplayResult(*identity, "cancelled")

            deviation_file = output_dir / "model_devi.out"
            self.state.set_committee_replay(*identity, "running")
            execution = self.submission_controller.submit(
                CommitteeReplayRequest(
                    model_iteration=model.iteration,
                    source_iteration=source.iteration,
                    plan_fingerprint=fingerprint,
                    work_dir=output_dir,
                    dataset=dataset,
                    models=tuple(models),
                    deviation_file=deviation_file,
                    frame_count=len(frames),
                    task_count=len({frame.task for frame in frames}),
                )
            )
            if execution.status == "running":
                self.state.set_committee_replay(*identity, "running")
                return CommitteeReplayResult(*identity, "running")
            (output_dir / "dp_model_devi.log").write_text(
                f"command: {' '.join(execution.command)}\n"
                f"candidate_manifest: {candidate_manifest}\n"
                f"executor_profile: {self.config.executor_profile}\n"
                f"returncode: {execution.returncode}\n\n"
                f"stdout:\n{execution.stdout}\n\n"
                f"stderr:\n{execution.stderr}\n",
                encoding="utf-8",
                errors="replace",
            )
            cancelled = self._stop_requested() or "KeyboardInterrupt" in (
                f"{execution.stdout or ''}\n{execution.stderr or ''}"
            )
            if cancelled:
                self.state.set_committee_replay(*identity, "cancelled")
                return CommitteeReplayResult(*identity, "cancelled")
            if not deviation_file.is_file() or deviation_file.stat().st_size == 0:
                raise RuntimeError("dp model-devi 未生成有效输出")

            summary = summarize_replay_output(
                deviation_file,
                holdout,
                f_trust_lo=self.config.f_trust_lo,
                f_trust_hi=self.config.f_trust_hi,
            )
            summary.update(
                {
                    "fingerprint": fingerprint,
                    "model_iteration": model.iteration,
                    "source_iteration": source.iteration,
                    "model_generation": model.generation,
                    "source_generation": source.generation,
                    "models": [str(path) for path in models],
                    "candidate_manifest": str(candidate_manifest),
                    "f_trust_lo": self.config.f_trust_lo,
                    "f_trust_hi": self.config.f_trust_hi,
                    "relative": self.config.relative,
                    "executor_profile": self.config.executor_profile,
                }
            )
            temporary = summary_file.with_name(f".{summary_file.name}.tmp-{os.getpid()}")
            temporary.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, summary_file)
            self.state.set_committee_replay(
                *identity, "complete", str(summary_file), None
            )
            return CommitteeReplayResult(*identity, "complete", summary_file)
        except (OSError, ValueError, RuntimeError) as exc:
            error = str(exc)
            status = "waiting" if error.startswith("等待") else "failed"
            self.state.set_committee_replay(*identity, status, last_error=error)
            return CommitteeReplayResult(*identity, status, error=error)

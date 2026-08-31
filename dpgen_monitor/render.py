from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dpgen-monitor-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from .evaluation import EvaluationResult
from .evaluation_plots import (
    best_force_model,
    load_force_parities,
    render_absorption_gain,
    render_dodecane_force_parity,
    render_force_density_comparison,
)


def format_percentage(value: float) -> str:
    """Preserve meaningful precision close to 0% and 100%."""
    value = float(value)
    distance = min(abs(value), abs(100.0 - value))
    decimals = 4 if 0.0 < distance < 0.01 else 2
    while decimals < 8 and 0.0 < value < 100.0:
        rounded = round(value, decimals)
        if rounded not in {0.0, 100.0}:
            break
        decimals += 1
    return f"{value:.{decimals}f}%"


def _annotation_positions(values: list[float], regular: int = 5) -> list[int]:
    values_array = np.asarray(values, dtype=float)
    size = len(values_array)
    if size <= regular:
        return list(range(size))
    selected = set(np.linspace(0, size - 1, regular, dtype=int))
    selected.update(range(max(0, size - 2), size))
    finite = np.flatnonzero(np.isfinite(values_array))
    if finite.size:
        selected.add(int(finite[np.argmin(values_array[finite])]))
        selected.add(int(finite[np.argmax(values_array[finite])]))
    return sorted(selected)


def render_statistics_trend(rows: list[dict], output_path: Path) -> Path:
    if not rows:
        raise ValueError("没有统计数据可绘制")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iterations = [int(row["iteration"]) for row in rows]
    fig, axes = plt.subplots(
        3, 1, figsize=(15, 12), sharex=True, constrained_layout=True
    )
    metrics = (
        ("candidate", "Candidate", "#277DA1"),
        ("failed", "Failed", "#F3722C"),
        ("accurate", "Accurate", "#43AA8B"),
    )
    for axis, (prefix, label, color) in zip(axes, metrics):
        percentages = []
        for row in rows:
            count = int(row[f"{prefix}_count"])
            total = int(row[f"{prefix}_total"])
            percentages.append(
                count / total * 100.0
                if total
                else float(row[f"{prefix}_percent"])
            )
        axis.plot(
            iterations, percentages, marker="o", linewidth=2,
            markersize=5, color=color, markeredgecolor="white",
            markeredgewidth=0.8, zorder=3,
        )
        axis.fill_between(iterations, percentages, 0, color=color, alpha=0.07)
        axis.set_title(f"{label} Trend", fontsize=15, weight="bold", pad=10)
        axis.set_ylabel("Percentage (%)", fontsize=11, weight="bold")
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.28)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for rank, position in enumerate(_annotation_positions(percentages)):
            iteration = iterations[position]
            percentage = percentages[position]
            row = rows[position]
            count = int(row[f"{prefix}_count"])
            total = int(row[f"{prefix}_total"])
            text = format_percentage(percentage)
            if total:
                text += f"\n{count:,} / {total:,}"
            x_offset = 0
            alignment = "center"
            if position == len(rows) - 2:
                x_offset, alignment = -8, "right"
            elif position == len(rows) - 1:
                x_offset, alignment = 8, "left"
            axis.annotate(
                text, (iteration, percentage),
                xytext=(x_offset, 9 + 18 * (rank % 3)),
                textcoords="offset points", ha=alignment, va="bottom",
                fontsize=8.2, weight="semibold",
                bbox={
                    "facecolor": "white", "alpha": 0.9,
                    "edgecolor": color, "linewidth": 0.7,
                    "boxstyle": "round,pad=0.3",
                },
                arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.45},
                zorder=4,
            )
        y_min, y_max = min(percentages), max(percentages)
        if prefix == "accurate":
            spread = max(y_max - y_min, 1.0)
            axis.set_ylim(max(0.0, y_min - spread * 0.18), y_max + spread * 0.38)
        else:
            axis.set_ylim(0.0, max(0.1, y_max * 1.35))
    axes[-1].set_xlabel("Iteration", fontsize=12, weight="bold")
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_evaluation(
    iteration: int,
    results: list[EvaluationResult],
    output_dir: Path,
    *,
    phase: str = "absorption",
) -> tuple[Path, ...]:
    complete = [result for result in results if result.force_file is not None]
    if not complete:
        return ()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"iter.{iteration:06d}"
    current_rows = load_force_parities({
        result.model_id: result.force_file
        for result in complete
        if result.force_file is not None
    })
    force_density_png = render_force_density_comparison(
        current_rows,
        output_dir / f"{name}_force_density.png",
        iteration=iteration,
        context_label=(
            "previous-FP absorption"
            if phase == "absorption"
            else "new-FP blind-spot"
        ),
    )
    best = best_force_model(current_rows)
    best_force_png = render_dodecane_force_parity(
        best,
        output_dir / f"{name}_best_model_{best.model_id}_force_parity.png",
    )

    images = [force_density_png, best_force_png]
    baseline_files = {
        result.model_id: result.baseline_force_file
        for result in complete
        if result.baseline_force_file is not None
    }
    if phase == "absorption" and baseline_files:
        baseline_rows = load_force_parities(baseline_files)
        absorption_png = render_absorption_gain(
            current_rows,
            baseline_rows,
            output_dir / f"{name}_absorption_gain.png",
            iteration=iteration,
        )
        images.append(absorption_png)

    lcurve_results = [
        result for result in complete
        if phase == "absorption" and result.lcurve_file
    ]
    if lcurve_results:
        columns = 2
        rows = math.ceil(len(lcurve_results) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(12, 5 * rows), squeeze=False)
        for axis in axes.flat:
            axis.set_axis_off()
        for axis, result in zip(axes.flat, lcurve_results):
            values = np.atleast_2d(np.loadtxt(result.lcurve_file, comments="#"))
            if values.shape[1] < 4:
                continue
            axis.set_axis_on()
            axis.plot(values[:, 0], values[:, 1], label="RMSE_trn")
            axis.plot(values[:, 0], values[:, 2], label="RMSE_e_trn")
            axis.plot(values[:, 0], values[:, 3], label="RMSE_f_trn")
            axis.set_xscale("symlog")
            axis.set_yscale("log")
            axis.set_title(f"model {result.model_id}")
            axis.set_xlabel("Step")
            axis.set_ylabel("RMSE")
            axis.grid(True, linestyle="--", alpha=0.3)
            axis.legend()
        fig.suptitle(f"{name} learning curves")
        fig.tight_layout()
        lcurve_png = output_dir / f"{name}_lcurve.png"
        fig.savefig(lcurve_png, dpi=220)
        plt.close(fig)
        images.append(lcurve_png)
    return tuple(images)

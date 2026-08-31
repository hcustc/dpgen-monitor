from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dpgen-monitor-matplotlib")

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interpn


@dataclass(frozen=True)
class ForceParityData:
    model_id: str
    reference: np.ndarray
    predicted: np.ndarray
    mae: float
    rmse: float
    vector_rmse: float


def load_force_parity(model_id: str, path: str | Path) -> ForceParityData:
    values = np.atleast_2d(np.loadtxt(path, comments="#"))
    if values.size == 0 or values.shape[1] < 6:
        raise ValueError(f"Expected a non-empty six-column force file: {path}")
    true_vectors = values[:, :3]
    predicted_vectors = values[:, 3:6]
    error = predicted_vectors - true_vectors
    return ForceParityData(
        model_id=str(model_id),
        reference=true_vectors.ravel(),
        predicted=predicted_vectors.ravel(),
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(np.square(error)))),
        vector_rmse=float(np.sqrt(np.mean(np.sum(np.square(error), axis=1)))),
    )


def load_force_parities(
    force_files: dict[str, str | Path],
) -> list[ForceParityData]:
    return [
        load_force_parity(model_id, path)
        for model_id, path in sorted(force_files.items())
    ]


def best_force_model(rows: list[ForceParityData]) -> ForceParityData:
    if not rows:
        raise ValueError("No force results are available")
    return min(rows, key=lambda row: row.rmse)


def _density_at_points(
    reference: np.ndarray,
    predicted: np.ndarray,
    bins: int,
) -> np.ndarray:
    histogram, x_edges, y_edges = np.histogram2d(reference, predicted, bins=bins)
    density = interpn(
        (
            0.5 * (x_edges[1:] + x_edges[:-1]),
            0.5 * (y_edges[1:] + y_edges[:-1]),
        ),
        histogram,
        np.vstack([reference, predicted]).T,
        method="splinef2d",
        bounds_error=False,
    )
    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    density[density < 0] = 0
    return density


def render_force_density_comparison(
    rows: list[ForceParityData],
    output_path: str | Path,
    *,
    iteration: int | None = None,
    context_label: str | None = None,
    bins: int = 800,
    limits: tuple[float, float] = (-101, 101),
) -> Path:
    if not rows:
        raise ValueError("No force results are available")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    densities = [
        _density_at_points(row.reference, row.predicted, bins)
        for row in rows
    ]
    maximum_density = max(float(np.max(density)) for density in densities)
    norm = matplotlib.colors.LogNorm(vmin=1, vmax=max(1.01, maximum_density))

    columns = 2
    row_count = math.ceil(len(rows) / columns)
    fig, axes = plt.subplots(
        row_count,
        columns,
        figsize=(13, 6 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    used_axes = []
    low, high = limits
    for axis, row, density in zip(axes.flat, rows, densities):
        order = density.argsort()
        reference = row.reference[order]
        predicted = row.predicted[order]
        density = density[order]
        image = axis.scatter(
            reference,
            predicted,
            s=3,
            c=density,
            cmap="rainbow",
            edgecolors="none",
            norm=norm,
            rasterized=True,
            zorder=30,
        )
        axis.plot(
            limits,
            limits,
            color="blue",
            linestyle="--",
            linewidth=2,
            zorder=20,
        )
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("DFT forces (eV/Å)")
        axis.set_ylabel("DP forces (eV/Å)")
        outside = int(
            np.count_nonzero(
                (reference < low)
                | (reference > high)
                | (predicted < low)
                | (predicted > high)
            )
        )
        axis.set_title(
            f"model {row.model_id}  MAE={row.mae:.3f}, RMSE={row.rmse:.3f}\n"
            f"outside view: {outside:,} components"
        )
        used_axes.append(axis)
    for axis in axes.flat[len(rows):]:
        axis.set_visible(False)

    title = "Force-component density parity"
    if iteration is not None:
        context = context_label or "previous-batch absorption"
        title = f"iter.{iteration:06d} {context} — {title}"
    fig.suptitle(title, fontsize=16)
    fig.colorbar(image, ax=used_axes, label="Local density", shrink=0.85)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_dodecane_force_parity(
    row: ForceParityData,
    output_path: str | Path,
    *,
    bins: int = 10000,
) -> Path:
    """Render the reference dodecane_fig2.ipynb visual style."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    a = row.reference.copy()
    b = row.predicted.copy()
    z = _density_at_points(a, b, bins)
    order = z.argsort()
    a, b, z = a[order], b[order], z[order]

    with matplotlib.rc_context({"font.size": 60}):
        fig = plt.figure(figsize=(32, 32))
        ax2 = plt.axes()
        image = ax2.scatter(
            a,
            b,
            s=10,
            c=z,
            cmap="rainbow",
            edgecolors="none",
            zorder=30,
            norm=matplotlib.colors.LogNorm(vmin=1),
            rasterized=True,
        )

        divider = make_axes_locatable(ax2)
        color_axis = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(image, ax=ax2, cax=color_axis)
        ax2.set_aspect("equal")

        limits = [-101, 101]
        ax2.set_xlim(limits)
        ax2.set_ylim(limits)
        ax2.set_xlabel("DFT forces (eV/Å)")
        ax2.set_ylabel("DP forces (eV/Å)")
        ax2.plot(
            limits,
            limits,
            color="blue",
            linestyle="--",
            linewidth=10,
            dashes=(2, 2),
            zorder=20,
        )
        ax2.set_xticks(np.arange(-100, 101, 50))
        ax2.set_yticks(np.arange(-100, 101, 50))

        ax3 = fig.add_axes([0.3, 0.6, 0.2, 0.2])
        ax3.scatter(
            a,
            b,
            s=10,
            c=z,
            cmap="rainbow",
            edgecolors="none",
            zorder=30,
            norm=matplotlib.colors.LogNorm(vmin=1),
            rasterized=True,
        )
        ax3.plot([-11, 11], [-11, 11], color="red", alpha=0.75)
        ax3.set_xlim([-11, 11])
        ax3.set_ylim([-11, 11])

        ax2.text(
            limits[0] + (limits[1] - limits[0]) * 0.3,
            limits[0] + (limits[1] - limits[0]) * 0.1,
            f"MAE={row.mae:.2f} eV/Å\nRMSE={row.rmse:.2f} eV/Å",
        )
        [item.set_linewidth(5) for item in ax3.spines.values()]
        [item.set_linewidth(10) for item in ax2.spines.values()]
        ax2.xaxis.set_tick_params(width=10, size=25)
        ax2.yaxis.set_tick_params(width=10, size=25)
        ax3.xaxis.set_tick_params(width=5, size=20)
        ax3.yaxis.set_tick_params(width=5, size=20)
        # ``tight_layout`` cannot handle the manually positioned inset and the
        # divider-created colorbar axes used above.  Let ``savefig`` calculate
        # a tight bounding box instead; this keeps every axes in the output
        # without enabling an incompatible layout engine.
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return output_path


def render_absorption_gain(
    current_rows: list[ForceParityData],
    previous_rows: list[ForceParityData],
    output_path: str | Path,
    *,
    iteration: int,
) -> Path:
    previous_by_model = {row.model_id: row for row in previous_rows}
    paired = [
        (row, previous_by_model[row.model_id])
        for row in current_rows
        if row.model_id in previous_by_model
    ]
    if not paired:
        raise ValueError("No previous/current model pairs are available")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [current.model_id for current, _ in paired]
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    for axis, metric, label in (
        (axes[0], "rmse", "Force RMSE (eV/Å)"),
        (axes[1], "mae", "Force MAE (eV/Å)"),
    ):
        before = np.array([getattr(previous, metric) for _, previous in paired])
        after = np.array([getattr(current, metric) for current, _ in paired])
        axis.bar(x - width / 2, before, width, label=f"iter.{iteration - 1:06d} model")
        axis.bar(x + width / 2, after, width, label=f"iter.{iteration:06d} model")
        axis.set_xticks(x, labels)
        axis.set_xlabel("Model")
        axis.set_ylabel(label)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.legend()
        for position, old, new in zip(x, before, after):
            reduction = 100.0 * (old - new) / old if old else 0.0
            axis.text(
                position,
                max(old, new) * 1.02,
                f"{reduction:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(
        f"iter.{iteration:06d}: absorption gain on iter.{iteration - 1:06d} FP data",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path

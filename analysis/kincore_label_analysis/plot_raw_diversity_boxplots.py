#!/usr/bin/env python3
"""
Diversity box plots for the csvs_for_diversity_analysis subset (3×4 grid, shared styling).

1) **Raw pooled values** — distribution of raw distances (Å) and dihedrals (°) over all
   correct predictions; optional second figure adds KinConfBench (experimental) reference values per panel.
   **Mean ± sample std bars** — same data summarized as bar height = mean, error bar = $s$ (ddof=1),
   for easier comparison to the reference mean spread.
2) **Per-system σ** — distribution of sample standard deviations within each system
   (n≥2 predictions per method and metric), same preprocessing (Asp χ2 π-fold to [−90, 90], Phe χ2 to [0,180) before σ; raw Asp panel y-axis is clipped to −80…50 for display).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from KinConfBench.analysis.kincore_label_analysis.analyze_correct_prediction_diversity import (
    ANGLE_COLUMNS,
    DISTANCE_COLUMNS,
)
from KinConfBench.analysis.plot_style import labels_for_methods, method_colors

# Typography
MAIN_TICK = 17
MAIN_TITLE = 24  # panel titles (+4 vs prior 20)
MAIN_LABEL = 22  # row y-labels Å / Degrees (+2 vs prior 20)
# Y-axis tick numerals: larger and sparse (≤5 ticks per panel)
YTICK_LABEL = MAIN_TICK + 4
YAXIS_MAX_TICKS = 5
# Legend text matches y-tick numeral size
LEGEND_FONT = YTICK_LABEL

plt.rcParams.setdefault("figure.dpi", 300)
plt.rcParams.setdefault("savefig.dpi", 300)

REFERENCE_LABEL = "KinConfBench"
REFERENCE_COLOR = (0.45, 0.45, 0.45, 0.85)

ALL_RAW_COLUMNS: tuple[str, ...] = tuple(DISTANCE_COLUMNS) + tuple(ANGLE_COLUMNS)

ASP_CHI2_COLUMN = "asp_chi2"
PHE_CHI2_COLUMN = "phe_chi2"


def _fold_asp_chi2_pm90(deg: float) -> float:
    """π-fold Asp χ2 to [−90, 90]; normalize to (−180, 180] first so negatives are not mangled by % 180."""
    x = (float(deg) + 180.0) % 360.0 - 180.0
    if x > 90.0:
        return x - 180.0
    if x < -90.0:
        return x + 180.0
    return x


def _fold_phe_chi2_0_180(deg: float) -> float:
    """π-fold Phe χ2 to [0, 180) (θ ≡ θ + 180°)."""
    return float(deg) % 180.0


def _normalize_pooled_value(column: str, value: float) -> float:
    if column == ASP_CHI2_COLUMN:
        return _fold_asp_chi2_pm90(value)
    if column == PHE_CHI2_COLUMN:
        return _fold_phe_chi2_0_180(value)
    return value


_METRIC_TITLES: dict[str, str] = {
    "lys_glu_distance": "Lys–Glu distance",
    "saltbridge_distance": "Salt bridge distance",
    "spine_distance": "Spine distance",
    "dfg_hrd_distance": "DFG–HRD distance",
    "asp_phi": "Asp φ",
    "asp_psi": "Asp ψ",
    "asp_chi1": "Asp χ1",
    "asp_chi2": "Asp χ2",
    "phe_phi": "Phe φ",
    "phe_psi": "Phe ψ",
    "phe_chi1": "Phe χ1",
    "phe_chi2": "Phe χ2",
}


def _parse_float(cell: str | None) -> float | None:
    if cell is None or not str(cell).strip():
        return None
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def collect_raw_diversity_pools(
    csv_dir: Path | str,
    methods: tuple[str, ...],
) -> tuple[dict[tuple[str, str], list[float]], dict[str, list[float]], int]:
    """
    Pool raw values across all diversity-subset CSVs.

    Returns:
        prediction_pools: (method, column_name) -> list of values from prediction rows
        reference_pools: column_name -> one value per CSV when a PDB reference row exists
        n_files: number of CSV files read
    """
    csv_dir = Path(csv_dir).resolve()
    methods = tuple(methods)
    prediction_pools: dict[tuple[str, str], list[float]] = defaultdict(list)
    reference_pools: dict[str, list[float]] = defaultdict(list)

    csv_files = sorted(
        f
        for f in csv_dir.glob("*.csv")
        if f.is_file() and not f.name.startswith("diversity_")
    )
    n_files = 0
    for csv_path in csv_files:
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            ref_taken = False
            for row in reader:
                sid = (row.get("structure_id") or "").strip()

                if len(sid) == 4 and not ref_taken:
                    for col in ALL_RAW_COLUMNS:
                        v = _parse_float(row.get(col))
                        if v is not None:
                            reference_pools[col].append(_normalize_pooled_value(col, v))
                    ref_taken = True

                for m in methods:
                    if sid.startswith(f"{m}_"):
                        for col in ALL_RAW_COLUMNS:
                            v = _parse_float(row.get(col))
                            if v is not None:
                                prediction_pools[(m, col)].append(_normalize_pooled_value(col, v))
        n_files += 1

    return prediction_pools, reference_pools, n_files


def collect_per_system_std_pools(
    csv_dir: Path | str,
    methods: tuple[str, ...],
) -> tuple[dict[tuple[str, str], list[float]], int]:
    """
    For each CSV (system), compute sample std (ddof=1) of each metric within each method's
    correct predictions; append one σ per system when n≥2. Same value normalization as raw pools.
    """
    csv_dir = Path(csv_dir).resolve()
    methods = tuple(methods)
    std_pools: dict[tuple[str, str], list[float]] = defaultdict(list)

    csv_files = sorted(
        f
        for f in csv_dir.glob("*.csv")
        if f.is_file() and not f.name.startswith("diversity_")
    )
    n_files = 0
    for _csv_path in csv_files:
        per_file: dict[tuple[str, str], list[float]] = defaultdict(list)
        with open(_csv_path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = (row.get("structure_id") or "").strip()
                for m in methods:
                    if sid.startswith(f"{m}_"):
                        for col in ALL_RAW_COLUMNS:
                            v = _parse_float(row.get(col))
                            if v is not None:
                                per_file[(m, col)].append(_normalize_pooled_value(col, v))
        for m in methods:
            for col in ALL_RAW_COLUMNS:
                vals = per_file[(m, col)]
                if len(vals) >= 2:
                    std_pools[(m, col)].append(float(np.std(vals, ddof=1)))
        n_files += 1

    return std_pools, n_files


def _draw_boxplot_grid(
    *,
    output_prefix: str,
    methods: tuple[str, ...],
    prediction_pools: dict[tuple[str, str], list[float]],
    reference_pools: dict[str, list[float]],
    include_reference: bool,
    suffix: str,
    row_ylabel_for_row_left_metric: Callable[[str], str],
    fixed_ylim_for_column: dict[str, tuple[float, float]],
) -> None:
    colors = method_colors(methods)
    method_labels = labels_for_methods(methods)

    fig, axes = plt.subplots(3, 4, figsize=(21, 17))
    axes_flat = axes.flatten()

    ncols = 4

    def _set_row_ylabel(ax_axis, subplot_idx: int) -> None:
        if subplot_idx % ncols != 0:
            ax_axis.set_ylabel("")
            return
        row = subplot_idx // ncols
        row_left_metric = ALL_RAW_COLUMNS[row * ncols]
        ax_axis.set_ylabel(
            row_ylabel_for_row_left_metric(row_left_metric),
            fontsize=MAIN_LABEL,
        )

    for idx, col in enumerate(ALL_RAW_COLUMNS):
        ax = axes_flat[idx]
        series: list[list[float]] = []
        facecolors: list[tuple] = []

        for m in methods:
            vals = prediction_pools.get((m, col), [])
            if vals:
                series.append(vals)
                facecolors.append(colors[m])

        if include_reference:
            ref_vals = reference_pools.get(col, [])
            if ref_vals:
                series.append(ref_vals)
                facecolors.append(REFERENCE_COLOR)

        if not series:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=MAIN_TICK,
            )
            ax.set_title(_METRIC_TITLES.get(col, col), fontsize=MAIN_TITLE)
            _set_row_ylabel(ax, idx)
            continue

        bp = ax.boxplot(series, patch_artist=True, showfliers=False)
        ax.set_xticks(np.arange(1, len(series) + 1))
        ax.tick_params(axis="x", labelbottom=False)
        for patch, fc in zip(bp["boxes"], facecolors):
            patch.set_facecolor(fc)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
        for wk in ("whiskers", "caps", "medians"):
            for artist in bp[wk]:
                artist.set_color("black")
                artist.set_linewidth(0.8)

        ax.set_title(_METRIC_TITLES.get(col, col), fontsize=MAIN_TITLE, fontweight="normal")
        _set_row_ylabel(ax, idx)
        ylim_fixed = fixed_ylim_for_column.get(col)
        if ylim_fixed is not None:
            ax.set_ylim(ylim_fixed[0], ylim_fixed[1])
        # At most YAXIS_MAX_TICKS y-tick marks; nbins = intervals ≈ ticks − 1 (MaxNLocator convention)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=max(YAXIS_MAX_TICKS - 1, 1)))
        ax.tick_params(axis="y", labelsize=YTICK_LABEL)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.align_ylabels([axes[r, 0] for r in range(3)])

    legend_handles = [Patch(facecolor=colors[m], edgecolor="black", label=method_labels[i]) for i, m in enumerate(methods)]
    if include_reference:
        legend_handles.append(
            Patch(facecolor=REFERENCE_COLOR[:3], edgecolor="black", label=REFERENCE_LABEL)
        )
    leg = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(len(legend_handles), 5),
        fontsize=LEGEND_FONT,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fancybox=False,
        borderpad=0.65,
        labelspacing=0.85,
        handlelength=2.4,
        handleheight=1.15,
        handletextpad=0.6,
        borderaxespad=0.35,
    )
    leg.get_frame().set_linewidth(1.75)
    leg.get_frame().set_facecolor("white")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ("png", "pdf", "svg"):
        out = f"{output_prefix}_{suffix}.{ext}"
        plt.savefig(out)
        print(f"Saved: {out}")
    plt.close(fig)


def _mean_and_sample_std(vals: list[float]) -> tuple[float, float] | None:
    if not vals:
        return None
    mu = float(np.mean(vals))
    if len(vals) < 2:
        return mu, 0.0
    return mu, float(np.std(vals, ddof=1))


def _ylim_for_mean_std_bars(
    col: str,
    means: list[float],
    stds: list[float],
    fixed_ylim_for_column: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    bottoms = [m - s for m, s in zip(means, stds)]
    tops = [m + s for m, s in zip(means, stds)]
    span = max(max(tops) - min(bottoms), 1e-9)
    pad = span * 0.06
    lo, hi = min(bottoms) - pad, max(tops) + pad
    fix = fixed_ylim_for_column.get(col)
    if fix is not None:
        lo = min(lo, fix[0])
        hi = max(hi, fix[1])
    return lo, hi


def _draw_mean_std_bar_grid(
    *,
    output_prefix: str,
    methods: tuple[str, ...],
    prediction_pools: dict[tuple[str, str], list[float]],
    reference_pools: dict[str, list[float]],
    include_reference: bool,
    suffix: str,
    row_ylabel_for_row_left_metric: Callable[[str], str],
    fixed_ylim_for_column: dict[str, tuple[float, float]],
) -> None:
    """Bar = mean of pooled values; error bar = sample std (ddof=1). Same layout/legend as boxplot grid."""
    colors = method_colors(methods)
    method_labels = labels_for_methods(methods)

    fig, axes = plt.subplots(3, 4, figsize=(21, 17))
    axes_flat = axes.flatten()
    ncols = 4

    def _set_row_ylabel(ax_axis, subplot_idx: int) -> None:
        if subplot_idx % ncols != 0:
            ax_axis.set_ylabel("")
            return
        row = subplot_idx // ncols
        row_left_metric = ALL_RAW_COLUMNS[row * ncols]
        ax_axis.set_ylabel(
            row_ylabel_for_row_left_metric(row_left_metric),
            fontsize=MAIN_LABEL,
        )

    for idx, col in enumerate(ALL_RAW_COLUMNS):
        ax = axes_flat[idx]
        means: list[float] = []
        stds: list[float] = []
        bar_colors: list = []

        for m in methods:
            vals = prediction_pools.get((m, col), [])
            ms = _mean_and_sample_std(vals)
            if ms is not None:
                means.append(ms[0])
                stds.append(ms[1])
                bar_colors.append(colors[m])

        if include_reference:
            ref_vals = reference_pools.get(col, [])
            ms = _mean_and_sample_std(ref_vals)
            if ms is not None:
                means.append(ms[0])
                stds.append(ms[1])
                bar_colors.append(REFERENCE_COLOR)

        if not means:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=MAIN_TICK,
            )
            ax.set_title(_METRIC_TITLES.get(col, col), fontsize=MAIN_TITLE)
            _set_row_ylabel(ax, idx)
            continue

        n_bars = len(means)
        x = np.arange(n_bars, dtype=float)
        width = min(0.78 / max(n_bars, 1), 0.22)

        ax.bar(
            x,
            means,
            width,
            yerr=stds,
            capsize=4,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.8,
            error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "black"},
        )
        ax.set_xticks(x)
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_title(_METRIC_TITLES.get(col, col), fontsize=MAIN_TITLE, fontweight="normal")
        _set_row_ylabel(ax, idx)
        lo, hi = _ylim_for_mean_std_bars(col, means, stds, fixed_ylim_for_column)
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=max(YAXIS_MAX_TICKS - 1, 1)))
        ax.tick_params(axis="y", labelsize=YTICK_LABEL)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.align_ylabels([axes[r, 0] for r in range(3)])

    legend_handles = [
        Patch(facecolor=colors[m], edgecolor="black", label=method_labels[i])
        for i, m in enumerate(methods)
    ]
    if include_reference:
        legend_handles.append(
            Patch(facecolor=REFERENCE_COLOR[:3], edgecolor="black", label=REFERENCE_LABEL)
        )
    leg = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(len(legend_handles), 5),
        fontsize=LEGEND_FONT,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fancybox=False,
        borderpad=0.65,
        labelspacing=0.85,
        handlelength=2.4,
        handleheight=1.15,
        handletextpad=0.6,
        borderaxespad=0.35,
    )
    leg.get_frame().set_linewidth(1.75)
    leg.get_frame().set_facecolor("white")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ("png", "pdf", "svg"):
        out = f"{output_prefix}_{suffix}.{ext}"
        plt.savefig(out)
        print(f"Saved: {out}")
    plt.close(fig)


def _raw_row_ylabel(row_left_metric: str) -> str:
    return "Å" if row_left_metric in DISTANCE_COLUMNS else "Degrees"


def _std_row_ylabel(row_left_metric: str) -> str:
    return "Std. dev. (Å)" if row_left_metric in DISTANCE_COLUMNS else "Std. dev. (°)"


RAW_FIXED_YLIM: dict[str, tuple[float, float]] = {
    ASP_CHI2_COLUMN: (-80.0, 50.0),
    PHE_CHI2_COLUMN: (55.0, 125.0),
}


def plot_raw_diversity_boxplots_from_csv_dir(
    csv_dir: Path | str,
    output_prefix: str,
    methods: tuple[str, ...],
) -> None:
    """
    Build two 3×4 figures from csvs_for_diversity_analysis: raw metrics, methods as colored boxes.

    Writes:
        {output_prefix}_raw_diversity_boxplot_3x4.{png,pdf,svg}
        {output_prefix}_raw_diversity_boxplot_3x4_with_reference.{png,pdf,svg}
    """
    csv_dir = Path(csv_dir)
    if not csv_dir.is_dir():
        print(f"Warning: diversity CSV directory not found: {csv_dir}")
        return

    methods = tuple(m for m in methods)
    prediction_pools, reference_pools, n_files = collect_raw_diversity_pools(csv_dir, methods)

    if n_files == 0:
        print(f"Warning: no CSV files in {csv_dir}")
        return

    has_predictions = any(len(prediction_pools.get((m, c), [])) > 0 for m in methods for c in ALL_RAW_COLUMNS)
    if not has_predictions:
        print(
            "Warning: no prediction rows found for requested methods in diversity CSVs; "
            "skipping raw diversity box plots."
        )
        return

    _draw_boxplot_grid(
        output_prefix=output_prefix,
        methods=methods,
        prediction_pools=prediction_pools,
        reference_pools=reference_pools,
        include_reference=False,
        suffix="raw_diversity_boxplot_3x4",
        row_ylabel_for_row_left_metric=_raw_row_ylabel,
        fixed_ylim_for_column=RAW_FIXED_YLIM,
    )

    has_ref = any(reference_pools.get(c) for c in ALL_RAW_COLUMNS)
    if has_ref:
        _draw_boxplot_grid(
            output_prefix=output_prefix,
            methods=methods,
            prediction_pools=prediction_pools,
            reference_pools=reference_pools,
            include_reference=True,
            suffix="raw_diversity_boxplot_3x4_with_reference",
            row_ylabel_for_row_left_metric=_raw_row_ylabel,
            fixed_ylim_for_column=RAW_FIXED_YLIM,
        )
    else:
        print("Warning: no reference rows parsed; skipped _with_reference figure.")


def plot_raw_diversity_mean_std_bars_from_csv_dir(
    csv_dir: Path | str,
    output_prefix: str,
    methods: tuple[str, ...],
) -> None:
    """
    Same pooled raw data as box plots, summarized as mean ± sample s (error bars), 3×4 grid.

    Writes:
        {output_prefix}_raw_diversity_mean_std_bars_3x4.{png,pdf,svg}
        {output_prefix}_raw_diversity_mean_std_bars_3x4_with_reference.{png,pdf,svg}
    """
    csv_dir = Path(csv_dir)
    if not csv_dir.is_dir():
        print(f"Warning: diversity CSV directory not found: {csv_dir}")
        return

    methods = tuple(m for m in methods)
    prediction_pools, reference_pools, n_files = collect_raw_diversity_pools(csv_dir, methods)

    if n_files == 0:
        print(f"Warning: no CSV files in {csv_dir}")
        return

    has_predictions = any(len(prediction_pools.get((m, c), [])) > 0 for m in methods for c in ALL_RAW_COLUMNS)
    if not has_predictions:
        print(
            "Warning: no prediction rows found for requested methods in diversity CSVs; "
            "skipping raw diversity mean±std bar plots."
        )
        return

    _draw_mean_std_bar_grid(
        output_prefix=output_prefix,
        methods=methods,
        prediction_pools=prediction_pools,
        reference_pools=reference_pools,
        include_reference=False,
        suffix="raw_diversity_mean_std_bars_3x4",
        row_ylabel_for_row_left_metric=_raw_row_ylabel,
        fixed_ylim_for_column=RAW_FIXED_YLIM,
    )

    has_ref = any(reference_pools.get(c) for c in ALL_RAW_COLUMNS)
    if has_ref:
        _draw_mean_std_bar_grid(
            output_prefix=output_prefix,
            methods=methods,
            prediction_pools=prediction_pools,
            reference_pools=reference_pools,
            include_reference=True,
            suffix="raw_diversity_mean_std_bars_3x4_with_reference",
            row_ylabel_for_row_left_metric=_raw_row_ylabel,
            fixed_ylim_for_column=RAW_FIXED_YLIM,
        )
    else:
        print("Warning: no reference rows parsed; skipped mean±std bars _with_reference figure.")


def plot_internal_std_boxplots_from_csv_dir(
    csv_dir: Path | str,
    output_prefix: str,
    methods: tuple[str, ...],
) -> None:
    """
    Per-system internal diversity: box plot of σ (sample std within each system), 3×4 grid.

    Writes:
        {output_prefix}_internal_std_boxplot_3x4.{png,pdf,svg}
    """
    csv_dir = Path(csv_dir)
    if not csv_dir.is_dir():
        print(f"Warning: diversity CSV directory not found: {csv_dir}")
        return

    methods = tuple(m for m in methods)
    std_pools, n_files = collect_per_system_std_pools(csv_dir, methods)

    if n_files == 0:
        print(f"Warning: no CSV files in {csv_dir}")
        return

    has_std = any(len(std_pools.get((m, c), [])) > 0 for m in methods for c in ALL_RAW_COLUMNS)
    if not has_std:
        print(
            "Warning: no per-system σ (need ≥2 predictions per method per system); "
            "skipping internal std box plots."
        )
        return

    _draw_boxplot_grid(
        output_prefix=output_prefix,
        methods=methods,
        prediction_pools=std_pools,
        reference_pools={},
        include_reference=False,
        suffix="internal_std_boxplot_3x4",
        row_ylabel_for_row_left_metric=_std_row_ylabel,
        fixed_ylim_for_column={},
    )

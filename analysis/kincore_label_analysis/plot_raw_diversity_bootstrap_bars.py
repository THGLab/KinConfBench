#!/usr/bin/env python3
"""
Bootstrap confidence intervals on the pooled mean for diversity-subset raw geometry.

Bar height = sample mean of all pooled values (same preprocessing as
plot_raw_diversity_boxplots). Error bars = percentile bootstrap 95% CI for the mean
(not the sample std). Wide bars; y-axis uses ≥2 Å (distances) or ≥20° (angles) minimum span,
centered on the data so bars sit mid-panel.

Run via plot_results.py --diversity-csv-dir or:

    python -m KinConfBench.analysis.kincore_label_analysis.plot_raw_diversity_bootstrap_bars \\
        --csv-dir .../csvs_for_diversity_analysis -o .../results --methods boltz,chai,protenix
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from KinConfBench.analysis.kincore_label_analysis.analyze_correct_prediction_diversity import (
    ANGLE_COLUMNS,
    DISTANCE_COLUMNS,
)
from KinConfBench.analysis.kincore_label_analysis.plot_raw_diversity_boxplots import (
    ALL_RAW_COLUMNS,
    ASP_CHI2_COLUMN,
    LEGEND_FONT,
    MAIN_LABEL,
    MAIN_TICK,
    MAIN_TITLE,
    REFERENCE_COLOR,
    REFERENCE_LABEL,
    RAW_FIXED_YLIM,
    YAXIS_MAX_TICKS,
    YTICK_LABEL,
    _METRIC_TITLES,
    _raw_row_ylabel,
    collect_raw_diversity_pools,
)
from KinConfBench.analysis.plot_style import labels_for_methods, method_colors, parse_methods_arg

# Bar width in x-data units (bar centers are 0,1,…); must be < 1 so neighbors do not overlap.
_BAR_WIDTH_CAP = 0.93
_BAR_SLOT_MARGIN = 0.05  # min gap between adjacent bar edges

# Minimum y-axis span so means/CIs sit visually centered (not a thin band at one edge)
_MIN_DISTANCE_Y_SPAN_A = 2.0
_MIN_ANGLE_Y_SPAN_DEG = 20.0


def _expand_ylim_min_span(lo: float, hi: float, min_span: float) -> tuple[float, float]:
    if hi - lo >= min_span - 1e-12:
        return lo, hi
    mid = 0.5 * (lo + hi)
    half = 0.5 * min_span
    return mid - half, mid + half


def _bootstrap_mean_ci(
    values: list[float],
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float, float] | None:
    """Return (mean, ci_low, ci_high) for the mean; None if empty."""
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return None
    mu = float(np.mean(arr))
    if n == 1:
        return mu, mu, mu
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    sample_means = np.mean(arr[idx], axis=1)
    lo = float(np.percentile(sample_means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(sample_means, 100.0 * (1.0 - alpha / 2.0)))
    return mu, lo, hi


def _ylim_for_ci_bars(
    col: str,
    means: list[float],
    lows: list[float],
    highs: list[float],
    fixed_ylim_for_column: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    span = max(max(highs) - min(lows), 1e-9)
    pad = span * 0.06
    lo, hi = min(lows) - pad, max(highs) + pad
    fix = fixed_ylim_for_column.get(col)
    if fix is not None:
        # Merging with fixed window via min(lo,fix[0]) / max(hi,fix[1]) forces the full
        # Asp χ2 (−80…50) range; means/CI are often a narrow band → bars look nearly flat.
        # Boxplots still use the fixed window; here we zoom to the data like other panels.
        if col != ASP_CHI2_COLUMN:
            lo = min(lo, fix[0])
            hi = max(hi, fix[1])

    if col in DISTANCE_COLUMNS:
        lo, hi = _expand_ylim_min_span(lo, hi, _MIN_DISTANCE_Y_SPAN_A)
    elif col in ANGLE_COLUMNS:
        lo, hi = _expand_ylim_min_span(lo, hi, _MIN_ANGLE_Y_SPAN_DEG)
    return lo, hi


def _draw_bootstrap_bar_grid(
    *,
    output_prefix: str,
    methods: tuple[str, ...],
    prediction_pools: dict[tuple[str, str], list[float]],
    reference_pools: dict[str, list[float]],
    include_reference: bool,
    suffix: str,
    n_bootstrap: int,
    rng: np.random.Generator,
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
        ax_axis.set_ylabel(_raw_row_ylabel(row_left_metric), fontsize=MAIN_LABEL)

    for idx, col in enumerate(ALL_RAW_COLUMNS):
        ax = axes_flat[idx]
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        bar_colors: list = []

        for m in methods:
            vals = prediction_pools.get((m, col), [])
            b = _bootstrap_mean_ci(vals, n_bootstrap, rng)
            if b is not None:
                mu, lo, hi = b
                means.append(mu)
                lows.append(lo)
                highs.append(hi)
                bar_colors.append(colors[m])

        if include_reference:
            ref_vals = reference_pools.get(col, [])
            b = _bootstrap_mean_ci(ref_vals, n_bootstrap, rng)
            if b is not None:
                mu, lo, hi = b
                means.append(mu)
                lows.append(lo)
                highs.append(hi)
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
        width = min(_BAR_WIDTH_CAP, 1.0 - _BAR_SLOT_MARGIN)

        yerr = np.array([[m - lo for m, lo in zip(means, lows)], [hi - m for m, hi in zip(means, highs)]])

        ax.bar(
            x,
            means,
            width,
            yerr=yerr,
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
        lo_y, hi_y = _ylim_for_ci_bars(col, means, lows, highs, RAW_FIXED_YLIM)
        ax.set_ylim(lo_y, hi_y)
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


def plot_raw_diversity_bootstrap_bars_from_csv_dir(
    csv_dir: Path | str,
    output_prefix: str,
    methods: tuple[str, ...],
    *,
    n_bootstrap: int = 2000,
    random_seed: int = 42,
) -> None:
    """
    Mean bars with bootstrap 95% CI on the mean (pooled values per method / KinConfBench).

    Writes:
        {output_prefix}_raw_diversity_bootstrap_bars_3x4.{png,pdf,svg}
        {output_prefix}_raw_diversity_bootstrap_bars_3x4_with_reference.{png,pdf,svg}
    """
    csv_dir = Path(csv_dir)
    if not csv_dir.is_dir():
        print(f"Warning: diversity CSV directory not found: {csv_dir}")
        return

    methods = tuple(m for m in methods)
    rng = np.random.default_rng(random_seed)
    prediction_pools, reference_pools, n_files = collect_raw_diversity_pools(csv_dir, methods)

    if n_files == 0:
        print(f"Warning: no CSV files in {csv_dir}")
        return

    has_predictions = any(len(prediction_pools.get((m, c), [])) > 0 for m in methods for c in ALL_RAW_COLUMNS)
    if not has_predictions:
        print(
            "Warning: no prediction rows found for requested methods in diversity CSVs; "
            "skipping bootstrap bar plots."
        )
        return

    _draw_bootstrap_bar_grid(
        output_prefix=output_prefix,
        methods=methods,
        prediction_pools=prediction_pools,
        reference_pools=reference_pools,
        include_reference=False,
        suffix="raw_diversity_bootstrap_bars_3x4",
        n_bootstrap=n_bootstrap,
        rng=rng,
    )

    has_ref = any(reference_pools.get(c) for c in ALL_RAW_COLUMNS)
    if has_ref:
        _draw_bootstrap_bar_grid(
            output_prefix=output_prefix,
            methods=methods,
            prediction_pools=prediction_pools,
            reference_pools=reference_pools,
            include_reference=True,
            suffix="raw_diversity_bootstrap_bars_3x4_with_reference",
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
    else:
        print("Warning: no reference rows parsed; skipped bootstrap bars _with_reference figure.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CI mean bars for raw diversity CSVs.")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        required=True,
        help="csvs_for_diversity_analysis directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output path prefix (same as plot_results -o)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated methods (default boltz,chai,protenix)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Bootstrap resamples per bar (default 2000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    args = parser.parse_args()
    methods = parse_methods_arg(args.methods)
    plot_raw_diversity_bootstrap_bars_from_csv_dir(
        args.csv_dir,
        args.output,
        methods,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()

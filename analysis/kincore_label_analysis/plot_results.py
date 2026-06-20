#!/usr/bin/env python3
"""
Generate publication-quality plots for accuracy and diversity results.

Creates:
1. Success@K figure: if --geometric-base-dir merges, pooled ligand RMSD + Success@K (replaces
   Top-K-only); else standalone Success@K curve. Always also writes the first-correct-rank CDF.
2. Optional wide combined figure: Success@K (left) + per-method ligand RMSD (right), same flags.
3. Number of correct predictions distribution per system
4. Top-K accuracy bar plots with error bars
5. Average diversity comparison with confidence intervals
6. Individual distance / angular metric diversity plots
7. Optional raw pooled box plots, raw mean±std bar plots, raw bootstrap-CI mean bars, and per-system σ box plots (3×4) from csvs_for_diversity_analysis (see plot_raw_diversity_boxplots.py, plot_raw_diversity_bootstrap_bars.py)

Usage:
    python plot_results.py --accuracy-pkl run_output/kincore_label_successful/comprehensive_results_data.pkl \\
                           --diversity run_output/kincore_label_successful/diversity_results_diversity_summary.json \\
                           -o run_output/figures_successful/results
"""

import json
import math
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter, MultipleLocator

from KinConfBench.analysis.plot_style import (
    PRIMARY_METHODS,
    parse_methods_arg,
    method_colors,
    labels_for_methods,
)
from KinConfBench.analysis.kincore_label_analysis.plot_num_correct_distribution import (
    plot_num_correct_distribution as plot_num_correct_distribution_impl,
)
from KinConfBench.analysis.kincore_label_analysis.geometric_correctness_correlation import (
    load_merged_data,
    stems_in_filtered_dataset,
)
from KinConfBench.analysis.kincore_label_analysis.plot_raw_diversity_bootstrap_bars import (
    plot_raw_diversity_bootstrap_bars_from_csv_dir,
)
from KinConfBench.analysis.kincore_label_analysis.plot_raw_diversity_boxplots import (
    plot_internal_std_boxplots_from_csv_dir,
    plot_raw_diversity_boxplots_from_csv_dir,
    plot_raw_diversity_mean_std_bars_from_csv_dir,
)


# Base typography (+2 pt vs earlier defaults) for main result figures
MAIN_LABEL = 16
MAIN_TITLE = 18
MAIN_TICK = 14
MAIN_LEGEND = 13
MAIN_BAR_TEXT = 12

plt.rcParams['font.size'] = MAIN_TICK
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = MAIN_LABEL
plt.rcParams['axes.titlesize'] = MAIN_TITLE
plt.rcParams['xtick.labelsize'] = MAIN_TICK
plt.rcParams['ytick.labelsize'] = MAIN_TICK
plt.rcParams['legend.fontsize'] = MAIN_LEGEND
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def _wilson_ci(successes, n, z=1.959963984540054):
    """Fast 95% binomial CI (Wilson score interval). Returns (lower%, upper%)."""
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n)))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return 100.0 * lo, 100.0 * hi


def _first_success_rank(conf_dict, max_k=20):
    """Return 1-based rank of the first correct prediction within top-K."""
    for r in range(0, max_k):
        if conf_dict.get(r, 0):
            return r + 1
    return None


def plot_success_vs_k_curve_only(pkl_path, output_prefix, max_k=20, methods=PRIMARY_METHODS):
    """Stand-alone Success@K curves (used when no geometric merge is available)."""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    per = obj["per_file_stats_with_conf"]
    methods = tuple(m for m in methods if m in per)
    if not methods:
        raise RuntimeError("No matching methods in pickle.")
    colors = method_colors(methods)
    labels = labels_for_methods(methods)

    file_sets = [set(per[m].keys()) for m in methods]
    common_files = sorted(set.intersection(*file_sets)) if file_sets else []
    if not common_files:
        raise RuntimeError("No common files across all methods; cannot compute paired curves.")

    ks = list(range(1, max_k + 1))
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    for method, label in zip(methods, labels):
        y = []
        for k in ks:
            successes = 0
            for fn in common_files:
                conf_dict = per[method][fn]
                ok = 0
                for r in range(0, k):
                    if conf_dict.get(r, 0):
                        ok = 1
                        break
                successes += ok
            n = len(common_files)
            mean = 100.0 * successes / n
            y.append(mean)

        ax.plot(ks, y, label=label, color=colors[method], linewidth=2.5)

    ax.set_xlabel("K (Top-K Predictions)", fontsize=MAIN_LABEL)
    ax.set_ylabel("System Success Rate (%)", fontsize=MAIN_LABEL)
    ax.set_title("Success@K Curve (All Labels Correct)", fontsize=MAIN_TITLE)
    ax.tick_params(axis="both", labelsize=MAIN_TICK)
    ax.set_xlim(1, max_k)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(45, 92)
    ax.grid(alpha=0.3, linestyle="--")
    leg = ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(MAIN_LEGEND)

    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{output_prefix}_success_vs_k.{ext}")
    print(f"Saved: {output_prefix}_success_vs_k.png/pdf/svg")
    plt.close()


def plot_first_correct_rank_cdf(pkl_path, output_prefix, max_k=20, methods=PRIMARY_METHODS):
    """CDF of rank of first correct prediction (among top-max_k)."""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    per = obj["per_file_stats_with_conf"]
    methods = tuple(m for m in methods if m in per)
    if not methods:
        raise RuntimeError("No matching methods in pickle.")
    colors = method_colors(methods)
    labels = labels_for_methods(methods)

    file_sets = [set(per[m].keys()) for m in methods]
    common_files = sorted(set.intersection(*file_sets)) if file_sets else []
    if not common_files:
        raise RuntimeError("No common files across all methods; cannot compute paired curves.")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, max_k + 1)

    for method, label in zip(methods, labels):
        ranks = []
        for fn in common_files:
            r = _first_success_rank(per[method][fn], max_k=max_k)
            if r is not None:
                ranks.append(r)

        ranks = np.array(ranks, dtype=int)
        if len(ranks) == 0:
            continue
        cdf = [(ranks <= k).mean() * 100.0 for k in x]
        fail_rate = 100.0 * (1.0 - len(ranks) / len(common_files))

        ax.step(x, cdf, where="post", label=f"{label} (fail@{max_k}: {fail_rate:.1f}%)",
                color=colors[method], linewidth=2)

    ax.set_xlabel("Rank of first correct prediction (lower is better)", fontsize=MAIN_LABEL)
    ax.set_ylabel("Fraction of systems (%)", fontsize=MAIN_LABEL)
    ax.set_title(f"CDF of First-Correct Rank (within Top-{max_k})", fontsize=MAIN_TITLE)
    ax.tick_params(axis="both", labelsize=MAIN_TICK)
    ax.set_xlim(1, max_k)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, linestyle="--")
    leg = ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(MAIN_LEGEND)

    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{output_prefix}_first_correct_rank_cdf.{ext}")
    print(f"Saved: {output_prefix}_first_correct_rank_cdf.png/pdf/svg")
    plt.close()


def plot_success_vs_k_and_first_rank_cdf(pkl_path, output_prefix, max_k=20, methods=PRIMARY_METHODS):
    """Success@K curve only + first-correct CDF (legacy entrypoint; pipeline uses split + stacked)."""
    plot_success_vs_k_curve_only(pkl_path, output_prefix, max_k=max_k, methods=methods)
    plot_first_correct_rank_cdf(pkl_path, output_prefix, max_k=max_k, methods=methods)


def _plot_pooled_ligand_rmsd_correct_vs_incorrect(ax, merged: pd.DataFrame) -> None:
    """Pooled ligand RMSD boxplots (correct vs incorrect); neutral boxes; y in [0, 2] Å, step 0.5."""
    metric = "ligand_rmsd"
    rmsd_cap = 2.0
    df = merged.dropna(subset=[metric])
    correct_vals = df[df["is_correct"]][metric].values
    incorrect_vals = df[~df["is_correct"]][metric].values
    if len(correct_vals) == 0 or len(incorrect_vals) == 0:
        ax.text(
            0.5,
            0.5,
            "Insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=MAIN_TICK,
        )
        ax.set_title("Conformational Correctness", fontsize=MAIN_TITLE, fontweight="normal")
        return

    try:
        bp = ax.boxplot(
            [correct_vals, incorrect_vals],
            positions=[0, 1],
            tick_labels=["Correct", "Incorrect"],
            patch_artist=True,
            showmeans=False,
        )
    except TypeError:
        bp = ax.boxplot(
            [correct_vals, incorrect_vals],
            positions=[0, 1],
            labels=["Correct", "Incorrect"],
            patch_artist=True,
            showmeans=False,
        )

    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    for elem in ("whiskers", "caps", "fliers", "medians"):
        if elem not in bp:
            continue
        for obj in bp[elem]:
            obj.set_color("0.15")
            obj.set_linewidth(1.0)

    ax.set_xticklabels(["Correct", "Incorrect"], fontsize=MAIN_TICK)
    ax.set_ylabel("Ligand RMSD (Å)", fontsize=MAIN_LABEL)
    ax.set_title("Conformational Correctness", fontsize=MAIN_TITLE, fontweight="normal")
    ax.tick_params(axis="y", labelsize=MAIN_TICK)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0.0, rmsd_cap)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))


def plot_success_vs_k_ligand_rmsd_stacked(
    pkl_path,
    merged_geometric: pd.DataFrame,
    output_prefix: str,
    *,
    max_k: int = 20,
    methods: tuple[str, ...] = PRIMARY_METHODS,
) -> None:
    """
    Success@K curve only (y-axis 60–85%, ticks 65/75/85; no title).
    ``merged_geometric`` is kept for call-site compatibility; the former left-hand
    ligand RMSD panel is no longer drawn.

    Writes ``{prefix}_success_vs_k.*`` and
    ``{prefix}_success_vs_k_ligand_rmsd_stacked.*`` (same content, legacy name).
    """
    _ = merged_geometric  # legacy arg; pooled RMSD panel removed
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    per = obj["per_file_stats_with_conf"]
    methods = tuple(m for m in methods if m in per)
    if not methods:
        raise RuntimeError("No matching methods in pickle.")

    colors = method_colors(methods)
    labels = labels_for_methods(methods)

    file_sets = [set(per[m].keys()) for m in methods]
    common_files = sorted(set.intersection(*file_sets)) if file_sets else []
    if not common_files:
        raise RuntimeError("No common files across all methods; cannot compute paired curves.")

    fig, ax_curve = plt.subplots(figsize=(5.25, 5.6))

    ks = list(range(1, max_k + 1))
    for method, label in zip(methods, labels):
        y = []
        for k in ks:
            successes = 0
            for fn in common_files:
                conf_dict = per[method][fn]
                ok = 0
                for r in range(0, k):
                    if conf_dict.get(r, 0):
                        ok = 1
                        break
                successes += ok
            n = len(common_files)
            y.append(100.0 * successes / n)
        ax_curve.plot(ks, y, label=label, color=colors[method], linewidth=2.5)

    ax_curve.set_xlabel("K (Top-K Predictions)", fontsize=MAIN_LABEL)
    ax_curve.set_ylabel("System Success Rate (%)", fontsize=MAIN_LABEL)
    ax_curve.tick_params(axis="both", labelsize=MAIN_TICK)
    ax_curve.set_xlim(1, max_k)
    ax_curve.set_xticks([1, 5, 10, 15, 20])
    ax_curve.set_ylim(60, 85)
    ax_curve.set_yticks([65, 75, 85])
    ax_curve.grid(alpha=0.3, linestyle="--")
    leg = ax_curve.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(MAIN_LEGEND)

    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{output_prefix}_success_vs_k.{ext}")
        plt.savefig(f"{output_prefix}_success_vs_k_ligand_rmsd_stacked.{ext}")
    print(
        f"Saved: {output_prefix}_success_vs_k.png/pdf/svg "
        f"(and _success_vs_k_ligand_rmsd_stacked, same figure)"
    )
    plt.close()


def _plot_ligand_rmsd_correct_vs_incorrect_axes(
    axes: list,
    merged: pd.DataFrame,
    methods: tuple[str, ...],
    colors: dict,
    labels: list[str],
) -> None:
    """SI-style boxplots: ligand RMSD for KinCoRe-correct vs incorrect, one panel per method."""
    metric = "ligand_rmsd"
    df = merged.dropna(subset=[metric])
    for idx, (method, label) in enumerate(zip(methods, labels)):
        ax = axes[idx]
        mdf = df[df["method"] == method]
        if mdf.empty:
            ax.set_visible(False)
            continue
        correct_vals = mdf[mdf["is_correct"]][metric].values
        incorrect_vals = mdf[~mdf["is_correct"]][metric].values
        if len(correct_vals) == 0 or len(incorrect_vals) == 0:
            ax.text(
                0.5, 0.5, "Insufficient data",
                ha="center", va="center", transform=ax.transAxes, fontsize=MAIN_TICK,
            )
            ax.set_title(label, fontsize=MAIN_TITLE, fontweight="normal")
            ax.tick_params(labelsize=MAIN_TICK)
            continue
        try:
            bp = ax.boxplot(
                [correct_vals, incorrect_vals],
                positions=[0, 1],
                tick_labels=["Correct", "Incorrect"],
                patch_artist=True,
                showmeans=True,
            )
        except TypeError:
            bp = ax.boxplot(
                [correct_vals, incorrect_vals],
                positions=[0, 1],
                labels=["Correct", "Incorrect"],
                patch_artist=True,
                showmeans=True,
            )
        c = colors[method]
        for patch in bp["boxes"]:
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for elem in ("whiskers", "caps", "fliers", "means", "medians"):
            if elem in bp:
                for obj in bp[elem]:
                    obj.set_color(c)
        ax.set_xticklabels(["Correct", "Incorrect"], fontsize=MAIN_TICK)
        if idx == 0:
            ax.set_ylabel("Ligand RMSD (Å)", fontsize=MAIN_LABEL)
        ax.set_title(label, fontsize=MAIN_TITLE, fontweight="normal")
        ax.tick_params(axis="y", labelsize=MAIN_TICK)
        ax.grid(axis="y", alpha=0.3, linestyle="--")


def plot_success_vs_k_with_ligand_rmsd(
    pkl_path,
    merged_geometric: pd.DataFrame,
    output_prefix: str,
    max_k: int = 20,
    methods: tuple[str, ...] = PRIMARY_METHODS,
) -> None:
    """
    Wide figure: Success@K curve (left) and SI-style ligand RMSD boxplots by
    correctness (right, one column per method).
    Stacked RMSD+Success@K is written as ``results_success_vs_k`` when geometric merge succeeds.
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    per = obj["per_file_stats_with_conf"]
    methods = tuple(m for m in methods if m in per)
    if not methods:
        raise RuntimeError("No matching methods in pickle.")
    colors = method_colors(methods)
    labels = labels_for_methods(methods)

    file_sets = [set(per[m].keys()) for m in methods]
    common_files = sorted(set.intersection(*file_sets)) if file_sets else []
    if not common_files:
        raise RuntimeError("No common files across all methods; cannot compute paired curves.")

    merged_g = merged_geometric[merged_geometric["method"].isin(methods)].copy()
    if merged_g.empty:
        print("Skipping success_vs_k_with_ligand_rmsd: no geometric rows for selected methods.")
        return

    # Success@K (left) + ligand RMSD boxplots (right), single row.
    fig = plt.figure(figsize=(18, 6.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.35], wspace=0.28)
    ax_left = fig.add_subplot(gs[0])
    inner = gridspec.GridSpecFromSubplotSpec(1, len(methods), subplot_spec=gs[1], wspace=0.35)
    axes_right = []
    for i in range(len(methods)):
        ax = fig.add_subplot(inner[0, i], sharey=axes_right[0] if axes_right else None)
        axes_right.append(ax)

    ks = list(range(1, max_k + 1))
    for method, label in zip(methods, labels):
        y = []
        for k in ks:
            successes = 0
            for fn in common_files:
                conf_dict = per[method][fn]
                ok = 0
                for r in range(0, k):
                    if conf_dict.get(r, 0):
                        ok = 1
                        break
                successes += ok
            n = len(common_files)
            y.append(100.0 * successes / n)
        ax_left.plot(ks, y, label=label, color=colors[method], linewidth=2.5)

    ax_left.set_xlabel("K (Top-K Predictions)", fontsize=MAIN_LABEL)
    ax_left.set_ylabel("System Success Rate (%)", fontsize=MAIN_LABEL)
    ax_left.set_title("Success@K (all labels correct)", fontsize=MAIN_TITLE)
    ax_left.tick_params(axis="both", labelsize=MAIN_TICK)
    ax_left.set_xlim(1, max_k)
    ax_left.set_xticks([1, 5, 10, 15, 20])
    ax_left.set_ylim(45, 92)
    ax_left.grid(alpha=0.3, linestyle="--")
    leg = ax_left.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(MAIN_LEGEND)

    _plot_ligand_rmsd_correct_vs_incorrect_axes(axes_right, merged_g, methods, colors, labels)
    for ax in axes_right[1:]:
        ax.tick_params(labelleft=False)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.12, wspace=0.28)
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{output_prefix}_success_vs_k_with_ligand_rmsd.{ext}")
    print(f"Saved: {output_prefix}_success_vs_k_with_ligand_rmsd.png/pdf/svg")
    plt.close()


def plot_num_correct_distribution(pkl_path, output_prefix, max_k=20, methods=PRIMARY_METHODS):
    """Plot distribution of number of correct predictions per system, per method (full common set)."""
    plot_num_correct_distribution_impl(
        pkl_path,
        output_prefix,
        max_k=max_k,
        subset_files=None,
        methods=methods,
        hist_x_min=0,
    )


def plot_topk_accuracy(topk_data, output_prefix, methods=PRIMARY_METHODS):
    """Plot top-K system success rates with confidence intervals."""

    k_values = ['top1', 'top5', 'top10', 'top20']
    k_labels = ['Top-1', 'Top-5', 'Top-10', 'Top-20']

    methods = tuple(
        m for m in methods
        if m in topk_data.get('top1', {})
    )
    if not methods:
        raise RuntimeError("No matching methods in top-K JSON.")
    colors = method_colors(methods)
    labels = labels_for_methods(methods)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(k_values))
    n_m = len(methods)
    width = min(0.22, 0.9 / (n_m + 0.5))

    for i, method in enumerate(methods):
        success_rates = []
        ci_widths = []

        for k in k_values:
            stats = topk_data[k][method]
            success_rates.append(stats['success_rate'])
            ci_widths.append(stats['ci_width'] / 2)

        offset = (i - (n_m - 1) / 2.0) * width
        ax.bar(x + offset, success_rates, width,
               label=labels[i],
               color=colors[method],
               yerr=ci_widths,
               capsize=4,
               alpha=0.8,
               edgecolor='black',
               linewidth=0.5)

    ax.set_xlabel('Top-K Predictions', fontsize=MAIN_LABEL)
    ax.set_ylabel('System Success Rate (%)', fontsize=MAIN_LABEL)
    ax.set_title('Top-K System-Level Success Rates (All Labels Correct)', fontsize=MAIN_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(k_labels, fontsize=MAIN_TICK)
    ax.tick_params(axis='y', labelsize=MAIN_TICK)
    leg = ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(MAIN_LEGEND)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

    for i, method in enumerate(methods):
        offset = (i - (n_m - 1) / 2.0) * width
        for j, k in enumerate(k_values):
            rate = topk_data[k][method]['success_rate']
            ax.text(j + offset, rate + 2, f'{rate:.1f}',
                   ha='center', va='bottom', fontsize=MAIN_BAR_TEXT, fontweight='normal')

    plt.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        plt.savefig(f"{output_prefix}_topk_accuracy.{ext}")
    print(f"Saved: {output_prefix}_topk_accuracy.png/pdf/svg")
    plt.close()


def _set_average_diversity_yticks(ax, y_hi: float, max_ticks: int = 31) -> None:
    """Y ticks in steps of 0.01 when possible (0.00, 0.01, 0.02, …); coarser step if needed."""
    n01 = int(np.ceil(y_hi / 0.01)) + 1
    if n01 <= max_ticks:
        step = 0.01
    else:
        step = max(0.01, np.ceil(y_hi / (max_ticks - 1) / 0.01) * 0.01)
    ticks = np.arange(0, y_hi + step * 0.5, step)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))


def plot_average_diversity(diversity_data, output_prefix, methods=PRIMARY_METHODS):
    """Plot average distance and angular diversity with confidence intervals."""

    methods = tuple(m for m in methods if m in diversity_data)
    if not methods:
        raise RuntimeError("No matching methods in diversity JSON.")
    colors = method_colors(methods)
    labels = labels_for_methods(methods)

    distance_stats = {}
    angular_stats = {}

    for method in methods:
        dist_means = []
        for key, stats in diversity_data[method].items():
            if key.startswith('distance_cv_'):
                dist_means.append(stats['mean'])

        if dist_means:
            distance_stats[method] = {
                'mean': np.mean(dist_means),
                'values': dist_means
            }

        ang_means = []
        for key, stats in diversity_data[method].items():
            if key.startswith('angular_variance_'):
                ang_means.append(stats['mean'])

        if ang_means:
            angular_stats[method] = {
                'mean': np.mean(ang_means),
                'values': ang_means
            }

    def bootstrap_ci(values, n_bootstrap=1000):
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))
        bootstrap_means = np.sort(bootstrap_means)
        ci_lower = bootstrap_means[int(0.025 * n_bootstrap)]
        ci_upper = bootstrap_means[int(0.975 * n_bootstrap)]
        return ci_lower, ci_upper

    for method in methods:
        if method in distance_stats:
            ci_lower, ci_upper = bootstrap_ci(distance_stats[method]['values'])
            distance_stats[method]['ci_lower'] = ci_lower
            distance_stats[method]['ci_upper'] = ci_upper

        if method in angular_stats:
            ci_lower, ci_upper = bootstrap_ci(angular_stats[method]['values'])
            angular_stats[method]['ci_lower'] = ci_lower
            angular_stats[method]['ci_upper'] = ci_upper

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(methods))
    width = 0.6

    means = [distance_stats[m]['mean'] for m in methods]
    errors = [(distance_stats[m]['mean'] - distance_stats[m]['ci_lower'],
               distance_stats[m]['ci_upper'] - distance_stats[m]['mean'])
              for m in methods]
    errors = np.array(errors).T

    ax1.bar(x, means, width,
            color=[colors[m] for m in methods],
            yerr=errors,
            capsize=5,
            alpha=0.8,
            edgecolor='black',
            linewidth=1)

    tips1 = np.asarray(means, dtype=float) + errors[1, :]
    y1_hi = float(np.max(tips1)) * 1.2 + 1e-15
    y1_hi = np.ceil(y1_hi * 100) / 100
    ax1.set_ylim(0, y1_hi)
    _set_average_diversity_yticks(ax1, y1_hi)

    ax1.set_xlabel('Method', fontsize=MAIN_LABEL)
    ax1.set_ylabel('Average Coefficient of Variation', fontsize=MAIN_LABEL)
    ax1.set_title('Average Distance Diversity', fontsize=MAIN_TITLE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=MAIN_TICK)
    ax1.tick_params(axis='y', labelsize=MAIN_TICK)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    for i, (mean, err) in enumerate(zip(means, errors.T)):
        ax1.text(i, mean + err[1] + 0.001, f'{mean:.3f}',
                ha='center', va='bottom', fontsize=MAIN_BAR_TEXT, fontweight='normal')

    means = [angular_stats[m]['mean'] for m in methods]
    errors = [(angular_stats[m]['mean'] - angular_stats[m]['ci_lower'],
               angular_stats[m]['ci_upper'] - angular_stats[m]['mean'])
              for m in methods]
    errors = np.array(errors).T

    ax2.bar(x, means, width,
            color=[colors[m] for m in methods],
            yerr=errors,
            capsize=5,
            alpha=0.8,
            edgecolor='black',
            linewidth=1)

    tips2 = np.asarray(means, dtype=float) + errors[1, :]
    y2_hi = float(np.max(tips2)) * 1.2 + 1e-15
    y2_hi = np.ceil(y2_hi * 100) / 100
    ax2.set_ylim(0, y2_hi)
    _set_average_diversity_yticks(ax2, y2_hi)

    ax2.set_xlabel('Method', fontsize=MAIN_LABEL)
    ax2.set_ylabel('Average Circular Variance', fontsize=MAIN_LABEL)
    ax2.set_title('Average Angular Diversity', fontsize=MAIN_TITLE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=MAIN_TICK)
    ax2.tick_params(axis='y', labelsize=MAIN_TICK)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for i, (mean, err) in enumerate(zip(means, errors.T)):
        ax2.text(i, mean + err[1] + 0.001, f'{mean:.3f}',
                ha='center', va='bottom', fontsize=MAIN_BAR_TEXT, fontweight='normal')

    plt.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        plt.savefig(f"{output_prefix}_average_diversity.{ext}")
    print(f"Saved: {output_prefix}_average_diversity.png/pdf/svg")
    plt.close()


def plot_distance_diversity_detailed(diversity_data, output_prefix, methods=PRIMARY_METHODS):
    """Plot individual distance metrics with confidence intervals."""

    methods = tuple(m for m in methods if m in diversity_data)
    if not methods:
        raise RuntimeError("No matching methods in diversity JSON.")
    colors = method_colors(methods)
    labels = labels_for_methods(methods)

    distance_metrics = {}
    for method in methods:
        for key, stats in diversity_data[method].items():
            if key.startswith('distance_cv_'):
                metric_name = key.replace('distance_cv_', '')
                if metric_name not in distance_metrics:
                    distance_metrics[metric_name] = {}
                distance_metrics[metric_name][method] = stats

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metric_titles = {
        'lys_glu_distance': 'Lys-Glu Distance',
        'saltbridge_distance': 'Salt Bridge Distance',
        'spine_distance': 'Spine Distance',
        'dfg_hrd_distance': 'DFG-HRD Distance'
    }

    for idx, (metric_name, method_stats) in enumerate(sorted(distance_metrics.items())):
        ax = axes[idx]
        present = tuple(m for m in methods if m in method_stats)
        x = np.arange(len(present))
        width = 0.6

        means = [method_stats[m]['mean'] for m in present]
        ci_lowers = [method_stats[m]['ci_lower'] for m in present]
        ci_uppers = [method_stats[m]['ci_upper'] for m in present]

        errors = [(means[i] - ci_lowers[i], ci_uppers[i] - means[i])
                  for i in range(len(present))]
        errors = np.array(errors).T

        ax.bar(x, means, width,
               color=[colors[m] for m in present],
               yerr=errors,
               capsize=4,
               alpha=0.8,
               edgecolor='black',
               linewidth=1)

        title = metric_titles.get(metric_name, metric_name)
        ax.set_title(title, fontsize=MAIN_TITLE, fontweight='normal')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_for_methods(present), rotation=0, fontsize=MAIN_TICK)
        ax.set_ylabel('Coefficient of Variation', fontsize=MAIN_LABEL)
        ax.tick_params(axis='y', labelsize=MAIN_TICK)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for i, (mean, err) in enumerate(zip(means, errors.T)):
            ax.text(i, mean + err[1] * 1.05, f'{mean:.4f}',
                   ha='center', va='bottom', fontsize=MAIN_BAR_TEXT, fontweight='normal')

    plt.suptitle('Distance Diversity Metrics (Coefficient of Variation)',
                 fontsize=MAIN_TITLE, fontweight='normal', y=0.995)
    plt.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        plt.savefig(f"{output_prefix}_distance_diversity_detailed.{ext}")
    print(f"Saved: {output_prefix}_distance_diversity_detailed.png/pdf/svg")
    plt.close()


def plot_angular_diversity_detailed(diversity_data, output_prefix, methods=PRIMARY_METHODS):
    """Plot individual angular metrics with confidence intervals."""

    methods = tuple(m for m in methods if m in diversity_data)
    if not methods:
        raise RuntimeError("No matching methods in diversity JSON.")
    colors = method_colors(methods)

    angular_metrics = {}
    for method in methods:
        for key, stats in diversity_data[method].items():
            if key.startswith('angular_variance_'):
                metric_name = key.replace('angular_variance_', '')
                if metric_name not in angular_metrics:
                    angular_metrics[metric_name] = {}
                angular_metrics[metric_name][method] = stats

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    metric_titles = {
        'asp_phi': 'Asp \u03c6',
        'asp_psi': 'Asp \u03c8',
        'asp_chi1': 'Asp \u03c71',
        'asp_chi2': 'Asp \u03c72',
        'phe_phi': 'Phe \u03c6',
        'phe_psi': 'Phe \u03c8',
        'phe_chi1': 'Phe \u03c71',
        'phe_chi2': 'Phe \u03c72'
    }

    for idx, (metric_name, method_stats) in enumerate(sorted(angular_metrics.items())):
        ax = axes[idx]
        present = tuple(m for m in methods if m in method_stats)
        x = np.arange(len(present))
        width = 0.6

        means = [method_stats[m]['mean'] for m in present]
        ci_lowers = [method_stats[m]['ci_lower'] for m in present]
        ci_uppers = [method_stats[m]['ci_upper'] for m in present]

        errors = [(means[i] - ci_lowers[i], ci_uppers[i] - means[i])
                  for i in range(len(present))]
        errors = np.array(errors).T

        ax.bar(x, means, width,
               color=[colors[m] for m in present],
               yerr=errors,
               capsize=4,
               alpha=0.8,
               edgecolor='black',
               linewidth=1)

        title = metric_titles.get(metric_name, metric_name)
        ax.set_title(title, fontsize=MAIN_TITLE, fontweight='normal')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_for_methods(present), rotation=0, fontsize=MAIN_TICK)
        ax.set_ylabel('Circular Variance', fontsize=MAIN_LABEL)
        ax.tick_params(axis='y', labelsize=MAIN_TICK)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for i, (mean, err) in enumerate(zip(means, errors.T)):
            ax.text(i, mean + err[1] * 1.05, f'{mean:.4f}',
                   ha='center', va='bottom', fontsize=MAIN_BAR_TEXT, fontweight='normal')

    plt.suptitle('Angular Diversity Metrics (Circular Variance)',
                 fontsize=MAIN_TITLE, fontweight='normal', y=0.995)
    plt.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        plt.savefig(f"{output_prefix}_angular_diversity_detailed.{ext}")
    print(f"Saved: {output_prefix}_angular_diversity_detailed.png/pdf/svg")
    plt.close()


def prepare_geometric_merged_for_plots(
    base_dir: Path,
    restrict_dataset: Path | None,
    success_only: bool,
    methods: tuple[str, ...],
) -> pd.DataFrame | None:
    """
    Load analysis+geometric merge (same filters as geometric_correctness_correlation).
    Returns None if base_dir is missing or merge is empty.
    """
    base_dir = Path(base_dir).resolve()
    if not base_dir.is_dir():
        print(f"Warning: --geometric-base-dir not found: {base_dir}")
        return None
    merged = load_merged_data(base_dir)
    if merged.empty:
        print("Warning: geometric merge produced no rows.")
        return None
    if restrict_dataset is not None:
        rdir = Path(restrict_dataset).resolve()
        if not rdir.is_dir():
            print(f"Warning: --geometric-restrict-dataset not found: {rdir}")
        else:
            stem_allow = stems_in_filtered_dataset(rdir)
            merged = merged[merged["stem"].isin(stem_allow)].copy()
    if success_only:
        merged = merged.dropna(subset=["lddt_ca", "lddt_pli", "ligand_rmsd"])
        mask = (
            (merged["lddt_ca"] >= 0.7)
            & (merged["lddt_pli"] >= 0.8)
            & (merged["ligand_rmsd"] < 2.0)
        )
        merged = merged[mask].copy()
    merged = merged[merged["method"].isin(methods)].copy()
    if merged.empty:
        print("Warning: no geometric rows after method / success filters.")
        return None
    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Generate plots for accuracy and diversity results'
    )
    parser.add_argument(
        '--accuracy',
        help='Path to top-K accuracy JSON file (comprehensive_results_statistics.json)',
        default='run_output/kincore_label_successful/comprehensive_results_statistics.json',
    )
    parser.add_argument(
        '--accuracy-pkl',
        help='Path to pickle with per-file per-rank correctness',
        default='run_output/kincore_label_successful/comprehensive_results_data.pkl',
    )
    parser.add_argument(
        '--diversity',
        help='Path to diversity summary JSON file (diversity_results_diversity_summary.json)',
        default='run_output/kincore_label_successful/diversity_results_diversity_summary.json',
    )
    parser.add_argument(
        '-o', '--output',
        help='Output prefix for plot files',
        default='run_output/figures_successful/results'
    )
    parser.add_argument(
        '--methods',
        type=str,
        default=None,
        help=(
            'Comma-separated methods (default: boltz,chai,protenix). '
            'Comma-separated method keys (default: boltz,chai,protenix).'
        ),
    )
    parser.add_argument(
        '--geometric-base-dir',
        type=Path,
        default=None,
        help=(
            'KinConfBench root with gene/analysis_csvs and geometric_analysis_csvs. '
            'If set with --accuracy-pkl, also writes results_success_vs_k.* (stacked RMSD+curve) '
            'and results_success_vs_k_with_ligand_rmsd.* (wide layout).'
        ),
    )
    parser.add_argument(
        '--geometric-restrict-dataset',
        type=Path,
        default=None,
        help=(
            'Optional successful-samples tree; only stems with analysis_csvs here are kept '
            '(same as geometric_correctness_correlation --restrict-to-dataset).'
        ),
    )
    parser.add_argument(
        '--geometric-success-only',
        action='store_true',
        help=(
            'Restrict geometric merge to lDDT-Cα≥0.7, lDDT-PLI≥0.8, ligand RMSD<2 Å '
            '(recommended for the combined Success@K + ligand RMSD figure).'
        ),
    )
    parser.add_argument(
        '--diversity-csv-dir',
        type=Path,
        default=None,
        help=(
            'Directory of merged *analysis.csv files (csvs_for_diversity_analysis). '
            'Writes {output}_raw_diversity_boxplot_3x4.*, '
            '{output}_raw_diversity_boxplot_3x4_with_reference.*, '
            '{output}_raw_diversity_mean_std_bars_3x4.*, '
            '{output}_raw_diversity_mean_std_bars_3x4_with_reference.*, '
            '{output}_raw_diversity_bootstrap_bars_3x4.*, '
            '{output}_raw_diversity_bootstrap_bars_3x4_with_reference.*, and '
            '{output}_internal_std_boxplot_3x4.*'
        ),
    )

    args = parser.parse_args()
    methods = parse_methods_arg(
        args.methods,
        default=PRIMARY_METHODS,
    )

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    print()

    if args.accuracy and Path(args.accuracy).exists():
        print(f"Reading accuracy data from {args.accuracy}...")
        with open(args.accuracy, 'r') as f:
            topk_data = json.load(f)

        plot_topk_accuracy(topk_data, args.output, methods=methods)
        print()
    elif args.accuracy:
        print(f"Warning: Accuracy file not found: {args.accuracy}")
        print()

    if args.accuracy_pkl and Path(args.accuracy_pkl).exists():
        print(f"Reading per-file accuracy data from {args.accuracy_pkl}...")
        gdf = None
        if args.geometric_base_dir:
            print(
                f"Loading geometric merge from {args.geometric_base_dir} "
                f"(restrict={args.geometric_restrict_dataset}, success_only="
                f"{args.geometric_success_only})..."
            )
            gdf = prepare_geometric_merged_for_plots(
                args.geometric_base_dir,
                args.geometric_restrict_dataset,
                args.geometric_success_only,
                methods,
            )
        if gdf is not None:
            plot_success_vs_k_ligand_rmsd_stacked(
                args.accuracy_pkl,
                gdf,
                args.output,
                max_k=20,
                methods=methods,
            )
        else:
            plot_success_vs_k_curve_only(
                args.accuracy_pkl,
                args.output,
                max_k=20,
                methods=methods,
            )
        plot_first_correct_rank_cdf(
            args.accuracy_pkl,
            args.output,
            max_k=20,
            methods=methods,
        )
        plot_num_correct_distribution(
            args.accuracy_pkl, args.output, max_k=20, methods=methods
        )
        if gdf is not None:
            plot_success_vs_k_with_ligand_rmsd(
                args.accuracy_pkl,
                gdf,
                args.output,
                max_k=20,
                methods=methods,
            )
        print()
    elif args.accuracy_pkl:
        print(f"Warning: Accuracy pickle not found: {args.accuracy_pkl}")
        print()

    if args.diversity and Path(args.diversity).exists():
        print(f"Reading diversity data from {args.diversity}...")
        with open(args.diversity, 'r') as f:
            diversity_data = json.load(f)

        def _diversity_has_distance_stats(data: dict, mlist: tuple) -> bool:
            for m in mlist:
                block = data.get(m) or {}
                for key, stats in block.items():
                    if key.startswith("distance_cv_") and isinstance(stats, dict) and "mean" in stats:
                        return True
            return False

        if _diversity_has_distance_stats(diversity_data, methods):
            plot_average_diversity(diversity_data, args.output, methods=methods)
            plot_distance_diversity_detailed(diversity_data, args.output, methods=methods)
            plot_angular_diversity_detailed(diversity_data, args.output, methods=methods)
        else:
            print(
                "Skipping diversity plots: summary has no per-method distance_cv stats "
                "(often an empty diversity subset)."
            )
        print()
    elif args.diversity:
        print(f"Warning: Diversity file not found: {args.diversity}")
        print()

    if args.diversity_csv_dir is not None:
        div_csv = Path(args.diversity_csv_dir)
        if div_csv.is_dir():
            print(f"Raw diversity box plots from {div_csv}...")
            plot_raw_diversity_boxplots_from_csv_dir(div_csv, args.output, methods=methods)
            plot_raw_diversity_mean_std_bars_from_csv_dir(div_csv, args.output, methods=methods)
            plot_raw_diversity_bootstrap_bars_from_csv_dir(div_csv, args.output, methods=methods)
            plot_internal_std_boxplots_from_csv_dir(div_csv, args.output, methods=methods)
            print()
        else:
            print(f"Warning: --diversity-csv-dir not found: {div_csv}")
            print()

    print("All plots generated successfully!")


if __name__ == '__main__':
    main()

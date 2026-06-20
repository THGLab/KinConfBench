#!/usr/bin/env python3
"""
Plot distribution of geometrically correct samples per system on the common-systems subset,
and print a table of total/avg samples per method used for downstream correctness.

Usage (from repo root):
    python KinConfBench/analysis/kincore_label_analysis/plot_common_samples_distribution.py \\
        run_output/kincore_label_successful/comprehensive_results_data.pkl \\
        -o run_output/figures_successful/common_samples_distribution
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from KinConfBench.analysis.plot_style import (
    PRIMARY_METHODS,
    parse_methods_arg,
    method_colors,
    labels_for_methods,
)


def load_common_and_sample_counts(pkl_path, methods, max_k=20):
    """Load pkl, intersect file sets for ``methods``, per-method sample counts and per-rank counts."""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    per = obj["per_file_stats_with_conf"]
    methods = tuple(m for m in methods if m in per)
    if not methods:
        raise RuntimeError("No matching methods in pickle.")
    file_sets = [set(per[m].keys()) for m in methods]
    common_files = sorted(set.intersection(*file_sets))
    n_common = len(common_files)

    samples_per_system = {m: [] for m in methods}
    count_at_rank = {m: np.zeros(max_k, dtype=int) for m in methods}
    for method in methods:
        for fn in common_files:
            conf_dict = per[method][fn]
            n_samples = len(conf_dict)
            samples_per_system[method].append(n_samples)
            for r in range(max_k):
                if r in conf_dict:
                    count_at_rank[method][r] += 1
    return common_files, samples_per_system, count_at_rank, n_common, methods


def print_table(samples_per_system, n_common, methods, label_by_method):
    """Print table of total/avg/min/max samples per method."""
    n_m = len(methods)
    print(f"\nCommon systems (all {n_m} methods): {n_common}")
    print(
        f"\n=== Table: Samples used for downstream correctness ({n_common} common systems) ===\n"
    )
    print(f"{'Method':<12} {'Total samples':>14} {'Avg per system':>16} {'Min':>6} {'Max':>6}")
    print("-" * 56)
    for method in methods:
        counts = samples_per_system[method]
        total = sum(counts)
        avg = total / n_common if n_common else 0
        print(
            f"{label_by_method[method]:<12} {total:>14,} {avg:>16.2f} "
            f"{min(counts):>6} {max(counts):>6}"
        )
    print(
        f"\n(Total samples = geometrically correct predictions per method across {n_common} systems.)"
    )


def _subplot_grid(n: int):
    if n <= 3:
        nrows, ncols = 1, n
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4), sharey=True)
        axes = np.atleast_1d(axes).flatten()
        for idx in range(n, len(axes)):
            fig.delaxes(axes[idx])
        return fig, axes[:n]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    return fig, np.atleast_1d(axes).flatten()[:n]


def plot_distribution(samples_per_system, output_prefix, n_common, methods, colors, labels, max_k=20):
    """Histogram: systems with 1..max_k geometrically correct samples."""
    plt.rcParams["font.size"] = 11
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"

    x_bins = np.arange(1, max_k + 1)
    fig, axes = _subplot_grid(len(methods))
    for idx, (method, label) in enumerate(zip(methods, labels)):
        ax = axes[idx]
        counts = np.array(samples_per_system[method])
        hist, _ = np.histogram(counts, bins=np.arange(0.5, max_k + 1.5))
        ax.bar(x_bins, hist, color=colors[method], alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.set_xlabel("Number of geometrically correct samples per system")
        ax.set_ylabel("Number of systems")
        ax.set_title(f"{label}\n(mean = {np.mean(counts):.1f})")
        ax.set_xticks(x_bins)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.suptitle(
        f"Distribution of geometrically correct samples per system ({n_common} common systems)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = Path(output_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{output_prefix}.{ext}")
    print(f"Saved: {output_prefix}.png/pdf/svg")
    plt.close()


def plot_by_rank(count_at_rank, output_prefix, n_common, methods, colors, labels, max_k=20):
    """Count of systems with a geometrically correct prediction at each rank."""
    plt.rcParams["font.size"] = 11
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"

    ranks = np.arange(max_k)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, label in zip(methods, labels):
        ax.plot(
            ranks,
            count_at_rank[method],
            "o-",
            label=label,
            color=colors[method],
            linewidth=2,
            markersize=5,
        )
    ax.set_xlabel("Confidence rank (0 = highest confidence)")
    ax.set_ylabel("Number of systems with geometrically correct prediction at this rank")
    ax.set_title(f"Geometrically correct predictions by rank ({n_common} common systems)")
    ax.set_xticks(ranks)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_ylim(0, None)
    plt.tight_layout()
    out = Path(output_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{output_prefix}.{ext}")
    print(f"Saved: {output_prefix}.png/pdf/svg")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Table + plot of samples per method on common-systems subset"
    )
    parser.add_argument("pkl", type=Path, help="Path to comprehensive_results_data.pkl")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output path prefix for figure",
    )
    parser.add_argument("--max_k", type=int, default=20, help="Max rank (default 20)")
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated methods (default: boltz,chai,protenix)",
    )
    args = parser.parse_args()
    if not args.pkl.exists():
        raise SystemExit(f"Pickle not found: {args.pkl}")

    methods = parse_methods_arg(args.methods, default=PRIMARY_METHODS)
    common_files, samples_per_system, count_at_rank, n_common, methods = (
        load_common_and_sample_counts(args.pkl, methods, max_k=args.max_k)
    )
    colors = method_colors(methods)
    labels = labels_for_methods(methods)
    label_by_method = dict(zip(methods, labels))

    print_table(samples_per_system, n_common, methods, label_by_method)
    out = Path(args.output)
    plot_distribution(
        samples_per_system,
        str(args.output),
        n_common,
        methods,
        colors,
        labels,
        max_k=args.max_k,
    )
    by_rank_prefix = str(out.parent / (out.stem + "_by_rank"))
    plot_by_rank(
        count_at_rank,
        by_rank_prefix,
        n_common,
        methods,
        colors,
        labels,
        max_k=args.max_k,
    )


if __name__ == "__main__":
    main()

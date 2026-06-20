#!/usr/bin/env python3
"""
Table and plots for KinCoRe label-correct (functional accuracy) on the common-systems subset.

Usage (from repo root):
    python KinConfBench/analysis/kincore_label_analysis/plot_common_label_correct_summary.py \\
        run_output/kincore_label_successful/comprehensive_results_data.pkl \\
        -o run_output/figures_successful/common_label_correct
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


def load_label_correct_counts(pkl_path, methods, max_k=20):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    per = obj["per_file_stats_with_conf"]
    methods = tuple(m for m in methods if m in per)
    if not methods:
        raise RuntimeError("No matching methods in pickle.")
    file_sets = [set(per[m].keys()) for m in methods]
    common_files = sorted(set.intersection(*file_sets))
    n_common = len(common_files)

    num_correct_per_system = {m: [] for m in methods}
    count_correct_at_rank = {m: np.zeros(max_k, dtype=int) for m in methods}
    for method in methods:
        for fn in common_files:
            conf_dict = per[method][fn]
            n_correct = sum(conf_dict.get(r, 0) for r in range(max_k))
            num_correct_per_system[method].append(n_correct)
            for r in range(max_k):
                if conf_dict.get(r, 0) == 1:
                    count_correct_at_rank[method][r] += 1
    return common_files, num_correct_per_system, count_correct_at_rank, n_common, methods


def print_table(num_correct_per_system, n_common, methods, label_by_method):
    n_m = len(methods)
    print(f"\nCommon systems (all {n_m} methods): {n_common}")
    print(
        f"\n=== Table: KinCoRe label-correct (functional accuracy) "
        f"({n_common} common systems) ===\n"
    )
    print(
        f"{'Method':<12} {'Total correct':>14} {'Avg/system':>12} {'Min':>6} "
        f"{'Max':>6}  {'Systems ≥1 correct':>22}"
    )
    print("-" * 76)
    for method in methods:
        counts = np.array(num_correct_per_system[method])
        total = int(counts.sum())
        avg = counts.mean()
        n_with_at_least_one = int((counts >= 1).sum())
        pct = 100.0 * n_with_at_least_one / n_common if n_common else 0
        print(
            f"{label_by_method[method]:<12} {total:>14,} {avg:>12.2f} {counts.min():>6} "
            f"{counts.max():>6}  "
            f"{n_with_at_least_one:>6} / {n_common} ({pct:.1f}%)"
        )
    print(
        f"\n(Total correct = all-labels-correct predictions in top-20 across {n_common} systems.)"
    )


def _subplot_grid(n: int):
    if n <= 3:
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
        axes = np.atleast_1d(axes).flatten()
        return fig, axes[:n]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    return fig, np.atleast_1d(axes).flatten()[:n]


def plot_distribution(num_correct_per_system, output_prefix, n_common, methods, colors, labels, max_k=20):
    plt.rcParams["font.size"] = 11
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"

    x_bins = np.arange(0, max_k + 1)
    fig, axes = _subplot_grid(len(methods))
    for idx, (method, label) in enumerate(zip(methods, labels)):
        ax = axes[idx]
        counts = np.array(num_correct_per_system[method])
        hist, _ = np.histogram(counts, bins=np.arange(-0.5, max_k + 1.5))
        ax.bar(x_bins, hist, color=colors[method], alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.set_xlabel("Number correct (out of top-20)")
        ax.set_ylabel("Number of systems")
        mean_correct = np.mean(counts)
        n_any = (counts >= 1).sum()
        ax.set_title(f"{label}\n(mean = {mean_correct:.2f}, ≥1 correct: {n_any}/{n_common})")
        ax.set_xticks(x_bins)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.suptitle(
        f"Distribution of KinCoRe label-correct predictions per system ({n_common} common systems)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{output_prefix}.{ext}")
    print(f"Saved: {output_prefix}.png/pdf/svg")
    plt.close()


def plot_by_rank(count_correct_at_rank, output_prefix, n_common, methods, colors, labels, max_k=20):
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
            count_correct_at_rank[method],
            "o-",
            label=label,
            color=colors[method],
            linewidth=2,
            markersize=5,
        )
    ax.set_xlabel("Confidence rank (0 = highest confidence)")
    ax.set_ylabel("Number of systems with correct prediction at this rank")
    ax.set_title(f"KinCoRe label-correct (functional) by rank ({n_common} common systems)")
    ax.set_xticks(ranks)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_ylim(0, None)
    plt.tight_layout()
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{output_prefix}.{ext}")
    print(f"Saved: {output_prefix}.png/pdf/svg")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Table + plots for KinCoRe label-correct on common-systems subset"
    )
    parser.add_argument("pkl", type=Path, help="Path to comprehensive_results_data.pkl")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output path prefix",
    )
    parser.add_argument("--max_k", type=int, default=20, help="Top-K (default 20)")
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
    common_files, num_correct_per_system, count_correct_at_rank, n_common, methods = (
        load_label_correct_counts(args.pkl, methods, max_k=args.max_k)
    )
    colors = method_colors(methods)
    labels = labels_for_methods(methods)
    label_by_method = dict(zip(methods, labels))

    print_table(num_correct_per_system, n_common, methods, label_by_method)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_distribution(
        num_correct_per_system,
        str(args.output),
        n_common,
        methods,
        colors,
        labels,
        max_k=args.max_k,
    )
    plot_by_rank(
        count_correct_at_rank,
        str(out.parent / (out.stem + "_by_rank")),
        n_common,
        methods,
        colors,
        labels,
        max_k=args.max_k,
    )


if __name__ == "__main__":
    main()

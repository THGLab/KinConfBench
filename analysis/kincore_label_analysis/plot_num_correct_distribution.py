#!/usr/bin/env python3
"""
Plot distribution of number of correct predictions (out of top-K) per system, per method.

Standalone script: takes a pickle from comprehensive_method_analysis (per_file_stats_with_conf)
and produces bar plots of how many systems have 0, 1, ..., K correct predictions (per method).

Can optionally restrict to a subset of systems (e.g. for diversity analysis where all methods
have >=10 correct predictions).

Usage:
    python plot_num_correct_distribution.py comprehensive_results_data.pkl -o figures/results

    # With diversity subset (histogram x-axis defaults to 10..max_k):
    python plot_num_correct_distribution.py comprehensive_results_data.pkl \\
        -o figures/results_diversity --subset-csv-dir csvs_for_diversity_analysis/
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

plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

# Panel titles larger than shared axis labels
_TITLE_FONTSIZE = 18
_SHARED_AXIS_LABEL_FONTSIZE = _TITLE_FONTSIZE - 2


def _shared_xlabel_text(max_k: int, subset_files, hist_x_min: int) -> str:
    """One x-label for the whole figure (top-K / subset noted in parentheses)."""
    base = "Number of correct predictions for each system"
    if subset_files is not None and hist_x_min > 0:
        return f"{base} (top-{max_k}; ≥{hist_x_min} in subset)"
    return f"{base} (out of top-{max_k})"


def _axes_for_methods(n: int) -> tuple[plt.Figure, list]:
    """
    One row per figure for 1–3 methods (wide figure for three models so x-axis labels fit).
    Four methods: 2×2 with sharey.
    """
    if n <= 0:
        raise ValueError("n_methods must be positive")
    if n == 1:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
        return fig, [ax]
    if n == 2:
        fig, axes = plt.subplots(
            1, 2, figsize=(11, 4.8), sharey=True, constrained_layout=True
        )
        return fig, list(np.atleast_1d(axes).flatten())
    if n == 3:
        # Wide single row: shared y-scale, room for full x tick labels on each panel
        fig, axes = plt.subplots(
            1, 3, figsize=(17, 5.2), sharey=True, constrained_layout=True
        )
        return fig, list(np.atleast_1d(axes).flatten())
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 10), sharey=True, constrained_layout=True
    )
    axes = list(np.atleast_1d(axes).flatten())
    return fig, axes[:n]


def plot_num_correct_distribution(
    pkl_path,
    output_prefix,
    max_k=20,
    subset_files=None,
    methods=PRIMARY_METHODS,
    *,
    hist_x_min: int | None = None,
):
    """
    Plot distribution of number of correct predictions (out of top-max_k) per system, per method.

    If subset_files is provided (set or list of file names), only systems in that subset
    are included (and must be present in all methods).

    hist_x_min: left edge of histogram bins (inclusive). If None and subset_files is set,
    defaults to 10 (diversity subset filter). Otherwise defaults to 0.
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
    if subset_files is not None:
        subset_set = set(subset_files)
        common_files = sorted(set(common_files) & subset_set)
    if not common_files:
        raise RuntimeError("No common files across all methods.")

    if hist_x_min is None:
        hist_x_min = 10 if subset_files is not None else 0
    hist_x_min = int(np.clip(hist_x_min, 0, max_k))

    counts_per_method = {}
    for method in methods:
        counts = []
        for fn in common_files:
            conf_dict = per[method][fn]
            count = sum(conf_dict.get(r, 0) for r in range(max_k))
            counts.append(count)
        counts_per_method[method] = np.array(counts)

    x_bins = np.arange(hist_x_min, max_k + 1)
    bin_edges = np.arange(hist_x_min - 0.5, max_k + 1.5)

    n = len(methods)
    fig, axes = _axes_for_methods(n)
    for idx, (method, label) in enumerate(zip(methods, labels)):
        ax = axes[idx]
        hist_full, _ = np.histogram(counts_per_method[method], bins=bin_edges)
        ax.bar(
            x_bins,
            hist_full,
            color=colors[method],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        if n == 1:
            ax.set_xlabel(
                _shared_xlabel_text(max_k, subset_files, hist_x_min),
                fontsize=_SHARED_AXIS_LABEL_FONTSIZE,
            )
        else:
            ax.set_xlabel("")
        if idx == 0 or n == 1:
            ax.set_ylabel(
                "Number of systems",
                fontsize=_SHARED_AXIS_LABEL_FONTSIZE,
            )
        else:
            ax.set_ylabel("")
        ax.set_title("%s" % label, fontsize=_TITLE_FONTSIZE)
        ax.tick_params(axis="both", which="major", labelsize=14)
        if len(x_bins) > 15:
            ax.set_xticks(x_bins[::2])
        else:
            ax.set_xticks(x_bins)
        ax.set_xlim(hist_x_min - 0.5, max_k + 0.5)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    if n > 1:
        fig.supxlabel(
            _shared_xlabel_text(max_k, subset_files, hist_x_min),
            fontsize=_SHARED_AXIS_LABEL_FONTSIZE,
        )

    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{output_prefix}_num_correct_distribution.{ext}")
    print(f"Saved: {output_prefix}_num_correct_distribution.png/pdf/svg")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot distribution of number of correct predictions per system"
    )
    parser.add_argument(
        "pkl",
        type=Path,
        help="Path to comprehensive_results_data.pkl (or *_data.pkl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output prefix for PNG/PDF (e.g. figures/results)",
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=20,
        help="Top-K to count (default 20)",
    )
    parser.add_argument(
        "--subset-csv-dir",
        type=Path,
        default=None,
        help="If set, restrict to systems whose *_analysis.csv files are in this directory",
    )
    parser.add_argument(
        "--hist-x-min",
        type=int,
        default=None,
        help=(
            "Left edge of histogram (inclusive). Default: 0 for full set, 10 when "
            "--subset-csv-dir is set."
        ),
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated methods (default: boltz,chai,protenix)",
    )
    args = parser.parse_args()
    if not args.pkl.exists():
        raise SystemExit("Pickle not found: %s" % args.pkl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset_files = None
    if args.subset_csv_dir is not None:
        if not args.subset_csv_dir.is_dir():
            raise SystemExit("Subset CSV dir not found: %s" % args.subset_csv_dir)
        subset_files = [
            p.stem.replace("_analysis", "")
            for p in args.subset_csv_dir.iterdir()
            if p.suffix == ".csv" and p.name.endswith("_analysis.csv")
        ]
        print("Subset: %d systems from %s" % (len(subset_files), args.subset_csv_dir))
    methods = parse_methods_arg(args.methods, default=PRIMARY_METHODS)
    plot_num_correct_distribution(
        str(args.pkl),
        str(args.output),
        max_k=args.max_k,
        subset_files=subset_files,
        methods=methods,
        hist_x_min=args.hist_x_min,
    )


if __name__ == "__main__":
    main()

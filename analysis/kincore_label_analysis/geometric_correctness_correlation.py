#!/usr/bin/env python3
"""
Correlate KinCoRe correctness with geometric metrics (lddt_ca, lddt_pli, ligand_rmsd, protein_rmsd).

Joins analysis_csvs (KinCoRe labels) with geometric_analysis_csvs to test whether correct
KinCoRe predictions correspond to better structural agreement.

Usage:
    python geometric_correctness_correlation.py --base-dir kinase_all_data/kinconfbench \\
        -o analysis/run_output/figures/kincore_geometric_correlation
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

# Prefix order for parsing structure_id (longest / most specific prefixes first).
METHOD_PREFIX_PARSE_ORDER: tuple[str, ...] = ("protenix", "boltz", "chai", "rfaa")

from KinConfBench.analysis.plot_style import (
    PRIMARY_METHODS,
    parse_methods_arg,
    method_colors,
    labels_for_methods,
    METHOD_DISPLAY_NAMES,
)
ALL_LABEL_KEYS = [
    "activity_label",
    "spatial_label",
    "dihedral_label",
    "chelix_label",
    "saltbr_label",
    "actloop_nt_label",
    "actloop_ct_label",
    "spine_label",
]
GEOMETRIC_METRICS = ["lddt_ca", "lddt_pli", "ligand_rmsd", "protein_rmsd"]

# Vertical stacked figure: rows top-to-bottom (matches common RMSD → lDDT ordering).
STACKED_FOUR_PANEL_METRICS: tuple[str, ...] = (
    "protein_rmsd",
    "ligand_rmsd",
    "lddt_ca",
    "lddt_pli",
)

STACKED_THREE_PANEL_METRICS: tuple[str, ...] = (
    "ligand_rmsd",
    "lddt_ca",
    "lddt_pli",
)

# Fixed y ticks for stacked multi-panel rows; None → MaxNLocator (lDDT-PLI).
STACKED_PANEL_Y_TICKS: dict[str, list[float] | None] = {
    "protein_rmsd": [0.0, 20.0, 40.0],
    "ligand_rmsd": [0.0, 1.0, 2.0],
    "lddt_ca": [0.7, 0.8, 0.9, 1.0],
    "lddt_pli": None,
}


def metric_row_label(metric: str) -> str:
    """Short y-axis label for stacked multi-metric figures."""
    labels = {
        "protein_rmsd": "Protein Cα RMSD (Å)",
        "ligand_rmsd": "Ligand RMSD (Å)",
        "lddt_ca": "lDDT-Cα",
        "lddt_pli": "lDDT-PLI",
    }
    return labels.get(metric, metric.replace("_", " ").title())


def matches_any_ground_truth(pred_row: dict, ground_truth_rows: list[dict], label_keys: list[str]) -> bool:
    """Check if prediction matches any ground truth row for specified labels."""
    for gt_row in ground_truth_rows:
        if all(pred_row.get(key) == gt_row.get(key) for key in label_keys):
            return True
    return False


def extract_method(structure_id: str) -> str | None:
    """Extract method from structure_id (e.g., 'boltz_conf0' -> 'boltz')."""
    sid = str(structure_id)
    for m in METHOD_PREFIX_PARSE_ORDER:
        if sid.startswith(m + "_conf"):
            return m
    return None


def stems_in_filtered_dataset(dataset_dir: Path) -> set[str]:
    """
    Collect stem names (e.g. ABL1_HUMAN_406) from a kinconfbench-style tree
    that contains gene/analysis_csvs/*_analysis.csv (e.g. successful_samples output).
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"--restrict-to-dataset not found: {dataset_dir}")
    stems: set[str] = set()
    for p in dataset_dir.rglob("analysis_csvs/*_analysis.csv"):
        stems.add(p.stem.replace("_analysis", ""))
    return stems


def load_merged_data(base_dir: Path) -> pd.DataFrame:
    """
    Load analysis and geometric CSVs for all stems, merge on structure_id.
    Returns DataFrame with: stem, structure_id, method, is_correct, lddt_ca, lddt_pli, ligand_rmsd, protein_rmsd.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    records = []
    gene_dirs = sorted(d for d in base_dir.iterdir() if d.is_dir())
    for gene_dir in tqdm(gene_dirs, desc="Processing stems"):
        analysis_dir = gene_dir / "analysis_csvs"
        geom_dir = gene_dir / "geometric_analysis_csvs"
        if not analysis_dir.exists() or not geom_dir.exists():
            continue

        for analysis_csv in analysis_dir.glob("*_analysis.csv"):
            stem = analysis_csv.stem.replace("_analysis", "")
            geom_csv = geom_dir / f"{stem}_geometric.csv"
            if not geom_csv.exists():
                continue

            # Load analysis CSV
            with open(analysis_csv) as f:
                rows = list(csv.DictReader(f))

            ground_truth = [
                r for r in rows
                if len(r.get("structure_id", "")) == 4
                and r["structure_id"].replace("_", "").replace("-", "").isalnum()
            ]
            if not ground_truth:
                continue

            # Compute correctness for cofolding rows
            correctness = {}
            for r in rows:
                sid = r.get("structure_id", "").strip()
                method = extract_method(sid)
                if method is None:
                    continue
                is_correct = matches_any_ground_truth(r, ground_truth, ALL_LABEL_KEYS)
                correctness[sid] = (method, is_correct)

            if not correctness:
                continue

            # Load geometric CSV
            geom_df = pd.read_csv(geom_csv)
            if geom_df.empty:
                continue

            for _, row in geom_df.iterrows():
                sid = str(row["structure_id"]).strip()
                if sid not in correctness:
                    continue
                method, is_correct = correctness[sid]
                rec = {
                    "stem": stem,
                    "structure_id": sid,
                    "method": method,
                    "is_correct": is_correct,
                }
                for m in GEOMETRIC_METRICS:
                    if m in row:
                        val = row[m]
                        rec[m] = None if (pd.isna(val) or val == "") else float(val)
                records.append(rec)

    return pd.DataFrame(records)


def _subplot_axes(n_methods: int):
    if n_methods <= 3:
        fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4), sharey=True)
        axes = np.atleast_1d(axes).flatten().tolist()
        return fig, axes
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    return fig, np.atleast_1d(axes).flatten().tolist()


def plot_metric(
    merged: pd.DataFrame,
    metric: str,
    output_prefix: str,
    higher_is_better: bool,
    methods: tuple[str, ...],
    colors: dict,
    labels: list[str],
) -> None:
    """
    Plot per-method distribution of metric for correct vs incorrect.
    higher_is_better: True for lddt_ca, lddt_pli; False for rmsd metrics.
    """
    df = merged.dropna(subset=[metric])
    if df.empty:
        print(f"Skipping {metric}: no valid data")
        return

    fig, axes = _subplot_axes(len(methods))

    for idx, (method, label) in enumerate(zip(methods, labels)):
        ax = axes[idx]
        mdf = df[df["method"] == method]
        if mdf.empty:
            ax.set_visible(False)
            continue

        correct_vals = mdf[mdf["is_correct"]][metric].values
        incorrect_vals = mdf[~mdf["is_correct"]][metric].values

        if len(correct_vals) == 0 or len(incorrect_vals) == 0:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        bp = ax.boxplot(
            [correct_vals, incorrect_vals],
            positions=[0, 1],
            tick_labels=["Correct", "Incorrect"],
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

        ax.set_xticklabels(["Correct", "Incorrect"])
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle(
        f"KinCoRe correctness vs {metric.replace('_', ' ').title()} (per method)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        out = f"{output_prefix}_{metric}.{ext}"
        plt.savefig(out)
    print(f"Saved: {output_prefix}_{metric}.png/pdf/svg")
    plt.close()


def _plot_metrics_stacked_panel(
    merged: pd.DataFrame,
    output_prefix: str,
    methods: tuple[str, ...],
    colors: dict,
    labels: list[str],
    *,
    row_metrics: tuple[str, ...],
    output_stem: str,
) -> None:
    """
    Metrics as rows, methods as columns; shared x per column group, sharey row.
    """
    n_rows = len(row_metrics)
    n_cols = len(methods)
    y_label_fs = 18
    y_tick_fs = 15
    x_tick_fs = 15
    title_fs = 18
    fig_w = 3.6 * n_cols
    fig_h = 3.0 * n_rows
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey="row",
    )
    if n_rows == 1:
        axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for mi, metric in enumerate(row_metrics):
        df = merged.dropna(subset=[metric])
        for j, (method, mlabel) in enumerate(zip(methods, labels)):
            ax = axes[mi, j]
            mdf = df[df["method"] == method]
            if mdf.empty:
                ax.set_visible(False)
                continue

            correct_vals = mdf[mdf["is_correct"]][metric].values
            incorrect_vals = mdf[~mdf["is_correct"]][metric].values

            if len(correct_vals) == 0 or len(incorrect_vals) == 0:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            else:
                bp = ax.boxplot(
                    [correct_vals, incorrect_vals],
                    positions=[0, 1],
                    tick_labels=["Correct", "Incorrect"],
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

            ax.set_xlim(-0.5, 1.5)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            if mi == 0:
                ax.set_title(mlabel, fontweight="bold", fontsize=title_fs)
            if mi < n_rows - 1:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.tick_params(axis="x", labelsize=x_tick_fs)
            if j > 0:
                ax.tick_params(axis="y", labelleft=False)

        axes[mi, 0].set_ylabel(metric_row_label(metric), fontsize=y_label_fs)
        axes[mi, 0].tick_params(axis="y", labelsize=y_tick_fs)
        fixed_ticks = STACKED_PANEL_Y_TICKS.get(metric)
        for j in range(n_cols):
            ax = axes[mi, j]
            if not ax.get_visible():
                continue
            if fixed_ticks is not None:
                ax.set_yticks(fixed_ticks)
                ax.yaxis.set_major_locator(mticker.FixedLocator(fixed_ticks))
                if metric == "lddt_ca":
                    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
                else:
                    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            else:
                ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=2, min_n_ticks=2))

    fig.align_ylabels(axes[:, 0])
    plt.tight_layout(rect=[0.02, 0.03, 1, 0.99])
    base = f"{output_prefix}_{output_stem}"
    for ext in ("png", "pdf", "svg"):
        out = f"{base}.{ext}"
        plt.savefig(out)
    print(f"Saved: {base}.png/pdf/svg")
    plt.close()


def plot_metrics_stacked_four_panel(
    merged: pd.DataFrame,
    output_prefix: str,
    methods: tuple[str, ...],
    colors: dict,
    labels: list[str],
) -> None:
    """
    Four rows: protein RMSD, ligand RMSD, lDDT-Cα, lDDT-PLI (see module constants).
    """
    _plot_metrics_stacked_panel(
        merged,
        output_prefix,
        methods,
        colors,
        labels,
        row_metrics=STACKED_FOUR_PANEL_METRICS,
        output_stem="four_panel",
    )


def plot_metrics_stacked_three_panel(
    merged: pd.DataFrame,
    output_prefix: str,
    methods: tuple[str, ...],
    colors: dict,
    labels: list[str],
) -> None:
    """
    Three rows: ligand RMSD, lDDT-Cα, lDDT-PLI (no protein RMSD).
    """
    _plot_metrics_stacked_panel(
        merged,
        output_prefix,
        methods,
        colors,
        labels,
        row_metrics=STACKED_THREE_PANEL_METRICS,
        output_stem="three_panel",
    )


def plot_overall_histograms(
    merged: pd.DataFrame,
    output_prefix: str,
    methods: tuple[str, ...],
    colors: dict,
    labels: list[str],
) -> None:
    """
    Per-method histograms for ligand_rmsd, lddt_ca, lddt_pli in a grid:
    one row per metric, one column per method (method fill color from plot_style).
    """
    row_metrics = STACKED_THREE_PANEL_METRICS
    n_rows = len(row_metrics)
    n_cols = len(methods)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        sharex="row",
        sharey="row",
        squeeze=False,
    )

    for ri, metric in enumerate(row_metrics):
        mdf = merged[merged["method"].isin(methods)].dropna(subset=[metric])
        if mdf.empty:
            edges = None
        else:
            edges = np.histogram_bin_edges(mdf[metric].to_numpy(), bins=50)

        for ci, (method, col_label) in enumerate(zip(methods, labels)):
            ax = axes[ri, ci]
            vals = merged.loc[merged["method"] == method, metric].dropna()
            if vals.empty or edges is None:
                ax.set_visible(False)
                continue

            ax.hist(
                vals,
                bins=edges,
                color=colors[method],
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )
            # Label every column in each row so Ligand RMSD / lDDT-Cα / lDDT-PLI all appear
            # (not only the bottom row).
            ax.set_xlabel(metric_row_label(metric))
            if ci == 0:
                ax.set_ylabel("Count")
            if ri == 0:
                ax.set_title(col_label)
            ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        out = f"{output_prefix}_overall_histograms.{ext}"
        plt.savefig(out)
    print(f"Saved: {output_prefix}_overall_histograms.png/pdf/svg")
    plt.close()


def write_overall_histogram_sample_stats(
    merged: pd.DataFrame,
    methods: tuple[str, ...],
    output_prefix: str,
) -> None:
    """
    Per-system (stem) prediction counts and per-method totals for figure captions.
    Writes a sidecar .txt next to the overall histograms outputs.
    """
    path = Path(f"{output_prefix}_overall_histograms_sample_stats.txt")
    n_systems = merged["stem"].nunique()
    lines = [
        "Sample counts for overall histograms (same merged table as the plots).",
        "",
        f"Systems (unique stems): {n_systems}",
        "",
    ]
    for method in methods:
        mdf = merged[merged["method"] == method]
        total = len(mdf)
        name = METHOD_DISPLAY_NAMES.get(method, method)
        lines.append(f"{name} ({method})")
        lines.append(f"  Total samples (all systems): {total:,}")
        if total == 0:
            lines.append("  Samples per system: (no rows)")
        else:
            per_stem = mdf.groupby("stem", sort=False).size()
            lines.append(
                f"  Samples per system: min={int(per_stem.min())}, "
                f"median={per_stem.median():.1f}, mean={per_stem.mean():.2f}, max={int(per_stem.max())}"
            )
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    print("\n" + text)
    path.write_text(text, encoding="utf-8")
    print(f"Saved: {path}")


def plot_method_histogram(
    merged: pd.DataFrame,
    metric: str,
    output_prefix: str,
    metric_label: str,
    methods: tuple[str, ...],
    colors: dict,
    labels: list[str],
) -> None:
    """
    Plot histogram of metric with 2x2 breakdown by method, using method colors.
    """
    df = merged.dropna(subset=[metric])
    if df.empty:
        print(f"Skipping {metric} method histogram: no valid data")
        return

    fig, axes = _subplot_axes(len(methods))

    for idx, (method, label) in enumerate(zip(methods, labels)):
        ax = axes[idx]
        vals = df[df["method"] == method][metric]
        if vals.empty:
            ax.set_visible(False)
            continue

        ax.hist(
            vals,
            bins=50,
            color=colors[method],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xlabel(metric_label)
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle(
        f"{metric_label} distribution by method",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        out = f"{output_prefix}_{metric}_by_method.{ext}"
        plt.savefig(out)
    print(f"Saved: {output_prefix}_{metric}_by_method.png/pdf/svg")
    plt.close()


def print_success_criteria_breakdown(
    merged: pd.DataFrame,
    lddt_ca_threshold: float = 0.7,
    lddt_pli_threshold: float = 0.8,
    ligand_rmsd_max: float | None = None,
    methods: tuple[str, ...] | None = None,
) -> None:
    """
    Break down KinCoRe correct/incorrect by geometric success.
    Success = lDDT-Cα >= 0.7 (global fold) and lDDT-PLI >= 0.8 (ligand pocket).
    Optionally add ligand_rmsd < ligand_rmsd_max (e.g. 2 Å).
    """
    required_cols = ["lddt_ca", "lddt_pli"]
    if ligand_rmsd_max is not None:
        required_cols.append("ligand_rmsd")
    df = merged.dropna(subset=required_cols)
    df = df.copy()
    success = (df["lddt_ca"] >= lddt_ca_threshold) & (df["lddt_pli"] >= lddt_pli_threshold)
    if ligand_rmsd_max is not None:
        success = success & (df["ligand_rmsd"] < ligand_rmsd_max)
    df["success"] = success

    criteria_str = "lDDT-Cα ≥ 0.7 AND lDDT-PLI ≥ 0.8"
    if ligand_rmsd_max is not None:
        criteria_str += f" AND ligand_rmsd < {ligand_rmsd_max}"
    print("\n" + "=" * 60)
    print(f"Geometric success criteria: {criteria_str}")
    print("=" * 60)

    # Overall (all methods)
    for label, success_val in [("Successful", True), ("Unsuccessful", False)]:
        subset = df[df["success"] == success_val]
        total = len(subset)
        correct = subset["is_correct"].sum()
        incorrect = total - correct
        print(f"\n{label} (all methods): n = {total:,}")
        if total > 0:
            print(f"  Correct KinCoRe:   {correct:6,} ({100 * correct / total:.1f}%)")
            print(f"  Incorrect KinCoRe: {incorrect:6,} ({100 * incorrect / total:.1f}%)")

    # Per method
    print("\n--- Per method ---")
    pm = methods if methods is not None else tuple(sorted(df["method"].unique()))
    for method in pm:
        mdf = df[df["method"] == method]
        print(f"\n{METHOD_DISPLAY_NAMES.get(method, method)}:")
        for label, success_val in [("Successful", True), ("Unsuccessful", False)]:
            subset = mdf[mdf["success"] == success_val]
            total = len(subset)
            correct = subset["is_correct"].sum()
            incorrect = total - correct
            if total > 0:
                print(f"  {label:12} n={total:6,}  Correct: {100 * correct / total:5.1f}%  Incorrect: {100 * incorrect / total:5.1f}%")


def print_summary_stats(merged: pd.DataFrame, methods: tuple[str, ...] | None = None) -> None:
    """Print mean/median per (method, is_correct) and Wilcoxon test for each metric."""
    for metric in GEOMETRIC_METRICS:
        df = merged.dropna(subset=[metric])
        if df.empty:
            continue
        print(f"\n--- {metric} ---")
        pm = methods if methods is not None else tuple(sorted(df["method"].unique()))
        for method in pm:
            mdf = df[df["method"] == method]
            correct = mdf[mdf["is_correct"]][metric]
            incorrect = mdf[~mdf["is_correct"]][metric]
            if len(correct) > 0:
                print(f"  {method} Correct:   n={len(correct):5d}  mean={correct.mean():.4f}  median={correct.median():.4f}")
            if len(incorrect) > 0:
                print(f"  {method} Incorrect: n={len(incorrect):5d}  mean={incorrect.mean():.4f}  median={incorrect.median():.4f}")
            if len(correct) > 5 and len(incorrect) > 5 and HAS_SCIPY:
                result = stats.mannwhitneyu(correct, incorrect, alternative="two-sided")
                pval = result.pvalue
                print(f"  {method} Mann-Whitney U p-value: {pval:.4e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlate KinCoRe correctness with geometric metrics"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("kinase_all_data/kinconfbench"),
        help="Base directory containing gene folders (default: kinase_all_data/kinconfbench). "
        "Ignored if --merged-csv is set.",
    )
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=None,
        help="Load merged rows from this CSV instead of scanning --base-dir.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("analysis/run_output/figures/kincore_geometric_correlation"),
        help="Output prefix for CSV and plots",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Restrict to successful samples (lDDT-Cα≥0.7, lDDT-PLI≥0.8, ligand_rmsd<2)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated methods to include in plots (default: boltz,chai,protenix).",
    )
    parser.add_argument(
        "--restrict-to-dataset",
        type=Path,
        default=None,
        help=(
            "Optional kinconfbench-style root (e.g. successful_samples_kinconfbench). "
            "Only stems that have an analysis CSV under this tree are kept after merging "
            "from --base-dir (filtered subset)."
        ),
    )
    args = parser.parse_args()

    output_prefix = Path(args.output)
    if not output_prefix.is_absolute():
        output_prefix = (Path.cwd() / output_prefix).resolve()

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if args.merged_csv:
        mc = Path(args.merged_csv).resolve()
        if not mc.is_file():
            raise SystemExit(f"--merged-csv not found: {mc}")
        print(f"Loading merged table from {mc}...")
        merged = pd.read_csv(mc)
    else:
        base_dir = Path(args.base_dir)
        if not base_dir.is_absolute():
            base_dir = (Path.cwd() / base_dir).resolve()
        print(f"Loading data from {base_dir}...")
        merged = load_merged_data(base_dir)
    if merged.empty:
        raise SystemExit("No merged data found. Check inputs and filters.")

    print(f"Merged {len(merged)} rows from {merged['stem'].nunique()} stems")

    if args.restrict_to_dataset:
        rdir = Path(args.restrict_to_dataset)
        if not rdir.is_absolute():
            rdir = (Path.cwd() / rdir).resolve()
        stem_allow = stems_in_filtered_dataset(rdir)
        before = len(merged)
        merged = merged[merged["stem"].isin(stem_allow)].copy()
        print(
            f"Restricted to {len(stem_allow)} stems from {rdir}: "
            f"{len(merged)} rows (dropped {before - len(merged)})."
        )
        if merged.empty:
            raise SystemExit("No rows left after --restrict-to-dataset filter.")

    if args.success_only:
        merged = merged.dropna(subset=["lddt_ca", "lddt_pli", "ligand_rmsd"])
        mask = (
            (merged["lddt_ca"] >= 0.7)
            & (merged["lddt_pli"] >= 0.8)
            & (merged["ligand_rmsd"] < 2.0)
        )
        merged = merged[mask].copy()
        print(f"Filtered to successful samples: {len(merged)} rows")

    methods = parse_methods_arg(args.methods, default=PRIMARY_METHODS)
    merged = merged[merged["method"].isin(methods)].copy()
    methods = tuple(m for m in methods if m in merged["method"].unique())
    if not methods:
        raise SystemExit("No rows left after --methods filter.")
    colors = method_colors(methods)
    labels = labels_for_methods(methods)

    merged_csv = f"{output_prefix}_merged.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"Saved: {merged_csv}")

    plot_overall_histograms(merged, str(output_prefix), methods, colors, labels)
    write_overall_histogram_sample_stats(merged, methods, str(output_prefix))
    plot_method_histogram(
        merged, "lddt_ca", str(output_prefix), "lDDT-Cα", methods, colors, labels
    )
    plot_method_histogram(
        merged, "lddt_pli", str(output_prefix), "lDDT-PLI", methods, colors, labels
    )

    higher_is_better = {"lddt_ca": True, "lddt_pli": True, "ligand_rmsd": False, "protein_rmsd": False}
    for metric in GEOMETRIC_METRICS:
        plot_metric(
            merged, metric, str(output_prefix), higher_is_better[metric],
            methods, colors, labels,
        )

    plot_metrics_stacked_four_panel(merged, str(output_prefix), methods, colors, labels)
    plot_metrics_stacked_three_panel(merged, str(output_prefix), methods, colors, labels)

    print_success_criteria_breakdown(merged, methods=methods)
    print_success_criteria_breakdown(merged, ligand_rmsd_max=2.0, methods=methods)
    print_summary_stats(merged, methods=methods)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Method-specific cutoff Holo/Apo analysis.

Uses each method's training data cutoff to define Seen/Unseen per method,
then recomputes pair holo/apo category counts and regenerates the plot.

Outputs under --output_dir:
  - all_results_combined.csv — holo_inference_apo rows with split reassigned per method
    (not the global pre/post Seen/Unseen from merging alone).
  - pair_holo_apo_category_counts_method_specific.csv — stacked-bar category counts.
  - figures/pair_holo_apo_categories_method_specific.png|.svg

Method cutoffs:
  Protenix:         2021-09-30
  Chai-1:           2021-01-12
  Boltz-2:          2023-06-01
  RoseTTAFold-AA:   2020-04-30

Usage:
    python pair_holo_apo_method_specific_analysis.py \\
        --pre_summaries /path/to/cofolding_results_pre_training \\
        --post_summaries /path/to/cofolding_results_post_training \\
        --pre_csv /path/to/method_performance_analysis_pre_training/all_results_combined.csv \\
        --post_csv /path/to/method_performance_analysis_post_training/all_results_combined.csv \\
        --output_dir /path/to/output

Or pass a single merged CSV with --all_results instead of --pre_csv/--post_csv.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, Tuple

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from KinConfBench.analysis.plot_style import PRIMARY_METHODS, parse_methods_arg

METHOD_CUTOFFS = {
    "protenix": "2021-09-30",
    "chai": "2021-01-12",
    "boltz": "2023-06-01",
    "rfaa": "2020-04-30",
}

METHOD_DISPLAY = {
    "protenix": "Protenix",
    "chai": "Chai-1",
    "boltz": "Boltz-2",
    "rfaa": "RoseTTAFold-AA",
}

# Tab10 (same hues as seaborn's default): blue / orange / … / gray at index 7 for "None"; Both = RGB blend.
_tab10 = matplotlib.colormaps["tab10"]
_SNS_HOLO = _tab10(0)
_SNS_APO = _tab10(1)
_SNS_NEITHER = _tab10(7)
_rgb_h = mcolors.to_rgb(_SNS_HOLO)
_rgb_a = mcolors.to_rgb(_SNS_APO)
_SNS_BOTH = tuple((h + a) / 2.0 for h, a in zip(_rgb_h, _rgb_a))
# Stack order bottom → top: (counts column, legend label, facecolor)
_CATEGORY_STACK = (
    ("n_holo_only", "Holo State", _SNS_HOLO),
    ("n_both", "Dual Match", _SNS_BOTH),
    ("n_apo_only", "Apo Drift", _SNS_APO),
    ("n_neither", "None", _SNS_NEITHER),
)

# Main figure typography (+2 pt vs earlier defaults); no bold on labels
FONT_TICK = 14
FONT_AXIS = 16
FONT_TITLE = 15
FONT_LEGEND = 14


def merge_pre_post_pair_csvs(
    pre_path: Path,
    post_path: Path,
    allowed_methods: frozenset[str],
) -> pd.DataFrame:
    """Merge pre/post method-performance CSVs (Seen/Unseen) for holo_inference_apo rows."""
    pre_df = pd.read_csv(pre_path)
    post_df = pd.read_csv(post_path)
    pre_df = pre_df.assign(training="pre_training", split="Seen")
    post_df = post_df.assign(training="post_training", split="Unseen")
    combined = pd.concat([pre_df, post_df], ignore_index=True)
    combined = combined[combined["classification"] == "holo_inference_apo"].copy()
    combined = combined[combined["method"].str.lower().isin(allowed_methods)].copy()
    return combined


def normalize_ligands(ligands_str: str) -> Tuple[str, ...]:
    """Normalize ligand string to tuple of ligand codes (same as create_pre_training_results)."""
    if pd.isna(ligands_str) or ligands_str == "No_ligand":
        return ("No_ligand",)
    ligand_codes = [lig.split(":")[0].strip() for lig in str(ligands_str).split(";")]
    return tuple(sorted(ligand_codes))


def inference_condition_to_ligands(inference_condition: str) -> Tuple[str, ...]:
    """Map inference_condition (e.g. '1G0', 'BYP_BWP') to normalized ligand tuple."""
    if pd.isna(inference_condition) or str(inference_condition).strip() == "":
        return ("No_ligand",)
    s = str(inference_condition).strip()
    if s.lower() == "apo":
        return ("No_ligand",)
    parts = s.split("_")
    return tuple(sorted(p.strip() for p in parts if p.strip()))


def get_new_ligands_only(post_df: pd.DataFrame, pre_df: pd.DataFrame) -> Set[Tuple[str, ...]]:
    """Ligand combos that appear only after cutoff."""
    post_ligands = set()
    for ligands_str in post_df["ligands"].dropna():
        s = str(ligands_str)
        if "peptide" in s.lower():
            continue
        post_ligands.add(normalize_ligands(s))
    pre_ligands = set()
    for ligands_str in pre_df["ligands"].dropna():
        s = str(ligands_str)
        if "peptide" in s.lower():
            continue
        pre_ligands.add(normalize_ligands(s))
    return post_ligands - pre_ligands


def load_all_summaries(pre_dir: Path, post_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load gene summaries from pre and post, merge per gene."""
    all_genes = set()
    if pre_dir.exists():
        all_genes |= {d.name for d in pre_dir.iterdir() if d.is_dir()}
    if post_dir.exists():
        all_genes |= {d.name for d in post_dir.iterdir() if d.is_dir()}

    gene_dfs = {}
    for gene in sorted(all_genes):
        dfs = []
        for base_dir in (pre_dir, post_dir):
            summary_path = base_dir / gene / f"{gene}_summary.csv"
            if summary_path.exists():
                try:
                    df = pd.read_csv(summary_path)
                    df["deposition_date"] = pd.to_datetime(df["deposition_date"], errors="coerce")
                    df = df[
                        ~df["ligands"].astype(str).str.contains("peptide", case=False, na=False)
                    ].copy()
                    dfs.append(df)
                except Exception as e:
                    logging.warning(f"Error loading {summary_path}: {e}")
        if dfs:
            gene_dfs[gene] = pd.concat(dfs, ignore_index=True).drop_duplicates(
                subset=["structure_id", "chain_id"], keep="first"
            )
    return gene_dfs


def compute_method_specific_splits(
    gene_dfs: Dict[str, pd.DataFrame],
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Build (canonical_gene_name, inference_condition) -> {method: "Seen"|"Unseen"}.

    For each (gene, ligand_combo) with non-apo ligands: Unseen if combo in new_ligands for that method.
    Apo (No_ligand) is always Seen (we only care about holo ligand novelty).
    """
    split_map = {}  # (gene, inference_condition) -> {method: split}
    methods = list(METHOD_CUTOFFS.keys())

    for gene_name, df in gene_dfs.items():
        for method, cutoff_str in METHOD_CUTOFFS.items():
            cutoff = datetime.strptime(cutoff_str, "%Y-%m-%d")
            post_df = df[df["deposition_date"] > cutoff]
            pre_df = df[df["deposition_date"] <= cutoff]
            new_ligands = get_new_ligands_only(post_df, pre_df)

            for _, row in df.iterrows():
                norm = normalize_ligands(str(row["ligands"]))
                if norm == ("No_ligand",):
                    # Apo: always Seen
                    inf_cond = "apo"
                    key = (gene_name, inf_cond)
                    if key not in split_map:
                        split_map[key] = {m: "Seen" for m in methods}
                    split_map[key][method] = "Seen"
                    continue

                # inference_condition format: "1G0" or "BYP_BWP"
                inf_cond = "_".join(sorted(norm))
                key = (gene_name, inf_cond)
                if key not in split_map:
                    split_map[key] = {m: "Seen" for m in methods}
                split_map[key][method] = "Unseen" if norm in new_ligands else "Seen"

    return split_map


def apply_method_splits_to_results(
    df: pd.DataFrame,
    split_map: Dict[Tuple[str, str], Dict[str, str]],
) -> pd.DataFrame:
    """Reassign split per row using method-specific mapping."""
    df = df.copy()

    def get_split(row):
        gene = row["canonical_gene_name"]
        inf = str(row["inference_condition"]).strip()
        method = row["method"].lower()
        key = (gene, inf)
        lig_key = inference_condition_to_ligands(inf)
        # Try exact match first
        if key in split_map:
            return split_map[key].get(method, row["split"])
        # Try alternative key (inference_condition might use different format)
        alt_key = (gene, "_".join(lig_key)) if lig_key != ("No_ligand",) else (gene, "apo")
        if alt_key in split_map:
            return split_map[alt_key].get(method, row["split"])
        return row["split"]

    df["split"] = df.apply(get_split, axis=1)
    return df


def build_category_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Compute n_holo_only, n_apo_only, n_both, n_neither per (method, split)."""
    rows = []
    for (method, split), grp in df.groupby(["method", "split"]):
        holo_ok = grp["reference_match"] > 0
        apo_ok = grp["linked_match"] > 0
        n_holo_only = ((holo_ok) & (~apo_ok)).sum()
        n_apo_only = ((~holo_ok) & (apo_ok)).sum()
        n_both = (holo_ok & apo_ok).sum()
        n_neither = ((~holo_ok) & (~apo_ok)).sum()
        n_systems = len(grp)
        rows.append(
            {
                "method": method,
                "split": split,
                "n_systems": n_systems,
                "n_holo_only": int(n_holo_only),
                "n_apo_only": int(n_apo_only),
                "n_both": int(n_both),
                "n_neither": int(n_neither),
            }
        )
    return pd.DataFrame(rows)


def create_stacked_bar_plot(
    counts_df: pd.DataFrame,
    output_path: Path,
):
    """Create stacked bar chart for Holo/Apo categories: Seen and Unseen (method-specific)."""
    # Plot whatever methods are present in the counts (PRIMARY_METHODS first, then extras
    # like rfaa) so a 4th method appears without hardcoding the order.
    present = set(counts_df["method"].astype(str).str.lower())
    method_order = [m for m in PRIMARY_METHODS if m in present]
    method_order += [m for m in sorted(present) if m not in method_order]
    labels = [METHOD_DISPLAY.get(m, m) for m in method_order]

    x = np.arange(len(method_order))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)

    for ax, split_filter, panel_title in [
        (ax1, "Seen", "Training Set"),
        (ax2, "Unseen", "Test Set"),
    ]:
        sub = counts_df[counts_df["split"] == split_filter]
        if sub.empty:
            ax.set_visible(False)
            continue

        bottom = np.zeros(len(method_order))
        for key, leg_label, facecolor in _CATEGORY_STACK:
            values = []
            for m in method_order:
                row = sub[sub["method"] == m]
                if row.empty:
                    values.append(0)
                else:
                    n = row.iloc[0]["n_systems"]
                    v = row.iloc[0][key]
                    pct = 100 * v / n if n else 0
                    values.append(pct)
            ax.bar(
                x,
                values,
                bottom=bottom,
                label=leg_label,
                color=facecolor,
                edgecolor="black",
                linewidth=0.5,
            )
            bottom += np.array(values)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FONT_TICK, fontweight="normal")
        ax.set_xlabel("Method", fontsize=FONT_AXIS, fontweight="normal")
        ax.set_ylabel("% of systems", fontsize=FONT_AXIS, fontweight="normal")
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", frameon=True, prop={"size": FONT_LEGEND})
        ax.set_title(panel_title, fontsize=FONT_TITLE, fontweight="normal")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    svg_path = output_path.with_suffix(".svg")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved plot: {output_path} and {svg_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Method-specific cutoff Holo/Apo analysis"
    )
    parser.add_argument(
        "--pre_summaries",
        type=Path,
        default=Path("run_output/cofolding_results_pre_training_successful"),
        help="Pre-training cofolding results (gene summaries)",
    )
    parser.add_argument(
        "--post_summaries",
        type=Path,
        default=Path("run_output/cofolding_results_post_training_successful"),
        help="Post-training cofolding results (gene summaries)",
    )
    parser.add_argument(
        "--all_results",
        type=Path,
        default=None,
        help="Merged all_results_combined.csv (alternative to --pre_csv + --post_csv)",
    )
    parser.add_argument(
        "--pre_csv",
        type=Path,
        default=None,
        help="Pre-training all_results_combined.csv (with --post_csv)",
    )
    parser.add_argument(
        "--post_csv",
        type=Path,
        default=None,
        help="Post-training all_results_combined.csv (with --pre_csv)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated lowercase methods to keep when merging (default: PRIMARY_METHODS)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("run_output/pair_exploratory_analysis_method_specific"),
        help="Output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    script_dir = Path(__file__).parent.resolve()
    workspace = script_dir.parent.parent.parent
    pre_dir = args.pre_summaries if args.pre_summaries.is_absolute() else workspace / args.pre_summaries
    post_dir = args.post_summaries if args.post_summaries.is_absolute() else workspace / args.post_summaries
    output_dir = args.output_dir if args.output_dir.is_absolute() else workspace / args.output_dir

    method_keys = frozenset(parse_methods_arg(args.methods))
    pre_csv = args.pre_csv if args.pre_csv and args.pre_csv.is_absolute() else (
        workspace / args.pre_csv if args.pre_csv else None
    )
    post_csv = args.post_csv if args.post_csv and args.post_csv.is_absolute() else (
        workspace / args.post_csv if args.post_csv else None
    )
    all_results_path = (
        args.all_results if args.all_results and args.all_results.is_absolute() else (
            workspace / args.all_results if args.all_results else None
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if pre_csv is not None and post_csv is not None:
        if not pre_csv.is_file() or not post_csv.is_file():
            logging.error(f"pre_csv/post_csv not found: {pre_csv} / {post_csv}")
            return
        logging.info("Merging pre/post all_results_combined...")
        df = merge_pre_post_pair_csvs(pre_csv, post_csv, method_keys)
    elif all_results_path is not None and all_results_path.is_file():
        logging.info("Loading all_results_combined...")
        df = pd.read_csv(all_results_path)
        df = df[df["classification"] == "holo_inference_apo"].copy()
        df = df[df["method"].str.lower().isin(method_keys)].copy()
        if "split" not in df.columns:
            df["split"] = "Seen"
    else:
        logging.error(
            "Provide --pre_csv and --post_csv, or a valid --all_results path"
        )
        return

    if not pre_dir.exists() and not post_dir.exists():
        logging.error("At least one of pre_summaries or post_summaries must exist")
        return

    # 1. Load summaries and compute method-specific splits
    logging.info("Loading gene summaries...")
    gene_dfs = load_all_summaries(pre_dir, post_dir)
    logging.info(f"Loaded {len(gene_dfs)} genes")
    split_map = compute_method_specific_splits(gene_dfs)

    # 2. Apply method-specific splits (per-row split matches that row's method cutoff)
    df = apply_method_splits_to_results(df, split_map)

    merged_out = output_dir / "all_results_combined.csv"
    df.to_csv(merged_out, index=False)
    logging.info(
        f"Saved: {merged_out} (split = Seen/Unseen per method-specific deposition cutoff)"
    )

    # 3. Build category counts
    counts_df = build_category_counts(df)

    counts_path = output_dir / "pair_holo_apo_category_counts_method_specific.csv"
    counts_df.to_csv(counts_path, index=False)
    logging.info(f"Saved: {counts_path}")

    # 4. Plot (Seen + Unseen)
    plot_path = figures_dir / "pair_holo_apo_categories_method_specific.png"
    create_stacked_bar_plot(counts_df, plot_path)

    logging.info("Done.")


if __name__ == "__main__":
    main()

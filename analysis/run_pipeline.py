#!/usr/bin/env python3
"""
Ordered pipeline: geometric analysis → three-way geometric intersection (Boltz, Chai, Protenix)
→ KinCoRe label / diversity analysis → pair (apo-drift) analysis.

Only systems where all three models have ≥1 geometrically successful prediction are kept.
Successful-samples output keeps ground truth plus cofolding rows only for those three methods.

Example (from kinconfbench repo root):

  python KinConfBench/analysis/run_pipeline.py \\
      --kinconfbench-root /path/to/kinconfbench \\
      --holo-apo-results KinConfBench/analysis/pair_analysis_label/holo_apo_results

Use --skip-geometric if geometric_analysis_csvs are already up to date on the source tree.
Use --skip-pair if you only need the kincore branch.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Repo root (directory that contains the KinConfBench package), not KinConfBench/analysis/.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
THREE = ("boltz", "chai", "protenix")
THREE_COMMA = ",".join(THREE)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n" + "=" * 80)
    print(" ", " ".join(cmd))
    print("=" * 80 + "\n")
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    root_pp = str(REPO_ROOT)
    env["PYTHONPATH"] = f"{root_pp}{os.pathsep}{pp}" if pp else root_pp
    subprocess.run(cmd, check=True, cwd=cwd or REPO_ROOT, env=env)


def copy_paper_assets(figures_dir: Path, paper_dir: Path, *, include_discovery: bool) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    # plot_results.py uses -o <figures_dir>/results → assets live in results/
    res = figures_dir / "results"
    names = [
        "results_success_vs_k.png",
        "results_success_vs_k.svg",
        "results_success_vs_k_ligand_rmsd_stacked.png",
        "results_success_vs_k_ligand_rmsd_stacked.svg",
        "results_success_vs_k_with_ligand_rmsd.png",
        "results_success_vs_k_with_ligand_rmsd.svg",
        "results_average_diversity.png",
        "results_average_diversity.svg",
    ]
    if include_discovery:
        pair_fig = (
            figures_dir
            / "figures"
            / "pair_holo_apo_categories_method_specific.png"
        )
        pair_svg = pair_fig.with_suffix(".svg")
        if pair_fig.is_file():
            shutil.copy2(pair_fig, paper_dir / pair_fig.name)
        if pair_svg.is_file():
            shutil.copy2(pair_svg, paper_dir / pair_svg.name)
    for name in names:
        src = res / name if (res / name).is_file() else figures_dir / name
        if src.is_file():
            shutil.copy2(src, paper_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run geometric → 3-way successful subset → label → pair."
    )
    parser.add_argument(
        "--kinconfbench-root",
        type=Path,
        required=True,
        help="Full KinConfBench data root (analysis_csvs, predictions, rcsb_files, …).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=THREE_COMMA,
        help=(
            "Comma-separated methods to REPORT in kincore/diversity/plots "
            f"(default: {THREE_COMMA}). Add a 4th method (e.g. rfaa) here to overlay it."
        ),
    )
    parser.add_argument(
        "--required-methods",
        type=str,
        default=None,
        help=(
            "Comma-separated methods for the successful-samples SYSTEM GATE "
            "(default: --methods). Set to boltz,chai,protenix to keep the paper's "
            "3-way system set while reporting a 4th method via --methods."
        ),
    )
    parser.add_argument(
        "--keep-methods",
        type=str,
        default=None,
        help=(
            "Comma-separated methods whose prediction rows are RETAINED in the "
            "successful-samples subset (default: --methods). Lets a 4th method's rows "
            "survive even when the gate (--required-methods) is the 3-way set."
        ),
    )
    parser.add_argument(
        "--successful-out",
        type=Path,
        default=REPO_ROOT / "run_output" / "successful_samples_kinconfbench",
        help="Output for create_successful_samples_dataset (3-way intersection).",
    )
    parser.add_argument(
        "--kincore-out",
        type=Path,
        default=REPO_ROOT / "run_output" / "kincore_label_successful",
        help="Prefix directory for comprehensive / extract / diversity outputs.",
    )
    parser.add_argument(
        "--figures-out",
        type=Path,
        default=REPO_ROOT / "run_output" / "figures_successful",
        help="Directory for figure outputs (pair + optional plot_results paths).",
    )
    parser.add_argument(
        "--holo-apo-results",
        type=Path,
        default=None,
        help="Path to holo_apo_results (required unless --skip-pair).",
    )
    parser.add_argument(
        "--pre-csv",
        type=Path,
        default=None,
        help="Override pre-training all_results_combined.csv for pair generalization plot.",
    )
    parser.add_argument(
        "--post-csv",
        type=Path,
        default=None,
        help="Override post-training all_results_combined.csv for pair generalization plot.",
    )
    parser.add_argument(
        "--skip-geometric",
        action="store_true",
        help="Skip run_all_geometric_analysis (expects geometric CSVs on source).",
    )
    parser.add_argument(
        "--skip-pair",
        action="store_true",
        help="Skip create_pre/post, analyze_cofolding, method_performance, pair_holo_apo generalization plot.",
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=None,
        help="Copy headline PNG/SVG figures here (default: repo_root/Figures).",
    )
    parser.add_argument(
        "--skip-geometric-correlation",
        action="store_true",
        help="Skip kincore_geometric_correlation plots (full --kinconfbench-root).",
    )
    parser.add_argument(
        "--pair-pre-dir",
        type=Path,
        default=None,
        help="Output for create_pre_training_results (default: run_output/cofolding_results_pre_training).",
    )
    parser.add_argument(
        "--pair-post-dir",
        type=Path,
        default=None,
        help="Output for create_post_training_results (default: run_output/cofolding_results_post_training).",
    )
    args = parser.parse_args()
    paper_dir = (
        args.paper_dir.resolve()
        if args.paper_dir
        else REPO_ROOT / "Figures"
    )

    # Method threading: report on --methods (4-way incl. rfaa), gate the successful
    # subset on --required-methods (the paper's 3-way set), retain rows for --keep-methods.
    methods_comma = args.methods or THREE_COMMA
    required_comma = args.required_methods or methods_comma
    keep_comma = args.keep_methods or methods_comma

    kin_root = args.kinconfbench_root.resolve()
    if not kin_root.is_dir():
        raise SystemExit(f"--kinconfbench-root not found: {kin_root}")

    successful = args.successful_out.resolve()
    kincore = args.kincore_out.resolve()
    kincore.mkdir(parents=True, exist_ok=True)
    figures = args.figures_out.resolve()
    figures.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # --- 1) Geometric analysis (prerequisite for merged geometric filters) ---
    if not args.skip_geometric:
        run(
            [
                py,
                "-m",
                "KinConfBench.analysis.geometric_analysis.run_all_geometric_analysis",
                str(kin_root),
            ]
        )
    else:
        print("Skipping geometric analysis (--skip-geometric).")

    # --- 2) Successful samples: three-way geometric intersection ---
    run(
        [
            py,
            str(REPO_ROOT / "KinConfBench" / "analysis" / "create_successful_samples_dataset.py"),
            "--source",
            str(kin_root),
            "--output",
            str(successful),
            "--required-methods",
            required_comma,
            "--keep-methods",
            keep_comma,
        ]
    )

    # --- 3) KinCoRe label analysis ---
    comprehensive_prefix = kincore / "comprehensive_results"
    run(
        [
            py,
            str(
                REPO_ROOT
                / "KinConfBench"
                / "analysis"
                / "kincore_label_analysis"
                / "comprehensive_method_analysis.py"
            ),
            str(successful),
            "-o",
            str(comprehensive_prefix),
            "--methods",
            methods_comma,
        ]
    )

    report_path = kincore / "interesting_cases_report.txt"
    diversity_dir = kincore / "csvs_for_diversity_analysis"
    all_correct = kincore / "all_correct_analysis_csvs"
    run(
        [
            py,
            str(
                REPO_ROOT
                / "KinConfBench"
                / "analysis"
                / "kincore_label_analysis"
                / "extract_correct_predictions.py"
            ),
            str(successful),
            "-o",
            str(all_correct),
            "-d",
            str(diversity_dir),
            "-r",
            str(report_path),
            "--methods",
            methods_comma,
        ]
    )

    diversity_prefix = kincore / "diversity_results"
    run(
        [
            py,
            str(
                REPO_ROOT
                / "KinConfBench"
                / "analysis"
                / "kincore_label_analysis"
                / "analyze_correct_prediction_diversity.py"
            ),
            "-i",
            str(diversity_dir),
            "-o",
            str(diversity_prefix),
            "--methods",
            methods_comma,
        ]
    )

    results_prefix = str(figures / "results")
    run(
        [
            py,
            str(
                REPO_ROOT
                / "KinConfBench"
                / "analysis"
                / "kincore_label_analysis"
                / "plot_results.py"
            ),
            "--accuracy",
            str(kincore / "comprehensive_results_statistics.json"),
            "--accuracy-pkl",
            str(kincore / "comprehensive_results_data.pkl"),
            "--diversity",
            str(kincore / "diversity_results_diversity_summary.json"),
            "-o",
            results_prefix,
            "--methods",
            methods_comma,
            "--geometric-base-dir",
            str(kin_root),
            "--geometric-restrict-dataset",
            str(successful),
            "--geometric-success-only",
        ]
    )

    run(
        [
            py,
            str(
                REPO_ROOT
                / "KinConfBench"
                / "analysis"
                / "kincore_label_analysis"
                / "plot_num_correct_distribution.py"
            ),
            str(kincore / "comprehensive_results_data.pkl"),
            "-o",
            str(figures / "results_diversity" / "results_diversity"),
            "--subset-csv-dir",
            str(diversity_dir),
            "--methods",
            methods_comma,
        ]
    )

    common_prefix = str(figures / "common_samples_distribution")
    run(
        [
            py,
            str(
                REPO_ROOT
                / "KinConfBench"
                / "analysis"
                / "kincore_label_analysis"
                / "plot_common_samples_distribution.py"
            ),
            str(kincore / "comprehensive_results_data.pkl"),
            "-o",
            common_prefix,
            "--methods",
            methods_comma,
        ]
    )

    run(
        [
            py,
            str(
                REPO_ROOT
                / "KinConfBench"
                / "analysis"
                / "kincore_label_analysis"
                / "plot_common_label_correct_summary.py"
            ),
            str(kincore / "comprehensive_results_data.pkl"),
            "-o",
            str(figures / "common_label_correct"),
            "--methods",
            methods_comma,
        ]
    )

    if not args.skip_geometric_correlation:
        geom_plot_base = figures / "kincore_geometric_correlation"
        run(
            [
                py,
                str(
                    REPO_ROOT
                    / "KinConfBench"
                    / "analysis"
                    / "kincore_label_analysis"
                    / "geometric_correctness_correlation.py"
                ),
                "--base-dir",
                str(kin_root),
                "-o",
                str(geom_plot_base),
                "--success-only",
                "--restrict-to-dataset",
                str(successful),
                "--methods",
                methods_comma,
            ]
        )

    print(
        "\nKincore branch complete. Outputs:\n"
        f"  {comprehensive_prefix}_statistics.json\n"
        f"  {diversity_prefix}_diversity_summary.json\n"
        f"  Figures under {figures}\n"
    )

    if args.skip_pair:
        copy_paper_assets(figures, paper_dir, include_discovery=False)
        print(f"Paper assets (no pair generalization plot): {paper_dir}")
        print("--skip-pair: not running holo/apo pair steps.")
        return

    ho = args.holo_apo_results
    if ho is None:
        raise SystemExit("Pair step requires --holo-apo-results (or pass --skip-pair).")
    ho = ho.resolve()
    if not ho.is_dir():
        raise SystemExit(f"--holo-apo-results not found: {ho}")

    pre_root = (
        args.pair_pre_dir.resolve()
        if args.pair_pre_dir
        else REPO_ROOT / "run_output" / "cofolding_results_pre_training"
    )
    post_root = (
        args.pair_post_dir.resolve()
        if args.pair_post_dir
        else REPO_ROOT / "run_output" / "cofolding_results_post_training"
    )

    run(
        [
            py,
            str(
                REPO_ROOT
                / "KinConfBench"
                / "analysis"
                / "pair_analysis"
                / "create_pre_training_results.py"
            ),
            "--source",
            str(successful),
            "--output",
            str(pre_root),
        ]
    )
    run(
        [
            py,
            str(
                REPO_ROOT
                / "KinConfBench"
                / "analysis"
                / "pair_analysis"
                / "create_post_training_results.py"
            ),
            "--source",
            str(successful),
            "--output",
            str(post_root),
        ]
    )

    run(
        [
            py,
            str(
                REPO_ROOT
                / "KinConfBench"
                / "analysis"
                / "pair_analysis"
                / "analyze_cofolding_results.py"
            ),
            "--holo_apo_results",
            str(ho),
            "--cofolding_results_dir",
            str(pre_root),
        ]
    )
    run(
        [
            py,
            str(
                REPO_ROOT
                / "KinConfBench"
                / "analysis"
                / "pair_analysis"
                / "analyze_cofolding_results.py"
            ),
            "--holo_apo_results",
            str(ho),
            "--cofolding_results_dir",
            str(post_root),
        ]
    )

    run(
        [
            py,
            str(
                REPO_ROOT
                / "KinConfBench"
                / "analysis"
                / "pair_analysis"
                / "analyze_method_performance.py"
            ),
            "--results_dir",
            str(ho),
            "--cofolding_results_dir",
            str(pre_root),
            "--create_plots",
        ]
    )
    run(
        [
            py,
            str(
                REPO_ROOT
                / "KinConfBench"
                / "analysis"
                / "pair_analysis"
                / "analyze_method_performance.py"
            ),
            "--results_dir",
            str(ho),
            "--cofolding_results_dir",
            str(post_root),
            "--create_plots",
        ]
    )

    pre_csv = (
        args.pre_csv.resolve()
        if args.pre_csv
        else ho / "method_performance_analysis_pre_training" / "all_results_combined.csv"
    )
    post_csv = (
        args.post_csv.resolve()
        if args.post_csv
        else ho / "method_performance_analysis_post_training" / "all_results_combined.csv"
    )
    if not pre_csv.is_file() or not post_csv.is_file():
        raise SystemExit(
            f"Pair CSVs not found.\n  pre:  {pre_csv}\n  post: {post_csv}\n"
            "Set --pre-csv / --post-csv if paths differ."
        )

    run(
        [
            py,
            str(
                REPO_ROOT
                / "KinConfBench"
                / "analysis"
                / "pair_analysis"
                / "pair_holo_apo_method_specific_analysis.py"
            ),
            "--pre_summaries",
            str(pre_root.resolve()),
            "--post_summaries",
            str(post_root.resolve()),
            "--pre_csv",
            str(pre_csv),
            "--post_csv",
            str(post_csv),
            "--output_dir",
            str(figures.resolve()),
            "--methods",
            methods_comma,
        ]
    )

    copy_paper_assets(figures, paper_dir, include_discovery=True)
    print(
        "\nPipeline finished.\n"
        f"  Successful 3-way subset: {successful}\n"
        f"  Kincore: {kincore}\n"
        f"  Figures (incl. pair_holo_apo generalization): {figures}\n"
        f"  Paper plots: {paper_dir}\n"
    )


if __name__ == "__main__":
    main()

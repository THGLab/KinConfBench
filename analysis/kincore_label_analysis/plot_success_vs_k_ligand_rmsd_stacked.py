#!/usr/bin/env python3
"""
CLI wrapper for the stacked RMSD + Success@K figure (same implementation as plot_results.py).

Prefer running ``plot_results.py`` with ``--geometric-base-dir`` (and filters); that writes
``results_success_vs_k.*`` and ``results_success_vs_k_ligand_rmsd_stacked.*``.

Usage:
    PYTHONPATH=. python KinConfBench/analysis/kincore_label_analysis/plot_success_vs_k_ligand_rmsd_stacked.py \\
        --accuracy-pkl run_output/kincore_label_successful/comprehensive_results_data.pkl \\
        --geometric-base-dir kinase_all_data/kinconfbench \\
        --geometric-restrict-dataset run_output/successful_samples_kinconfbench \\
        --geometric-success-only \\
        -o run_output/figures_successful/results \\
        --methods boltz,chai,protenix
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from KinConfBench.analysis.kincore_label_analysis.plot_results import (
    plot_success_vs_k_ligand_rmsd_stacked,
    prepare_geometric_merged_for_plots,
)
from KinConfBench.analysis.plot_style import PRIMARY_METHODS, parse_methods_arg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stacked pooled ligand RMSD + Success@K (delegates to plot_results)."
    )
    parser.add_argument("--accuracy-pkl", type=Path, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--geometric-base-dir", type=Path, required=True)
    parser.add_argument("--geometric-restrict-dataset", type=Path, default=None)
    parser.add_argument("--geometric-success-only", action="store_true")
    parser.add_argument("--max-k", type=int, default=20)
    args = parser.parse_args()

    methods = parse_methods_arg(args.methods, default=PRIMARY_METHODS)
    gdf = prepare_geometric_merged_for_plots(
        args.geometric_base_dir,
        args.geometric_restrict_dataset,
        args.geometric_success_only,
        methods,
    )
    if gdf is None:
        print("Geometric merge failed or empty.", file=sys.stderr)
        sys.exit(1)
    plot_success_vs_k_ligand_rmsd_stacked(
        args.accuracy_pkl,
        gdf,
        args.output,
        max_k=args.max_k,
        methods=methods,
    )


if __name__ == "__main__":
    main()

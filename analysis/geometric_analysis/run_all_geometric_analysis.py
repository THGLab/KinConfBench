#!/usr/bin/env python3
"""
Run geometric analysis for all gene_ligand stems in kinconfbench.

Iterates over analysis_csvs/*_analysis.csv, extracts (gene, stem), and runs
geometric analysis for each. Optionally supports batch/slurm.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .run_geometric_analysis import run_geometric_analysis


def _run_one(args: tuple[Path, str, str]) -> tuple[str, str, Path]:
    """Worker for parallel execution; must be at module level for pickling."""
    root, gene, stem = args
    gene_folder = root / gene
    out = run_geometric_analysis(gene_folder, stem)
    return gene, stem, out


def discover_stems(kinconfbench_root: Path) -> list[tuple[str, str]]:
    """
    Discover all (gene, stem) from analysis_csvs.

    Returns:
        List of (gene_folder_name, stem) e.g. [("ABL1_HUMAN", "ABL1_HUMAN_1N1"), ...]
    """
    stems = []
    for gene_dir in sorted(kinconfbench_root.iterdir()):
        if not gene_dir.is_dir():
            continue
        acsv_dir = gene_dir / "analysis_csvs"
        if not acsv_dir.exists():
            continue
        for csv_path in acsv_dir.glob("*_analysis.csv"):
            stem = csv_path.stem.replace("_analysis", "")
            stems.append((gene_dir.name, stem))
    return stems


def main():
    parser = argparse.ArgumentParser(
        description="Run geometric analysis for all kinconfbench entries"
    )
    parser.add_argument(
        "kinconfbench_root",
        type=str,
        help="Root of kinconfbench data (e.g. kinase_all_data/kinconfbench)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list stems, do not run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of stems to process (for testing)",
    )
    parser.add_argument(
        "--gene",
        type=str,
        default=None,
        help="Only process stems for this gene folder (e.g. ABL1_HUMAN)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1). Use e.g. -j 8 to run 8 stems in parallel.",
    )
    args = parser.parse_args()

    root = Path(args.kinconfbench_root)
    if not root.exists():
        print(f"Error: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    stems = discover_stems(root)
    if args.gene is not None:
        stems = [(g, s) for g, s in stems if g == args.gene]
        print(f"Filtered to {len(stems)} stems for gene {args.gene}")
    if args.limit is not None:
        stems = stems[: args.limit]
    print(f"Found {len(stems)} stems")

    if args.dry_run:
        for gene, stem in stems:
            print(f"  {gene} / {stem}")
        return

    failed = []
    work = [(root, gene, stem) for gene, stem in stems]
    n_jobs = max(1, min(args.jobs, len(work)))
    if n_jobs > 1:
        print(f"Running with {n_jobs} parallel jobs")

    if n_jobs == 1:
        for _, gene, stem in work:
            gene_folder = root / gene
            try:
                out = run_geometric_analysis(gene_folder, stem)
                print(f"OK {stem} -> {out}")
            except Exception as e:
                print(f"FAIL {stem}: {e}", file=sys.stderr)
                failed.append((stem, str(e)))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_run_one, w): w for w in work}
            for future in as_completed(futures):
                _, _, stem = futures[future]
                try:
                    _, _, out = future.result()
                    print(f"OK {stem} -> {out}")
                except Exception as e:
                    print(f"FAIL {stem}: {e}", file=sys.stderr)
                    failed.append((stem, str(e)))

    if failed:
        print(f"\n{len(failed)} failed", file=sys.stderr)
        for s, msg in failed[:5]:
            print(f"  {s}: {msg}", file=sys.stderr)
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

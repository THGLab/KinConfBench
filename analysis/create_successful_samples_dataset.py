#!/usr/bin/env python3
"""
Create a filtered kinconfbench dataset containing only geometrically successful samples.

Success = lDDT-Cα ≥ 0.7 AND lDDT-PLI ≥ 0.8 AND ligand_rmsd < 2 Å.

For each stem, the analysis CSV is filtered to keep:
- All ground truth rows (reference structures)
- Only cofolding prediction rows that meet the success criteria

The output directory mirrors kinconfbench structure (gene/summary, gene/analysis_csvs)
so it can be used as input to the full analysis pipeline (comprehensive_method_analysis,
create_pre_training_results, create_post_training_results, etc.).

Usage:
    python create_successful_samples_dataset.py --source kinase_all_data/kinconfbench \\
        --output analysis/run_output/successful_samples_kinconfbench
"""

from __future__ import annotations

import argparse
import csv
import sys
import shutil
from pathlib import Path

from tqdm import tqdm

# Ensure workspace root is on path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _SCRIPT_DIR.parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from KinConfBench.analysis.kincore_label_analysis.geometric_correctness_correlation import (
    load_merged_data,
    extract_method,
)


def is_ground_truth(structure_id: str) -> bool:
    sid = str(structure_id).strip()
    return (
        len(sid) == 4
        and sid.replace("_", "").replace("-", "").isalnum()
    )


DEFAULT_REQUIRED_METHODS = ("boltz", "chai", "protenix")


def create_successful_dataset(
    source_dir: Path,
    output_dir: Path,
    lddt_ca_min: float = 0.7,
    lddt_pli_min: float = 0.8,
    ligand_rmsd_max: float = 2.0,
    *,
    required_methods: tuple[str, ...] = DEFAULT_REQUIRED_METHODS,
    keep_methods: tuple[str, ...] | None = None,
    drop_methods: tuple[str, ...] = (),
) -> int:
    """
    Create filtered dataset. Returns number of analysis CSV files written.

    A stem is written only if every name in ``required_methods`` has at least one
    geometrically successful cofolding row after filtering (intersection over methods) —
    this is the *system gate* and defines which systems are kept.

    Cofolding rows are kept only for methods listed in ``keep_methods`` (defaults to
    ``required_methods``; ground truth rows are always kept). Decoupling the two lets a
    new method (e.g. rfaa) be retained as an extra column of predictions while the system
    gate stays on the original methods — i.e. "evaluate rfaa on the same systems the 3-way
    figures use." ``drop_methods`` additionally removes prediction rows for any extra
    methods (rare; empty by default).
    """
    drop_set = frozenset(m.strip().lower() for m in drop_methods if m.strip())
    req = tuple(m.strip().lower() for m in required_methods if m.strip())
    req_set = frozenset(req)
    if not req:
        raise ValueError("required_methods must be non-empty")
    keep = tuple(m.strip().lower() for m in (keep_methods if keep_methods else required_methods) if m.strip())
    keep_set = frozenset(keep)
    print(f"Required methods (system gate / geometric intersection): {', '.join(req)}")
    print(f"Keeping prediction rows for methods: {', '.join(keep)}")
    if drop_set:
        print(f"Dropping prediction rows for methods: {', '.join(sorted(drop_set))}")

    print(f"Loading merged data from {source_dir}...")
    merged = load_merged_data(source_dir)
    if merged.empty:
        raise ValueError("No merged data found.")

    # (stem, structure_id) pool for per-row geometric pass (internal; not system count).
    # Apo systems (no ligand -> lddt_pli/ligand_rmsd undefined) pass on the lDDT-Cα gate
    # ALONE; holo systems use the full lDDT-Cα + lDDT-PLI + ligand-RMSD gate (a holo row
    # with missing ligand metrics fails the NaN comparisons, as intended).
    df = merged.dropna(subset=["lddt_ca"]).copy()
    is_apo = df["stem"].astype(str).str.endswith("_apo")
    holo_success = (
        (~is_apo)
        & (df["lddt_ca"] >= lddt_ca_min)
        & (df["lddt_pli"] >= lddt_pli_min)
        & (df["ligand_rmsd"] < ligand_rmsd_max)
    )
    apo_success = is_apo & (df["lddt_ca"] >= lddt_ca_min)
    success_mask = holo_success | apo_success
    success_set = set(zip(df[success_mask]["stem"], df[success_mask]["structure_id"]))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files_written = 0

    for gene_dir in tqdm(sorted(d for d in source_dir.iterdir() if d.is_dir()), desc="Creating dataset"):
        gene_name = gene_dir.name
        analysis_dir = gene_dir / "analysis_csvs"
        summary_file = gene_dir / f"{gene_name}_summary.csv"

        if not analysis_dir.exists() or not summary_file.exists():
            continue

        target_gene = output_dir / gene_name
        target_gene.mkdir(exist_ok=True)
        target_analysis = target_gene / "analysis_csvs"
        target_analysis.mkdir(exist_ok=True)

        # Copy summary
        shutil.copy2(summary_file, target_gene / f"{gene_name}_summary.csv")

        for analysis_csv in analysis_dir.glob("*_analysis.csv"):
            stem = analysis_csv.stem.replace("_analysis", "")

            with open(analysis_csv) as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                rows = list(reader)

            # Filter: keep ground truth + successful cofolding rows
            def keep_row(r):
                sid = r.get("structure_id", "").strip()
                if is_ground_truth(sid):
                    return True
                method = extract_method(sid)
                if method is None:
                    return False
                if method not in keep_set:
                    return False
                if method in drop_set:
                    return False
                return (stem, sid) in success_set

            filtered_rows = [r for r in rows if keep_row(r)]

            # Must have ground truth
            gt_rows = [r for r in filtered_rows if is_ground_truth(r.get("structure_id", ""))]
            if not gt_rows:
                continue

            cofolding_methods = set()
            for r in filtered_rows:
                sid = r.get("structure_id", "").strip()
                if is_ground_truth(sid):
                    continue
                m = extract_method(sid)
                if m is not None:
                    cofolding_methods.add(m)
            if not all(m in cofolding_methods for m in req):
                continue

            target_csv = target_analysis / analysis_csv.name
            with open(target_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(filtered_rows)
            files_written += 1

    req_label = ", ".join(req)
    print(
        f"Geometric subset: {files_written} systems "
        f"(each has ≥1 successful prediction for every required method: {req_label})."
    )
    return files_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create kinconfbench subset with only geometrically successful samples"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("kinase_all_data/kinconfbench"),
        help="Source kinconfbench directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/run_output/successful_samples_kinconfbench"),
        help="Output directory (mirrors kinconfbench structure)",
    )
    parser.add_argument("--lddt-ca", type=float, default=0.7)
    parser.add_argument("--lddt-pli", type=float, default=0.8)
    parser.add_argument("--ligand-rmsd-max", type=float, default=2.0)
    parser.add_argument(
        "--required-methods",
        type=str,
        default=",".join(DEFAULT_REQUIRED_METHODS),
        help=(
            "Comma-separated methods; stem is kept only if each has ≥1 geometrically "
            "successful prediction after filtering (default: boltz, chai, protenix)."
        ),
    )
    parser.add_argument(
        "--keep-methods",
        type=str,
        default="",
        help=(
            "Comma-separated methods whose prediction rows are retained in the output "
            "(default: same as --required-methods). Use this to keep an extra method "
            "(e.g. rfaa) while the system gate stays on --required-methods."
        ),
    )
    parser.add_argument(
        "--drop-methods",
        type=str,
        default="",
        help="Comma-separated methods whose prediction rows are omitted from output (optional).",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()

    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    required = tuple(x.strip().lower() for x in args.required_methods.split(",") if x.strip())
    keep = tuple(x.strip().lower() for x in args.keep_methods.split(",") if x.strip())
    drop = tuple(x.strip().lower() for x in args.drop_methods.split(",") if x.strip())
    create_successful_dataset(
        source,
        output,
        lddt_ca_min=args.lddt_ca,
        lddt_pli_min=args.lddt_pli,
        ligand_rmsd_max=args.ligand_rmsd_max,
        required_methods=required,
        keep_methods=keep if keep else None,
        drop_methods=drop,
    )
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()

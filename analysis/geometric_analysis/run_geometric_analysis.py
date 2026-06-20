#!/usr/bin/env python3
"""
Run geometric analysis for a single gene_ligand (stem).

Computes protein_rmsd, lddt_ca, lddt_pli, ligand_rmsd for each prediction
vs the reference structure. Writes results to geometric_analysis_csvs/{stem}_geometric.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .geometric_metrics import (
    load_structure,
    compute_protein_rmsd,
    compute_lddt_ca,
    compute_ligand_metrics_per_ligand,
    _parse_dfg_resnums,
)


def get_reference_row(df: pd.DataFrame) -> pd.Series | None:
    """
    Return the first reference row (experimental structure, not cofolding prediction).

    When multiple reference PDBs exist, uses only the first one (by row order).
    """
    candidates = []
    for _, row in df.iterrows():
        mt = row.get("model_type", "")
        if pd.isna(mt) or str(mt).strip() == "":
            candidates.append(row)
    if not candidates:
        return None
    return candidates[0]


def get_prediction_rows(df: pd.DataFrame) -> list[pd.Series]:
    """Return rows for cofolding predictions."""
    return [row for _, row in df.iterrows() if str(row.get("model_type", "")).strip() == "cofolding"]


def run_geometric_analysis(
    gene_folder: str | Path,
    stem: str,
    *,
    data_root: str | Path | None = None,
) -> Path:
    """
    Run geometric analysis for stem under gene_folder.

    Args:
        gene_folder: Path to gene folder (e.g. .../ABL1_HUMAN)
        stem: Stem (e.g. ABL1_HUMAN_1N1)
        data_root: Override root; if None, gene_folder is used as base

    Returns:
        Path to output CSV
    """
    gene_folder = Path(gene_folder)
    base = Path(data_root) if data_root else gene_folder

    analysis_csv = gene_folder / "analysis_csvs" / f"{stem}_analysis.csv"
    if not analysis_csv.exists():
        raise FileNotFoundError(f"Analysis CSV not found: {analysis_csv}")

    df = pd.read_csv(analysis_csv)
    ref_row = get_reference_row(df)
    if ref_row is None:
        raise ValueError(f"No reference row found in {analysis_csv}")

    ref_struct = str(ref_row["structure_id"]).strip()
    ref_chain = str(ref_row["chain_id"]).strip()
    ref_cif = base / "rcsb_files" / stem / f"{ref_struct}_{ref_chain}.cif"
    if not ref_cif.exists():
        raise FileNotFoundError(f"Reference CIF not found: {ref_cif}")

    pred_dir = base / "predictions" / stem
    if not pred_dir.exists():
        raise FileNotFoundError(f"Predictions dir not found: {pred_dir}")

    out_dir = gene_folder / "geometric_analysis_csvs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{stem}_geometric.csv"

    # Load reference
    ref_ent, ref_ligands = load_structure(str(ref_cif), extract_ligands=True)
    ref_dfg = _parse_dfg_resnums(ref_row)

    pred_rows = get_prediction_rows(df)
    results = []

    for row in pred_rows:
        struct_id = str(row["structure_id"]).strip()
        pred_cif = pred_dir / f"{struct_id}.cif"
        if not pred_cif.exists():
            results.append({
                "structure_id": struct_id,
                "protein_rmsd": None,
                "lddt_ca": None,
                "lddt_pli": None,
                "ligand_rmsd": None,
            })
            continue

        try:
            pred_ent, pred_ligands = load_structure(
                str(pred_cif), extract_ligands=True, sanitize_comp_ids=True
            )
        except Exception:
            results.append({
                "structure_id": struct_id,
                "protein_rmsd": None,
                "lddt_ca": None,
                "lddt_pli": None,
                "ligand_rmsd": None,
            })
            continue

        pred_dfg = _parse_dfg_resnums(row)
        protein_rmsd = compute_protein_rmsd(
            ref_ent, pred_ent, ref_dfg=ref_dfg, pred_dfg=pred_dfg
        )
        lddt_ca = compute_lddt_ca(ref_ent, pred_ent)
        per_lig = compute_ligand_metrics_per_ligand(
            ref_ent, ref_ligands, pred_ent, pred_ligands,
        )

        row_data: dict = {
            "structure_id": struct_id,
            "protein_rmsd": protein_rmsd,
            "lddt_ca": lddt_ca,
        }
        if per_lig:
            row_data["lddt_pli"] = per_lig[0]["lddt_pli"]
            row_data["ligand_rmsd"] = per_lig[0]["ligand_rmsd"]
            for i, m in enumerate(per_lig, 1):
                row_data[f"ligand_{i}_name"] = m["ligand_name"]
                row_data[f"ligand_{i}_lddt_pli"] = m["lddt_pli"]
                row_data[f"ligand_{i}_rmsd"] = m["ligand_rmsd"]
        else:
            row_data["lddt_pli"] = None
            row_data["ligand_rmsd"] = None

        results.append(row_data)

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)
    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Run geometric analysis for a gene_ligand stem")
    parser.add_argument("gene_folder", type=str, help="Path to gene folder (e.g. .../ABL1_HUMAN)")
    parser.add_argument("stem", type=str, help="Stem (e.g. ABL1_HUMAN_1N1)")
    parser.add_argument("--data-root", type=str, default=None, help="Override data root (default: gene_folder)")
    args = parser.parse_args()

    out = run_geometric_analysis(
        args.gene_folder,
        args.stem,
        data_root=args.data_root,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

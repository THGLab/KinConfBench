#!/usr/bin/env python3
"""
Create cofolding_results_post_training directory with filtered data.

Keeps only entries with ligands that appear ONLY after cutoff (2023-06-01),
plus all apo structures.

Usage:
    python create_post_training_results.py --source /path/to/kinconfbench \\
        --output /path/to/cofolding_results_post_training
"""

import argparse
import logging
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime
from typing import Set, Tuple


def normalize_ligands(ligands_str: str) -> Tuple[str, ...]:
    if pd.isna(ligands_str) or ligands_str == 'No_ligand':
        return ('No_ligand',)
    ligand_codes = [lig.split(':')[0].strip() for lig in str(ligands_str).split(';')]
    return tuple(sorted(ligand_codes))


def get_new_ligands_only(post_df: pd.DataFrame, pre_df: pd.DataFrame) -> Set[Tuple[str, ...]]:
    post_ligands = set()
    for s in post_df['ligands'].dropna():
        s = str(s)
        if 'peptide' in s.lower():
            continue
        post_ligands.add(normalize_ligands(s))
    pre_ligands = set()
    for s in pre_df['ligands'].dropna():
        s = str(s)
        if 'peptide' in s.lower():
            continue
        pre_ligands.add(normalize_ligands(s))
    return post_ligands - pre_ligands


def filter_analysis_csvs(analysis_csvs_dir: Path, ligands_to_keep: Set[Tuple[str, ...]],
                        gene_name: str) -> int:
    if not analysis_csvs_dir.exists():
        return 0
    kept = 0
    for csv_file in list(analysis_csvs_dir.glob("*.csv")):
        fn = csv_file.name
        if not (fn.startswith(f"{gene_name}_") and fn.endswith("_analysis.csv")):
            continue
        ligand_part = fn[len(f"{gene_name}_"):-len("_analysis.csv")]
        norm = ('No_ligand',) if ligand_part == "apo" else tuple(sorted(ligand_part.split('_')))
        if norm in ligands_to_keep:
            kept += 1
        else:
            csv_file.unlink()
    return kept


def create_post_training_results(source_dir: Path, output_dir: Path, cutoff_date: str):
    cutoff = datetime.strptime(cutoff_date, '%Y-%m-%d')
    output_dir.mkdir(parents=True, exist_ok=True)
    genes_with_data = 0
    for gene_dir in sorted(source_dir.iterdir()):
        if not gene_dir.is_dir():
            continue
        gene_name = gene_dir.name
        target = output_dir / gene_name
        target.mkdir(exist_ok=True)
        summary = gene_dir / f"{gene_name}_summary.csv"
        if not summary.exists():
            target.rmdir()
            continue
        try:
            df = pd.read_csv(summary)
            df['deposition_date'] = pd.to_datetime(df['deposition_date'], errors='coerce')
            df = df[~df['ligands'].astype(str).str.contains('peptide', case=False, na=False)].copy()
            post_df = df[df['deposition_date'] > cutoff]
            pre_df = df[df['deposition_date'] <= cutoff]
            new_ligands = get_new_ligands_only(post_df, pre_df)
            all_apo = df[df['ligands'].apply(
                lambda x: str(x) == 'No_ligand' if pd.notna(x) else False)]
            new_ligands_df = post_df[post_df['ligands'].apply(
                lambda x: normalize_ligands(str(x)) in new_ligands)]
            filtered = pd.concat([all_apo, new_ligands_df]).drop_duplicates()
            if len(filtered) == 0:
                target.rmdir()
                continue
            filtered.to_csv(target / f"{gene_name}_summary.csv", index=False)
            src_csvs = gene_dir / "analysis_csvs"
            if src_csvs.exists():
                tgt_csvs = target / "analysis_csvs"
                tgt_csvs.mkdir(exist_ok=True)
                ligands_to_keep = new_ligands | {('No_ligand',)}
                for f in src_csvs.glob("*.csv"):
                    shutil.copy2(f, tgt_csvs / f.name)
                kept = filter_analysis_csvs(tgt_csvs, ligands_to_keep, gene_name)
                if kept == 0:
                    shutil.rmtree(tgt_csvs)
            genes_with_data += 1
            logging.info(f"{gene_name}: {len(filtered)} rows, {len(new_ligands)} new ligand combos")
        except Exception as e:
            logging.warning(f"{gene_name}: {e}")
            if target.exists():
                shutil.rmtree(target)
    logging.info(f"Finished. Genes with data: {genes_with_data}")


def main():
    parser = argparse.ArgumentParser(description="Create post-training filtered results")
    parser.add_argument("--source", type=Path, required=True,
                       help="Source directory (kinconfbench or cofolding_results)")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output directory (cofolding_results_post_training)")
    parser.add_argument("--cutoff_date", type=str, default="2023-06-01")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    create_post_training_results(args.source, args.output, args.cutoff_date)


if __name__ == "__main__":
    main()

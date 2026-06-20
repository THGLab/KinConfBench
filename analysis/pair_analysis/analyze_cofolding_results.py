#!/usr/bin/env python3
"""
Analyze cofolding inference results for unique pair combinations.

For each unique combo (grouped by structural labels and normalized ligands):
1. Construct combo name (sorted for holo_holo to prevent swapping)
2. Load corresponding analysis CSV files from cofolding_results
3. For each inference condition (struc1 or struc2), count exact matches to ground truth
4. Output results to {gene_name}/analysis_results/{combo_name}.csv
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
import argparse


def normalize_ligands(ligands_str: str) -> List[str]:
    """
    Normalize ligand string by extracting identifiers, preserving order.
    
    Returns a list of ligand identifiers in the order they appear.
    Examples:
        "BYP:299;BWP:300" -> ["BYP", "BWP"]
        "DBQ:398" -> ["DBQ"]
        "No_ligand" -> [] (empty list)
    """
    if pd.isna(ligands_str) or ligands_str == "No_ligand" or ligands_str == "":
        return []
    
    ligand_parts = ligands_str.split(";")
    identifiers = []
    for part in ligand_parts:
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            identifier = part.split(":")[0]
        else:
            identifier = part
        identifiers.append(identifier)
    
    return identifiers


def construct_combo_name(ligands1: List[str], ligands2: List[str], pair_type: str) -> str:
    """
    Construct combo name from two ligand lists.
    
    For holo_holo pairs, sort ligand1 and ligand2 to prevent swapping,
    but preserve order within multi-ligand groups.
    
    Args:
        ligands1: List of ligand identifiers for structure1
        ligands2: List of ligand identifiers for structure2
        pair_type: 'holo_apo' or 'holo_holo'
        
    Returns:
        Combo name like '1N1_struc1_BYP_BWP_struc2' or 'apo_struc1_1G0_struc2'
    """
    # Convert lists to strings for comparison
    lig1_str = "_".join(ligands1) if ligands1 else "apo"
    lig2_str = "_".join(ligands2) if ligands2 else "apo"
    
    if pair_type == 'holo_holo':
        # Sort to prevent swapping, but preserve order within multi-ligand groups
        if lig1_str < lig2_str:
            return f"{lig1_str}_struc1_{lig2_str}_struc2"
        else:
            return f"{lig2_str}_struc1_{lig1_str}_struc2"
    else:  # holo_apo
        # For holo_apo, structure1 is holo, structure2 is apo
        if ligands1:  # struc1 is holo
            lig1_str = "_".join(ligands1)
            return f"{lig1_str}_struc1_apo_struc2"
        else:  # struc1 is apo, struc2 is holo
            lig2_str = "_".join(ligands2)
            return f"apo_struc1_{lig2_str}_struc2"


def get_ground_truth_labels(pairs_df: pd.DataFrame, struc_num: int) -> List[Dict]:
    """
    Extract ground truth labels for a structure from pairs DataFrame.
    
    Args:
        pairs_df: DataFrame with pairs data
        struc_num: 1 for structure1, 2 for structure2
        
    Returns:
        List of dictionaries with categorical label values
    """
    label_columns = [
        'activity_label', 'spatial_label', 'dihedral_label', 'chelix_label',
        'saltbr_label', 'actloop_nt_label', 'actloop_ct_label', 'spine_label'
    ]
    
    ground_truths = []
    for _, row in pairs_df.iterrows():
        labels = {}
        for col in label_columns:
            col_name = f'structure{struc_num}_{col}'
            if col_name in row:
                labels[col] = row[col_name]
        ground_truths.append(labels)
    
    return ground_truths


def extract_cofolding_method(structure_id: str) -> str:
    """
    Extract cofolding method from structure_id by splitting on "_" and taking first part.
    
    Examples:
        "boltz_conf0" -> "boltz"
        "alphafold3_conf1" -> "alphafold3"
        "structure_id" -> "structure_id" (if no underscore)
    """
    if pd.isna(structure_id):
        return "unknown"
    structure_id_str = str(structure_id)
    if "_" in structure_id_str:
        return structure_id_str.split("_")[0]
    return structure_id_str


def count_exact_matches_by_method(analysis_df: pd.DataFrame, ground_truths: List[Dict]) -> pd.DataFrame:
    """
    Count rows in analysis_df that exactly match any ground truth, grouped by cofolding method.
    Also calculates percentage of matches for each individual categorical label.
    
    The cofolding method is extracted from structure_id by splitting on "_" and taking the first part.
    Only includes rows where structure_id starts with: boltz, chai, or protenix.
    
    Args:
        analysis_df: DataFrame with inference results
        ground_truths: List of dictionaries with ground truth label values
        
    Returns:
        DataFrame with columns: method, total_rows, matches, percentage, and individual label percentages
    """
    if analysis_df.empty or not ground_truths:
        base_columns = ['method', 'total_rows', 'matches', 'percentage']
        label_columns = [
            'activity_label', 'spatial_label', 'dihedral_label', 'chelix_label',
            'saltbr_label', 'actloop_nt_label', 'actloop_ct_label', 'spine_label'
        ]
        label_perc_columns = [f'{col}_percentage' for col in label_columns]
        return pd.DataFrame(columns=base_columns + label_perc_columns)
    
    label_columns = [
        'activity_label', 'spatial_label', 'dihedral_label', 'chelix_label',
        'saltbr_label', 'actloop_nt_label', 'actloop_ct_label', 'spine_label'
    ]
    
    # Filter to only columns that exist in analysis_df
    available_labels = [col for col in label_columns if col in analysis_df.columns]
    
    if not available_labels:
        logging.warning("  No label columns found in analysis CSV")
        base_columns = ['method', 'total_rows', 'matches', 'percentage']
        label_perc_columns = [f'{col}_percentage' for col in label_columns]
        return pd.DataFrame(columns=base_columns + label_perc_columns)
    
    # Check if structure_id column exists
    if 'structure_id' not in analysis_df.columns:
        logging.warning("  No 'structure_id' column found in analysis CSV")
        base_columns = ['method', 'total_rows', 'matches', 'percentage']
        label_perc_columns = [f'{col}_percentage' for col in label_columns]
        return pd.DataFrame(columns=base_columns + label_perc_columns)
    
    # Filter to only computational methods: boltz, chai, protenix, rfaa
    valid_methods = {'boltz', 'chai', 'protenix', 'rfaa'}
    analysis_df = analysis_df.copy()
    analysis_df['cofolding_method'] = analysis_df['structure_id'].apply(extract_cofolding_method)
    
    # Filter to only include valid computational methods
    analysis_df = analysis_df[analysis_df['cofolding_method'].isin(valid_methods)]
    
    if analysis_df.empty:
        base_columns = ['method', 'total_rows', 'matches', 'percentage']
        label_perc_columns = [f'{col}_percentage' for col in label_columns]
        return pd.DataFrame(columns=base_columns + label_perc_columns)
    
    # Group by cofolding method and count matches
    method_results = []
    
    for method, method_df in analysis_df.groupby('cofolding_method'):
        total_rows = len(method_df)
        exact_matches = 0
        
        # Track matches for each label individually
        label_matches = {col: 0 for col in available_labels}
        
        for _, row in method_df.iterrows():
            row_labels = {col: row[col] for col in available_labels if col in row}
            
            # Check if this row matches any ground truth
            is_exact_match = False
            matched_gt = None
            
            for gt in ground_truths:
                # Check for exact match
                if all(row_labels.get(col) == gt.get(col) for col in available_labels):
                    is_exact_match = True
                    matched_gt = gt
                    break
                # Also track which labels match (for partial matching)
                elif matched_gt is None:
                    # Use the first ground truth that has any matches
                    matched_gt = gt
            
            if is_exact_match:
                exact_matches += 1
            
            # Count matches for each individual label
            if matched_gt:
                for col in available_labels:
                    if row_labels.get(col) == matched_gt.get(col):
                        label_matches[col] += 1
        
        exact_match_percentage = (exact_matches / total_rows * 100) if total_rows > 0 else 0.0
        
        # Calculate percentage for each label
        result = {
            'method': method,
            'total_rows': total_rows,
            'matches': exact_matches,
            'percentage': exact_match_percentage
        }
        
        # Add individual label percentages
        for col in label_columns:
            if col in available_labels:
                label_pct = (label_matches[col] / total_rows * 100) if total_rows > 0 else 0.0
                result[f'{col}_percentage'] = label_pct
            else:
                result[f'{col}_percentage'] = 0.0
        
        method_results.append(result)
    
    return pd.DataFrame(method_results)


def analyze_combo(
    combo_df: pd.DataFrame,
    gene_name: str,
    ligands1: List[str],
    ligands2: List[str],
    pair_type: str,
    cofolding_results_dir: Path
) -> pd.DataFrame:
    """
    Analyze a unique combo.
    
    Args:
        combo_df: DataFrame with rows for this unique combo
        gene_name: Gene name
        ligands1: Normalized ligands for structure1
        ligands2: Normalized ligands for structure2
        pair_type: 'holo_apo' or 'holo_holo'
        cofolding_results_dir: Directory containing cofolding_results
        
    Returns:
        DataFrame with analysis results
    """
    # Construct combo name
    combo_name = construct_combo_name(ligands1, ligands2, pair_type)
    logging.info(f"  Analyzing combo: {combo_name}")
    
    # Get ground truth labels for both structures
    gt_struc1 = get_ground_truth_labels(combo_df, 1)
    gt_struc2 = get_ground_truth_labels(combo_df, 2)
    
    # Construct analysis CSV filenames
    lig1_str = "_".join(ligands1) if ligands1 else "apo"
    lig2_str = "_".join(ligands2) if ligands2 else "apo"
    
    csv1_name = f"{gene_name}_{lig1_str}_analysis.csv" if ligands1 else f"{gene_name}_apo_analysis.csv"
    csv2_name = f"{gene_name}_{lig2_str}_analysis.csv" if ligands2 else f"{gene_name}_apo_analysis.csv"
    
    # Load analysis CSVs
    analysis_csvs_dir = cofolding_results_dir / gene_name / "analysis_csvs"
    
    # Check if both required CSV files exist
    csv1_path = analysis_csvs_dir / csv1_name
    csv2_path = analysis_csvs_dir / csv2_name
    
    if not csv1_path.exists() or not csv2_path.exists():
        # Skip if either required CSV file is missing
        missing = []
        if not csv1_path.exists():
            missing.append(csv1_name)
        if not csv2_path.exists():
            missing.append(csv2_name)
        logging.debug(f"    Skipping combo: missing analysis CSVs {missing}")
        return pd.DataFrame()
    
    results = []
    
    # Analysis 1: struc1 inference condition
    if csv1_path.exists():
        analysis_df1 = pd.read_csv(csv1_path)
        
        # Count matches by method for both structures
        struc1_method_results = count_exact_matches_by_method(analysis_df1, gt_struc1)
        struc2_method_results = count_exact_matches_by_method(analysis_df1, gt_struc2)
        
        # Determine classification
        if pair_type == 'holo_apo':
            if not ligands1:  # struc1 is apo
                classification = "apo_inference_holo"
            else:  # struc1 is holo
                classification = "holo_inference_apo"
        else:  # holo_holo
            classification = "holo_holo"
        
        # Merge results by method
        label_columns = [
            'activity_label', 'spatial_label', 'dihedral_label', 'chelix_label',
            'saltbr_label', 'actloop_nt_label', 'actloop_ct_label', 'spine_label'
        ]
        label_perc_cols = [f'{col}_percentage' for col in label_columns]
        
        if not struc1_method_results.empty and not struc2_method_results.empty:
            # Prepare struc1 columns
            struc1_cols = ['method', 'matches', 'percentage'] + label_perc_cols
            struc1_cols = [col for col in struc1_cols if col in struc1_method_results.columns]
            struc1_renamed = {col: f'struc1_{col}' if col not in ['method'] else col 
                            for col in struc1_cols if col != 'method'}
            struc1_renamed['matches'] = 'struc1_match'
            struc1_renamed['percentage'] = 'struc1_percentage'
            for col in label_perc_cols:
                if col in struc1_renamed:
                    struc1_renamed[col] = f'struc1_{col}'
            
            # Prepare struc2 columns
            struc2_cols = ['method', 'matches', 'percentage'] + label_perc_cols
            struc2_cols = [col for col in struc2_cols if col in struc2_method_results.columns]
            struc2_renamed = {col: f'struc2_{col}' if col not in ['method'] else col 
                            for col in struc2_cols if col != 'method'}
            struc2_renamed['matches'] = 'struc2_match'
            struc2_renamed['percentage'] = 'struc2_percentage'
            for col in label_perc_cols:
                if col in struc2_renamed:
                    struc2_renamed[col] = f'struc2_{col}'
            
            merged = pd.merge(
                struc1_method_results[struc1_cols].rename(columns=struc1_renamed),
                struc2_method_results[struc2_cols].rename(columns=struc2_renamed),
                on='method',
                how='outer'
            ).fillna(0)
            
            # Add metadata columns
            merged['canonical_gene_name'] = gene_name
            merged['struc1_condition'] = lig1_str
            merged['struc2_condition'] = lig2_str
            merged['inference_condition'] = lig1_str
            merged['classification'] = classification
            
            # Reorder columns
            base_cols = ['canonical_gene_name', 'struc1_condition', 'struc2_condition',
                        'inference_condition', 'method', 'struc1_match', 'struc1_percentage',
                        'struc2_match', 'struc2_percentage']
            struc1_label_cols = [f'struc1_{col}_percentage' for col in label_columns]
            struc2_label_cols = [f'struc2_{col}_percentage' for col in label_columns]
            final_cols = base_cols + struc1_label_cols + struc2_label_cols + ['classification']
            merged = merged[[col for col in final_cols if col in merged.columns]]
            
            results.append(merged)
        elif not struc1_method_results.empty:
            # Only struc1 results
            merged = struc1_method_results.copy()
            merged['canonical_gene_name'] = gene_name
            merged['struc1_condition'] = lig1_str
            merged['struc2_condition'] = lig2_str
            merged['inference_condition'] = lig1_str
            merged['struc1_match'] = merged['matches']
            merged['struc1_percentage'] = merged['percentage']
            # Rename label percentage columns
            for col in label_perc_cols:
                if col in merged.columns:
                    merged[f'struc1_{col}'] = merged[col]
                    merged = merged.drop(columns=[col])
            # Add zero columns for struc2
            merged['struc2_match'] = 0
            merged['struc2_percentage'] = 0.0
            for col in label_perc_cols:
                merged[f'struc2_{col}'] = 0.0
            merged['classification'] = classification
            
            # Reorder columns
            base_cols = ['canonical_gene_name', 'struc1_condition', 'struc2_condition',
                        'inference_condition', 'method', 'struc1_match', 'struc1_percentage',
                        'struc2_match', 'struc2_percentage']
            struc1_label_cols = [f'struc1_{col}_percentage' for col in label_columns]
            struc2_label_cols = [f'struc2_{col}_percentage' for col in label_columns]
            final_cols = base_cols + struc1_label_cols + struc2_label_cols + ['classification']
            merged = merged[[col for col in final_cols if col in merged.columns]]
            
            results.append(merged)
    
    # Analysis 2: struc2 inference condition
    if csv2_path.exists():
        analysis_df2 = pd.read_csv(csv2_path)
        
        # Count matches by method for both structures
        struc1_method_results = count_exact_matches_by_method(analysis_df2, gt_struc1)
        struc2_method_results = count_exact_matches_by_method(analysis_df2, gt_struc2)
        
        # Determine classification
        if pair_type == 'holo_apo':
            if not ligands2:  # struc2 is apo
                classification = "apo_inference_holo"
            else:  # struc2 is holo
                classification = "holo_inference_apo"
        else:  # holo_holo
            classification = "holo_holo"
        
        # Merge results by method
        label_columns = [
            'activity_label', 'spatial_label', 'dihedral_label', 'chelix_label',
            'saltbr_label', 'actloop_nt_label', 'actloop_ct_label', 'spine_label'
        ]
        label_perc_cols = [f'{col}_percentage' for col in label_columns]
        
        if not struc1_method_results.empty and not struc2_method_results.empty:
            # Prepare struc1 columns
            struc1_cols = ['method', 'matches', 'percentage'] + label_perc_cols
            struc1_cols = [col for col in struc1_cols if col in struc1_method_results.columns]
            struc1_renamed = {col: f'struc1_{col}' if col not in ['method'] else col 
                            for col in struc1_cols if col != 'method'}
            struc1_renamed['matches'] = 'struc1_match'
            struc1_renamed['percentage'] = 'struc1_percentage'
            for col in label_perc_cols:
                if col in struc1_renamed:
                    struc1_renamed[col] = f'struc1_{col}'
            
            # Prepare struc2 columns
            struc2_cols = ['method', 'matches', 'percentage'] + label_perc_cols
            struc2_cols = [col for col in struc2_cols if col in struc2_method_results.columns]
            struc2_renamed = {col: f'struc2_{col}' if col not in ['method'] else col 
                            for col in struc2_cols if col != 'method'}
            struc2_renamed['matches'] = 'struc2_match'
            struc2_renamed['percentage'] = 'struc2_percentage'
            for col in label_perc_cols:
                if col in struc2_renamed:
                    struc2_renamed[col] = f'struc2_{col}'
            
            merged = pd.merge(
                struc1_method_results[struc1_cols].rename(columns=struc1_renamed),
                struc2_method_results[struc2_cols].rename(columns=struc2_renamed),
                on='method',
                how='outer'
            ).fillna(0)
            
            # Add metadata columns
            merged['canonical_gene_name'] = gene_name
            merged['struc1_condition'] = lig1_str
            merged['struc2_condition'] = lig2_str
            merged['inference_condition'] = lig2_str
            merged['classification'] = classification
            
            # Reorder columns
            base_cols = ['canonical_gene_name', 'struc1_condition', 'struc2_condition',
                        'inference_condition', 'method', 'struc1_match', 'struc1_percentage',
                        'struc2_match', 'struc2_percentage']
            struc1_label_cols = [f'struc1_{col}_percentage' for col in label_columns]
            struc2_label_cols = [f'struc2_{col}_percentage' for col in label_columns]
            final_cols = base_cols + struc1_label_cols + struc2_label_cols + ['classification']
            merged = merged[[col for col in final_cols if col in merged.columns]]
            
            results.append(merged)
        elif not struc2_method_results.empty:
            # Only struc2 results
            merged = struc2_method_results.copy()
            merged['canonical_gene_name'] = gene_name
            merged['struc1_condition'] = lig1_str
            merged['struc2_condition'] = lig2_str
            merged['inference_condition'] = lig2_str
            # Add zero columns for struc1
            merged['struc1_match'] = 0
            merged['struc1_percentage'] = 0.0
            for col in label_perc_cols:
                merged[f'struc1_{col}'] = 0.0
            merged['struc2_match'] = merged['matches']
            merged['struc2_percentage'] = merged['percentage']
            # Rename label percentage columns
            for col in label_perc_cols:
                if col in merged.columns:
                    merged[f'struc2_{col}'] = merged[col]
                    merged = merged.drop(columns=[col])
            merged['classification'] = classification
            
            # Reorder columns
            base_cols = ['canonical_gene_name', 'struc1_condition', 'struc2_condition',
                        'inference_condition', 'method', 'struc1_match', 'struc1_percentage',
                        'struc2_match', 'struc2_percentage']
            struc1_label_cols = [f'struc1_{col}_percentage' for col in label_columns]
            struc2_label_cols = [f'struc2_{col}_percentage' for col in label_columns]
            final_cols = base_cols + struc1_label_cols + struc2_label_cols + ['classification']
            merged = merged[[col for col in final_cols if col in merged.columns]]
            
            results.append(merged)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()


def process_gene(
    gene_name: str,
    pairs_csv: Path,
    pair_type: str,
    cofolding_results_dir: Path,
    results_base_dir: Path,
    analysis_folder: str
) -> Dict:
    """
    Process all unique combos for a gene.
    
    Args:
        gene_name: Gene name
        pairs_csv: Path to pairs CSV file
        pair_type: 'holo_apo' or 'holo_holo'
        cofolding_results_dir: Directory containing gene/analysis_csvs (kinconfbench)
        results_base_dir: Base results dir (holo_apo_results or holo_holo_results)
        analysis_folder: Output subfolder (cofolding_analysis, cofolding_analysis_pre_training, etc.)
    """
    logging.info(f"\nProcessing {gene_name} ({pair_type})...")
    
    try:
        pairs_df = pd.read_csv(pairs_csv)
    except Exception as e:
        logging.error(f"  Error reading pairs CSV: {e}")
        return {'gene': gene_name, 'combos_processed': 0, 'errors': 1}
    
    if pairs_df.empty:
        logging.warning(f"  Empty pairs CSV, skipping")
        return {'gene': gene_name, 'combos_processed': 0, 'errors': 0}
    
    pairs_df['normalized_structure1_ligands'] = pairs_df['structure1_ligands'].apply(
        lambda x: tuple(normalize_ligands(x))
    )
    pairs_df['normalized_structure2_ligands'] = pairs_df['structure2_ligands'].apply(
        lambda x: tuple(normalize_ligands(x))
    )
    unique_combos = pairs_df.groupby([
        'structure1_structural_label',
        'structure2_structural_label',
        'normalized_structure1_ligands',
        'normalized_structure2_ligands'
    ])
    logging.info(f"  Found {len(unique_combos)} unique combinations")
    
    gene_output_dir = results_base_dir / gene_name / analysis_folder
    gene_output_dir.mkdir(parents=True, exist_ok=True)
    
    combos_processed = 0
    combos_skipped = 0
    
    for (struc1_label, struc2_label, lig1_tuple, lig2_tuple), combo_df in unique_combos:
        ligands1 = list(lig1_tuple)
        ligands2 = list(lig2_tuple)
        
        try:
            results_df = analyze_combo(
                combo_df, gene_name, ligands1, ligands2, pair_type, cofolding_results_dir
            )
            
            if not results_df.empty:
                combo_name = construct_combo_name(ligands1, ligands2, pair_type)
                output_csv = gene_output_dir / f"{combo_name}.csv"
                results_df.to_csv(output_csv, index=False)
                logging.info(f"    Saved results to {combo_name}.csv")
                combos_processed += 1
            else:
                combos_skipped += 1
        except Exception as e:
            logging.error(f"    Error processing combo: {e}")
            combos_skipped += 1
    
    logging.info(f"  Processed {combos_processed} combos, skipped {combos_skipped}")
    
    return {
        'gene': gene_name,
        'combos_processed': combos_processed,
        'combos_skipped': combos_skipped,
        'errors': 0
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cofolding inference results for unique pair combinations. "
                    "Works with kinconfbench or cofolding_results (same structure: gene/analysis_csvs/)."
    )
    parser.add_argument(
        "--holo_apo_results",
        type=str,
        default=None,
        help="Directory containing holo_apo_results (gene subdirs with holo_apo_pairs_by_labels.csv)"
    )
    parser.add_argument(
        "--holo_holo_results",
        type=str,
        default=None,
        help="Directory containing holo_holo_results (optional)"
    )
    parser.add_argument(
        "--cofolding_results_dir",
        type=str,
        required=True,
        help="Directory containing gene/analysis_csvs (kinconfbench or cofolding_results)"
    )
    parser.add_argument(
        "--analysis_folder",
        type=str,
        default=None,
        help="Output subfolder: cofolding_analysis, cofolding_analysis_pre_training, or cofolding_analysis_post_training (auto-detected from path if contains 'pre_training' or 'post_training')"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    script_dir = Path(__file__).parent
    cofolding_results_dir = Path(args.cofolding_results_dir)

    def _analysis_folder_from_path(p: Path) -> str:
        s = str(p)
        if "post_training" in s:
            return "cofolding_analysis_post_training"
        if "pre_training" in s:
            return "cofolding_analysis_pre_training"
        return "cofolding_analysis"

    analysis_folder = args.analysis_folder or _analysis_folder_from_path(cofolding_results_dir)

    if not args.holo_apo_results and not args.holo_holo_results:
        logging.error("At least one of --holo_apo_results or --holo_holo_results is required")
        return
    
    logging.info("=" * 80)
    logging.info("ANALYZING COFOLDING RESULTS")
    logging.info("=" * 80)
    logging.info(f"Cofolding results directory: {cofolding_results_dir}")
    logging.info(f"Analysis folder: {analysis_folder}")
    
    all_stats = []
    
    if args.holo_apo_results:
        holo_apo_dir = Path(args.holo_apo_results)
        if not holo_apo_dir.is_absolute():
            holo_apo_dir = script_dir / holo_apo_dir
        if holo_apo_dir.exists():
            logging.info(f"\n{'='*80}")
            logging.info("Processing holo_apo_results")
            logging.info(f"{'='*80}")
            for kinase_dir in sorted(holo_apo_dir.iterdir()):
                if not kinase_dir.is_dir():
                    continue
                gene_name = kinase_dir.name
                csv_path = kinase_dir / "holo_apo_pairs_by_labels.csv"
                if csv_path.exists():
                    stats = process_gene(gene_name, csv_path, 'holo_apo', cofolding_results_dir, holo_apo_dir, analysis_folder)
                    all_stats.append(stats)
    
    if args.holo_holo_results:
        holo_holo_dir = Path(args.holo_holo_results)
        if not holo_holo_dir.is_absolute():
            holo_holo_dir = script_dir / holo_holo_dir
        if holo_holo_dir.exists():
            logging.info(f"\n{'='*80}")
            logging.info("Processing holo_holo_results")
            logging.info(f"{'='*80}")
            for kinase_dir in sorted(holo_holo_dir.iterdir()):
                if not kinase_dir.is_dir():
                    continue
                gene_name = kinase_dir.name
                csv_path = kinase_dir / "holo_holo_pairs_by_labels.csv"
                if csv_path.exists():
                    stats = process_gene(gene_name, csv_path, 'holo_holo', cofolding_results_dir, holo_holo_dir, analysis_folder)
                    all_stats.append(stats)
    
    # Summary
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        logging.info(f"\n{'='*80}")
        logging.info("SUMMARY")
        logging.info(f"{'='*80}")
        logging.info(f"Total genes processed: {len(stats_df)}")
        logging.info(f"Total combos processed: {stats_df['combos_processed'].sum()}")
        logging.info(f"Total combos skipped: {stats_df['combos_skipped'].sum()}")
        logging.info(f"Errors: {stats_df['errors'].sum()}")


if __name__ == "__main__":
    main()

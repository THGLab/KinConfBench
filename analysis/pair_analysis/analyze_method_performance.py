#!/usr/bin/env python3
"""
Analyze cofolding method performance by classification.

Aggregates results from cofolding_analysis folders (or cofolding_analysis_pre_training,
cofolding_analysis_post_training) and provides summary statistics.

Usage:
    # Single run (e.g. pre-training only):
    python analyze_method_performance.py --results_dir /path/to/holo_apo_results \\
        --cofolding_results_dir /path/to/cofolding_results_pre_training

    # Seen/Unseen (pre + post) for discovery success comparison:
    python analyze_method_performance.py --results_dir /path/to/holo_apo_results \\
        --cofolding_results_dir_pre /path/to/cofolding_results_pre_training \\
        --cofolding_results_dir_post /path/to/cofolding_results_post_training --create_plots
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, Set, Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def remap_struc_to_reference_linked(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remap struc1/struc2 columns to reference/linked based on inference_condition.
    
    Reference = structure matching inference_condition
    Linked = structure NOT matching inference_condition
    
    Args:
        df: DataFrame with struc1_* and struc2_* columns
        
    Returns:
        DataFrame with reference_* and linked_* columns
    """
    df = df.copy()
    
    # Find all struc1 and struc2 columns (excluding condition columns which we'll handle separately)
    struc1_cols = [col for col in df.columns if col.startswith('struc1_') and col != 'struc1_condition']
    struc2_cols = [col for col in df.columns if col.startswith('struc2_') and col != 'struc2_condition']
    
    # Determine all new column names we'll need
    # For each struc1/struc2 column, we need both reference_ and linked_ versions
    # Example: struc1_match -> reference_match AND linked_match
    new_column_names = set()
    for col in struc1_cols + struc2_cols:
        # Add reference_ version
        new_column_names.add(col.replace('struc1_', 'reference_').replace('struc2_', 'reference_'))
        # Add linked_ version
        new_column_names.add(col.replace('struc1_', 'linked_').replace('struc2_', 'linked_'))
    
    # Initialize all new columns with None (will be filled in row by row)
    for new_col in new_column_names:
        df[new_col] = None
    
    # Map values row by row based on which structure matches inference_condition
    for idx, row in df.iterrows():
        inference_condition = str(row['inference_condition']).strip()
        struc1_condition = str(row['struc1_condition']).strip()
        struc2_condition = str(row['struc2_condition']).strip()
        
        # Determine which structure is the reference (matches inference_condition)
        if inference_condition == struc1_condition:
            # struc1 is reference, struc2 is linked
            for col in struc1_cols:
                new_col = col.replace('struc1_', 'reference_')
                df.at[idx, new_col] = row[col]
            for col in struc2_cols:
                new_col = col.replace('struc2_', 'linked_')
                df.at[idx, new_col] = row[col]
            df.at[idx, 'reference_condition'] = struc1_condition
            df.at[idx, 'linked_condition'] = struc2_condition
        elif inference_condition == struc2_condition:
            # struc2 is reference, struc1 is linked
            for col in struc2_cols:
                new_col = col.replace('struc2_', 'reference_')
                df.at[idx, new_col] = row[col]
            for col in struc1_cols:
                new_col = col.replace('struc1_', 'linked_')
                df.at[idx, new_col] = row[col]
            df.at[idx, 'reference_condition'] = struc2_condition
            df.at[idx, 'linked_condition'] = struc1_condition
        else:
            logging.warning(f"Row {idx}: inference_condition '{inference_condition}' doesn't match struc1 '{struc1_condition}' or struc2 '{struc2_condition}'")
    
    # Drop old struc1 and struc2 columns
    cols_to_drop = struc1_cols + struc2_cols + ['struc1_condition', 'struc2_condition']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df


def normalize_ligands(ligands_str: str) -> list:
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


def construct_combo_name(ligands1: list, ligands2: list, pair_type: str) -> str:
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


def get_valid_combo_names(
    results_base_dir: Path,
    cofolding_results_dir: Optional[Path]
) -> Set[Tuple[str, str]]:
    """
    Get set of valid (gene_name, combo_name) tuples based on available structures.
    
    Args:
        results_base_dir: Base directory containing holo_apo_results or holo_holo_results
        cofolding_results_dir: Directory containing cofolding_results (if None, no filtering)
        
    Returns:
        Set of (gene_name, combo_name) tuples
    """
    if cofolding_results_dir is None:
        return set()  # No filtering
    
    valid_combos = set()
    
    # Determine pair type from results_base_dir name
    pair_type = None
    if 'holo_apo' in str(results_base_dir):
        pair_type = 'holo_apo'
    elif 'holo_holo' in str(results_base_dir):
        pair_type = 'holo_holo'
    else:
        logging.warning(f"Could not determine pair type from {results_base_dir}")
        return set()
    
    logging.info(f"Building valid combo names from {cofolding_results_dir} (pair_type: {pair_type})...")
    
    # Iterate through gene directories
    for gene_dir in sorted(results_base_dir.iterdir()):
        if not gene_dir.is_dir():
            continue
        
        gene_name = gene_dir.name
        
        # Check if pairs CSV exists
        pairs_csv = gene_dir / f"{pair_type}_pairs_by_labels.csv"
        if not pairs_csv.exists():
            continue
        
        # Check if summary CSV exists in cofolding_results
        summary_csv = cofolding_results_dir / gene_name / f"{gene_name}_summary.csv"
        if not summary_csv.exists():
            logging.debug(f"  {gene_name}: Summary CSV not found, skipping")
            continue
        
        try:
            # Load pairs
            pairs_df = pd.read_csv(pairs_csv)
            if pairs_df.empty:
                continue
            
            # Load summary to get available structure IDs
            summary_df = pd.read_csv(summary_csv)
            available_structure_ids = set(summary_df['structure_id'].unique())
            
            # Filter pairs where both structures exist
            valid_pairs = pairs_df[
                (pairs_df['structure1_id'].isin(available_structure_ids)) &
                (pairs_df['structure2_id'].isin(available_structure_ids))
            ].copy()
            
            if valid_pairs.empty:
                logging.debug(f"  {gene_name}: No valid pairs found")
                continue
            
            # Build combo names from valid pairs
            for _, row in valid_pairs.iterrows():
                ligands1 = normalize_ligands(row['structure1_ligands'])
                ligands2 = normalize_ligands(row['structure2_ligands'])
                combo_name = construct_combo_name(ligands1, ligands2, pair_type)
                valid_combos.add((gene_name, combo_name))
            
            logging.debug(f"  {gene_name}: {len(valid_pairs)} valid pairs, {len([c for g, c in valid_combos if g == gene_name])} unique combos")
            
        except Exception as e:
            logging.warning(f"  {gene_name}: Error processing: {e}")
            continue
    
    logging.info(f"Found {len(valid_combos)} valid combo names")
    return valid_combos


def collect_all_results(
    results_base_dir: Path,
    cofolding_results_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Collect all CSV files from cofolding_analysis directories.
    
    Args:
        results_base_dir: Base directory containing holo_apo_results or holo_holo_results
        cofolding_results_dir: Directory containing cofolding_results for filtering (optional)
        
    Returns:
        Combined DataFrame with all results (with reference/linked columns)
    """
    all_dfs = []
    
    # Get valid combo names if filtering is requested
    valid_combos = None
    if cofolding_results_dir is not None:
        valid_combos = get_valid_combo_names(results_base_dir, cofolding_results_dir)
    
    # Determine analysis folder name based on cofolding_results_dir
    analysis_folder = "cofolding_analysis"
    if cofolding_results_dir and "post_training" in str(cofolding_results_dir):
        analysis_folder = "cofolding_analysis_post_training"
    elif cofolding_results_dir and "pre_training" in str(cofolding_results_dir):
        analysis_folder = "cofolding_analysis_pre_training"
    
    # Find all cofolding_analysis directories
    for gene_dir in sorted(results_base_dir.iterdir()):
        if not gene_dir.is_dir():
            continue
        
        cofolding_analysis_dir = gene_dir / analysis_folder
        if not cofolding_analysis_dir.exists():
            continue
        
        gene_name = gene_dir.name
        logging.info(f"Processing {gene_name}...")
        
        # Read all CSV files in cofolding_analysis
        csv_files = list(cofolding_analysis_dir.glob("*.csv"))
        for csv_file in csv_files:
            # Extract combo name from filename (without .csv extension)
            combo_name = csv_file.stem
            
            # Filter by valid combos if filtering is requested
            if valid_combos is not None:
                if (gene_name, combo_name) not in valid_combos:
                    logging.debug(f"  Skipping {combo_name} (not in valid combos)")
                    continue
            
            try:
                # Read CSV with condition columns as strings to preserve leading zeros
                df = pd.read_csv(csv_file, dtype={
                    'struc1_condition': str,
                    'struc2_condition': str,
                    'inference_condition': str
                })
                if not df.empty:
                    # Remap struc1/struc2 to reference/linked
                    df = remap_struc_to_reference_linked(df)
                    all_dfs.append(df)
            except Exception as e:
                logging.warning(f"  Error reading {csv_file}: {e}")
    
    if not all_dfs:
        logging.warning("No results found!")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Collected {len(combined_df)} total rows from {len(all_dfs)} files")
    
    return combined_df


def calculate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics grouped by method, classification, and label.
    
    Args:
        df: Combined results DataFrame
        
    Returns:
        DataFrame with label-specific statistics, including 'all_labels' for overall match
    """
    label_cols = [
        'activity_label', 'spatial_label', 'dihedral_label', 'chelix_label',
        'saltbr_label', 'actloop_nt_label', 'actloop_ct_label', 'spine_label',
        'all_labels'  # Add all_labels to treat reference_percentage/linked_percentage as a label
    ]
    
    label_stats = []
    for method in df['method'].unique():
        for classification in df['classification'].unique():
            subset = df[(df['method'] == method) & (df['classification'] == classification)]
            if subset.empty:
                continue
            
            for label in label_cols:
                if label == 'all_labels':
                    # Use reference_percentage and linked_percentage as the "all_labels" label
                    reference_col = 'reference_percentage'
                    linked_col = 'linked_percentage'
                else:
                    reference_col = f'reference_{label}_percentage'
                    linked_col = f'linked_{label}_percentage'
                
                if reference_col in subset.columns:
                    label_stats.append({
                        'method': method,
                        'classification': classification,
                        'label': label,
                        'structure': 'reference',
                        'mean': subset[reference_col].mean(),
                        'median': subset[reference_col].median(),
                        'std': subset[reference_col].std(),
                        'n_cases': len(subset)
                    })
                
                if linked_col in subset.columns:
                    label_stats.append({
                        'method': method,
                        'classification': classification,
                        'label': label,
                        'structure': 'linked',
                        'mean': subset[linked_col].mean(),
                        'median': subset[linked_col].median(),
                        'std': subset[linked_col].std(),
                        'n_cases': len(subset)
                    })
    
    return pd.DataFrame(label_stats)


def _bootstrap_mean_ci(values: List[float], n_bootstrap: int = 1000, ci_alpha: float = 0.05) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for the mean. Returns (mean, ci_lower, ci_upper)."""
    if not values:
        return 0.0, 0.0, 0.0
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        sample = random.choices(values, k=n)
        means.append(np.mean(sample))
    means.sort()
    mean = np.mean(values)
    lower_idx = int((ci_alpha / 2) * n_bootstrap)
    upper_idx = int((1 - ci_alpha / 2) * n_bootstrap)
    ci_lower = means[lower_idx] if lower_idx < n_bootstrap else means[0]
    ci_upper = means[upper_idx] if upper_idx < n_bootstrap else means[-1]
    return mean, ci_lower, ci_upper


def build_paired_comparison_data(
    df: pd.DataFrame,
    figures_dir: Path,
    n_bootstrap: int = 1000,
) -> Optional[pd.DataFrame]:
    """
    Build raw and summary (with bootstrap CI) data for the paired comparison plot.
    Saves paired_comparison_raw.csv and paired_comparison_summary.csv to figures_dir.
    Returns summary DataFrame (method, split, state, mean, ci_lower, ci_upper, n).
    """
    if 'split' not in df.columns or 'reference_percentage' not in df.columns or 'linked_percentage' not in df.columns:
        return None
    methods = sorted(df['method'].unique())
    raw_rows = []
    for _, row in df.iterrows():
        method = row['method']
        split_val = row['split']
        if row['classification'] == 'holo_inference_apo':
            holo_pct = row['reference_percentage']
            apo_pct = row['linked_percentage']
        else:
            holo_pct = row['linked_percentage']
            apo_pct = row['reference_percentage']
        raw_rows.append({'method': method, 'split': split_val, 'state': 'Holo', 'percentage': float(holo_pct)})
        raw_rows.append({'method': method, 'split': split_val, 'state': 'Apo', 'percentage': float(apo_pct)})
    raw_df = pd.DataFrame(raw_rows)
    figures_dir.mkdir(parents=True, exist_ok=True)
    raw_path = figures_dir / 'paired_comparison_raw.csv'
    raw_df.to_csv(raw_path, index=False)
    logging.info(f"Saved: {raw_path}")

    summary_rows = []
    for method in methods:
        for split_val in ['Seen', 'Unseen']:
            for state in ['Holo', 'Apo']:
                sub = raw_df[(raw_df['method'] == method) & (raw_df['split'] == split_val) & (raw_df['state'] == state)]
                vals = sub['percentage'].dropna().tolist()
                mean, ci_lower, ci_upper = _bootstrap_mean_ci(vals, n_bootstrap=n_bootstrap)
                summary_rows.append({
                    'method': method.upper(),
                    'split': split_val,
                    'state': state,
                    'mean': mean,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'n': len(vals),
                })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = figures_dir / 'paired_comparison_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"Saved: {summary_path}")
    return summary_df


def _save_plot_to_both(plt_handle, output_dir: Path, figures_dir: Path, base_name: str):
    """Save current figure as PNG and SVG to both output_dir and figures_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        path_out = output_dir / f"{base_name}.{ext}"
        path_fig = figures_dir / f"{base_name}.{ext}"
        plt_handle.savefig(path_out, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        plt_handle.savefig(path_fig, dpi=300 if ext == 'png' else None, bbox_inches='tight')
    logging.info(f"Saved: {base_name}.png, {base_name}.svg (current dir + figures/)")


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """
    Create visualization plots for method performance.
    Saves to both output_dir and output_dir/figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Overall percentage comparison by method and classification
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, col in enumerate(['reference_percentage', 'linked_percentage']):
        if col not in df.columns:
            continue
        ax = axes[idx]
        sns.boxplot(data=df, x='method', y=col, hue='classification', ax=ax)
        ax.set_title(f'{col.replace("_", " ").title()}')
        ax.set_xlabel('Method')
        ax.set_ylabel('Percentage (%)')
        ax.legend(title='Classification')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_plot_to_both(plt, output_dir, figures_dir, 'overall_percentage_comparison')
    plt.close()
    
    # 2. Individual label percentages (heatmap)
    label_cols = [
        'activity_label', 'spatial_label', 'dihedral_label', 'chelix_label',
        'saltbr_label', 'actloop_nt_label', 'actloop_ct_label', 'spine_label'
    ]
    
    for classification in df['classification'].unique():
        subset = df[df['classification'] == classification]
        
        # Prepare data for heatmap
        heatmap_data = []
        for method in subset['method'].unique():
            method_subset = subset[subset['method'] == method]
            row = {'method': method}
            for label in label_cols:
                reference_col = f'reference_{label}_percentage'
                linked_col = f'linked_{label}_percentage'
                if reference_col in method_subset.columns and linked_col in method_subset.columns:
                    # Average of reference and linked
                    row[label] = (method_subset[reference_col].mean() + method_subset[linked_col].mean()) / 2
            heatmap_data.append(row)
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data).set_index('method')
            
            plt.figure(figsize=(12, 4))
            sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd', 
                       vmin=0, vmax=100, cbar_kws={'label': 'Mean Percentage (%)'})
            plt.title(f'Label Match Percentage by Method - {classification}')
            plt.ylabel('Method')
            plt.xlabel('Label')
            plt.tight_layout()
            _save_plot_to_both(plt, output_dir, figures_dir, f'label_heatmap_{classification}')
            plt.close()
    
    # 3. Paired Comparison plot (Apo-Drift narrative): Holo vs Apo, Seen vs Unseen (with bootstrap CI)
    if 'split' in df.columns and 'reference_percentage' in df.columns and 'linked_percentage' in df.columns:
        summary_path = figures_dir / 'paired_comparison_summary.csv'
        if summary_path.exists():
            pair_df = pd.read_csv(summary_path)
            logging.info(f"Loaded paired comparison summary from {summary_path}")
        else:
            pair_df = build_paired_comparison_data(df, figures_dir, n_bootstrap=1000)
        if pair_df is not None and not pair_df.empty:
            method_order = sorted(pair_df['method'].unique())
            n_methods = len(method_order)
            x = np.arange(n_methods)
            width = 0.2
            gap = 0.08
            # Seaborn 3-color palette; 4 series: use 3 colors (Holo-Seen, Holo-Unseen share hue; Apo-Seen, Apo-Unseen share hue; or 3 distinct)
            palette = sns.color_palette("Set2", 3)
            # Map 4 series to 3 colors: Holo-Seen, Holo-Unseen = first two; Apo-Seen, Apo-Unseen = second two (reuse first for 4th)
            series_colors = [palette[0], palette[1], palette[2], palette[0]]
            # Order: Holo-Seen, Holo-Unseen, Apo-Seen, Apo-Unseen
            series_order = [('Holo', 'Seen'), ('Holo', 'Unseen'), ('Apo', 'Seen'), ('Apo', 'Unseen')]
            fig, ax = plt.subplots(figsize=(14, 6))
            for series_idx, (state, split_val) in enumerate(series_order):
                color = series_colors[series_idx]
                means = []
                err_lo = []
                err_hi = []
                for method in method_order:
                    row = pair_df[(pair_df['method'] == method) & (pair_df['state'] == state) & (pair_df['split'] == split_val)]
                    if not row.empty:
                        r = row.iloc[0]
                        means.append(r['mean'])
                        err_lo.append(r['mean'] - r['ci_lower'])
                        err_hi.append(r['ci_upper'] - r['mean'])
                    else:
                        means.append(0)
                        err_lo.append(0)
                        err_hi.append(0)
                yerr = np.array([err_lo, err_hi])
                offset = series_idx * width - 1.5 * width
                if series_idx >= 2:
                    offset += gap
                bars = ax.bar(x + offset, means, width, label=f'{state}-{split_val}',
                              color=color, edgecolor='black', linewidth=0.6)
                ax.errorbar(x + offset, means, yerr=yerr, fmt='none', color='black', capsize=2, capthick=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(method_order)
            ax.set_ylabel('Mean match percentage (%)')
            ax.set_xlabel('Method')
            ax.set_title('Paired comparison: Reference (Holo) vs Linked (Apo), Seen vs Unseen (95% bootstrap CI)')
            ax.legend(loc='upper right', frameon=True)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(0, 105)
            plt.tight_layout()
            _save_plot_to_both(plt, output_dir, figures_dir, 'paired_comparison_holo_apo_seen_unseen')
            plt.close()
    
    # 4. Box plots for reference, linked, and difference (most important visualization)
    if 'reference_percentage' in df.columns and 'linked_percentage' in df.columns:
        # Calculate difference (reference - linked) for each row
        df_with_diff = df.copy()
        df_with_diff['reference_linked_difference'] = df_with_diff['reference_percentage'] - df_with_diff['linked_percentage']
        
        # Prepare data for box plots - reshape to long format
        plot_data = []
        for _, row in df_with_diff.iterrows():
            plot_data.append({
                'method': row['method'],
                'classification': row['classification'],
                'type': 'reference',
                'value': row['reference_percentage']
            })
            plot_data.append({
                'method': row['method'],
                'classification': row['classification'],
                'type': 'linked',
                'value': row['linked_percentage']
            })
            plot_data.append({
                'method': row['method'],
                'classification': row['classification'],
                'type': 'difference',
                'value': row['reference_linked_difference']
            })
        plot_df = pd.DataFrame(plot_data)
        
        # Create two plots: one for each classification
        classifications = ['apo_inference_holo', 'holo_inference_apo']
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        
        for cls_idx, classification in enumerate(classifications):
            ax = axes[cls_idx]
            cls_df = plot_df[plot_df['classification'] == classification]
            
            # Create box plot with seaborn (handles grouping better)
            colors = {'reference': '#3498db', 'linked': '#e74c3c', 'difference': '#2ecc71'}
            palette = [colors['reference'], colors['linked'], colors['difference']]
            sns.boxplot(data=cls_df, x='method', y='value', hue='type', ax=ax, palette=palette)
            
            # Add horizontal line at 0 for difference reference
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            
            ax.set_title(classification.replace('_', ' ').title())
            ax.set_ylabel('Percentage / Difference (%)')
            ax.set_xlabel('Method')
            ax.legend(title='Type', labels=['Reference', 'Linked', 'Difference'])
            ax.grid(True, alpha=0.3, axis='y')
            # Capitalize method names on x-axis
            labels = ax.get_xticklabels()
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([label.get_text().upper() for label in labels])
        
        plt.tight_layout()
        _save_plot_to_both(plt, output_dir, figures_dir, 'reference_linked_box_plots')
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cofolding method performance by classification"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Results directory (holo_apo_results or holo_holo_results)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for summary files (default: results_dir/method_performance_analysis)"
    )
    parser.add_argument(
        "--create_plots",
        action="store_true",
        help="Create visualization plots"
    )
    parser.add_argument(
        "--cofolding_results_dir",
        type=str,
        default=None,
        help="Directory containing cofolding_results for filtering (only include pairs where both structures exist)"
    )
    parser.add_argument(
        "--cofolding_results_dir_pre",
        type=str,
        default=None,
        help="Pre-training cofolding_results dir (use with --cofolding_results_dir_post for Seen/Unseen paired plot)"
    )
    parser.add_argument(
        "--cofolding_results_dir_post",
        type=str,
        default=None,
        help="Post-training cofolding_results dir (use with --cofolding_results_dir_pre for Seen/Unseen paired plot)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    script_dir = Path(__file__).parent
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = script_dir / results_dir
    
    if not results_dir.exists():
        logging.error(f"Results directory not found: {results_dir}")
        return
    
    # Determine cofolding_results directory (or pre/post for Seen/Unseen)
    cofolding_results_dir = None
    cofolding_results_dir_pre = None
    cofolding_results_dir_post = None
    if args.cofolding_results_dir_pre and args.cofolding_results_dir_post:
        cofolding_results_dir_pre = Path(args.cofolding_results_dir_pre)
        cofolding_results_dir_post = Path(args.cofolding_results_dir_post)
        if not cofolding_results_dir_pre.is_absolute():
            cofolding_results_dir_pre = script_dir / cofolding_results_dir_pre
        if not cofolding_results_dir_post.is_absolute():
            cofolding_results_dir_post = script_dir / cofolding_results_dir_post
        if not cofolding_results_dir_pre.exists():
            logging.warning(f"Pre-training dir not found: {cofolding_results_dir_pre}")
            cofolding_results_dir_pre = None
        if not cofolding_results_dir_post.exists():
            logging.warning(f"Post-training dir not found: {cofolding_results_dir_post}")
            cofolding_results_dir_post = None
    elif args.cofolding_results_dir:
        cofolding_results_dir = Path(args.cofolding_results_dir)
        if not cofolding_results_dir.is_absolute():
            cofolding_results_dir = script_dir / cofolding_results_dir
        if not cofolding_results_dir.exists():
            logging.warning(f"Cofolding results directory not found: {cofolding_results_dir}, proceeding without filtering")
            cofolding_results_dir = None
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        perf_analysis_folder = "method_performance_analysis"
        if cofolding_results_dir_pre is not None and cofolding_results_dir_post is not None:
            perf_analysis_folder = "method_performance_analysis_generalization"
        elif cofolding_results_dir and "post_training" in str(cofolding_results_dir):
            perf_analysis_folder = "method_performance_analysis_post_training"
        elif cofolding_results_dir and "pre_training" in str(cofolding_results_dir):
            perf_analysis_folder = "method_performance_analysis_pre_training"
        output_dir = results_dir / perf_analysis_folder
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 80)
    logging.info("ANALYZING METHOD PERFORMANCE")
    logging.info("=" * 80)
    logging.info(f"Results directory: {results_dir}")
    if cofolding_results_dir:
        logging.info(f"Cofolding results directory (for filtering): {cofolding_results_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    # Collect all results (optionally merge pre + post for Seen/Unseen)
    logging.info("\nCollecting all results...")
    if cofolding_results_dir_pre is not None and cofolding_results_dir_post is not None:
        df_pre = collect_all_results(results_dir, cofolding_results_dir_pre)
        df_post = collect_all_results(results_dir, cofolding_results_dir_post)
        if not df_pre.empty:
            df_pre = df_pre.copy()
            df_pre['split'] = 'Seen'
        if not df_post.empty:
            df_post = df_post.copy()
            df_post['split'] = 'Unseen'
        combined_df = pd.concat([df_pre, df_post], ignore_index=True)
        logging.info(f"Merged pre (Seen) + post (Unseen): {len(combined_df)} rows")
    else:
        combined_df = collect_all_results(results_dir, cofolding_results_dir)
    
    if combined_df.empty:
        logging.error("No results found!")
        return
    
    # Save raw combined data
    combined_df.to_csv(output_dir / "all_results_combined.csv", index=False)
    logging.info(f"Saved: all_results_combined.csv ({len(combined_df)} rows)")
    
    # Calculate summary statistics
    logging.info("\nCalculating summary statistics...")
    summary_df = calculate_summary_statistics(combined_df)
    
    # Save summary table
    summary_df.to_csv(output_dir / "summary_by_label.csv", index=False)
    logging.info(f"Saved: summary_by_label.csv")
    
    # Create visualizations if requested
    if args.create_plots:
        if not HAS_SEABORN:
            logging.warning("seaborn not installed, skipping plots (pip install seaborn)")
        else:
            logging.info("\nCreating visualizations...")
            try:
                create_visualizations(combined_df, output_dir)
            except Exception as e:
                logging.error(f"Error creating visualizations: {e}")
    
    logging.info("\n" + "=" * 80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

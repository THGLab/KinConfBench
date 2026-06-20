#!/usr/bin/env python3
"""
Merge newly generated MSA folders with existing MSA folders.
This script handles merging of protein_msa_metadata.pkl files and MSA directories.

Author: Oliver Sun
Date: 2025-01-12
"""

import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def load_pkl_file(pkl_path: str) -> Dict[str, Any]:
    """
    Load a pickle file and return its contents.
    
    Args:
        pkl_path: Path to the pickle file
        
    Returns:
        Dictionary containing the pickle file contents, empty dict if file doesn't exist
    """
    if not os.path.exists(pkl_path):
        return {}
    
    try:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file {pkl_path}: {e}")
        return {}


def save_pkl_file(data: Dict[str, Any], pkl_path: str) -> bool:
    """
    Save data to a pickle file.
    
    Args:
        data: Dictionary to save
        pkl_path: Path to save the pickle file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving pickle file {pkl_path}: {e}")
        return False


def compare_sequences(seq1: str, seq2: str) -> bool:
    """
    Compare two protein sequences for equality.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        True if sequences are identical, False otherwise
    """
    return seq1.strip().upper() == seq2.strip().upper()


def merge_msa_directories(source_dir: str, target_dir: str, protein_name: str, method: str) -> bool:
    """
    Merge MSA directories for a specific protein and method.
    
    Args:
        source_dir: Source MSA directory (newly generated)
        target_dir: Target MSA directory (existing)
        protein_name: Name of the protein
        method: MSA method (boltz, chai, protenix)
        
    Returns:
        True if successful, False otherwise
    """
    source_path = Path(source_dir) / method
    target_path = Path(target_dir) / method
    
    # Create target method directory if it doesn't exist
    os.makedirs(target_path, exist_ok=True)
    
    try:
        if method == 'boltz':
            # For Boltz, copy the CSV file
            source_file = source_path / f"{protein_name}.csv"
            target_file = target_path / f"{protein_name}.csv"
            
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                print(f"  Copied Boltz MSA: {protein_name}.csv")
                return True
            else:
                print(f"  Warning: Boltz MSA file not found: {source_file}")
                return False
                
        elif method in ['chai', 'protenix']:
            # For Chai and Protenix, copy the entire protein directory
            source_protein_dir = source_path / protein_name
            target_protein_dir = target_path / protein_name
            
            if source_protein_dir.exists():
                if target_protein_dir.exists():
                    shutil.rmtree(target_protein_dir)
                shutil.copytree(source_protein_dir, target_protein_dir)
                print(f"  Copied {method.capitalize()} MSA directory: {protein_name}/")
                return True
            else:
                print(f"  Warning: {method.capitalize()} MSA directory not found: {source_protein_dir}")
                return False
                
    except Exception as e:
        print(f"  Error copying {method} MSA for {protein_name}: {e}")
        return False
    
    return False


def merge_msa_folders(source_msa_dir: str, target_msa_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Merge MSA folders from source to target directory.
    
    Args:
        source_msa_dir: Path to newly generated MSA folder
        target_msa_dir: Path to existing MSA folder to merge into
        
    Returns:
        Tuple of (added_proteins, skipped_proteins, flagged_proteins)
    """
    source_path = Path(source_msa_dir)
    target_path = Path(target_msa_dir)
    
    # Load pickle files
    source_pkl_path = source_path.glob("*.pkl")[0] if source_path.glob("*.pkl") else None
    target_pkl_path = target_path.glob("*.pkl")[0] if target_path.glob("*.pkl") else None
    
    source_data = load_pkl_file(str(source_pkl_path))
    target_data = load_pkl_file(str(target_pkl_path))
    
    if not source_data:
        print(f"Error: No data found in source pickle file: {source_pkl_path}")
        return [], [], []
    
    print(f"Source MSA data contains {len(source_data)} proteins")
    print(f"Target MSA data contains {len(target_data)} proteins")
    
    # Create target directory if it doesn't exist
    os.makedirs(target_path, exist_ok=True)
    
    # If target pkl doesn't exist, create empty dict
    if not target_data:
        print(f"Target pickle file not found, will create new one at: {target_pkl_path}")
    
    # Track results
    added_proteins = []
    skipped_proteins = []
    flagged_proteins = []
    
    # Process each protein in source data
    for protein_name, protein_info in source_data.items():
        print(f"\nProcessing protein: {protein_name}")
        
        if protein_name in target_data:
            # Protein exists in target, check sequences
            source_seq = protein_info.get('sequence', '')
            target_seq = target_data[protein_name].get('sequence', '')
            
            if compare_sequences(source_seq, target_seq):
                print(f"  Sequences match - skipping {protein_name}")
                skipped_proteins.append(protein_name)
                continue
            else:
                print(f"  WARNING: Sequences differ for {protein_name}!")
                print(f"    Source seq length: {len(source_seq)}")
                print(f"    Target seq length: {len(target_seq)}")
                flagged_proteins.append(protein_name)
                # Still proceed with merge but flag it
        
        # Add/update protein in target data
        target_data[protein_name] = protein_info.copy()
        
        # Copy MSA files for each method
        methods_to_copy = []
        if 'boltz_msa' in protein_info:
            methods_to_copy.append('boltz')
        if 'chai_msa' in protein_info:
            methods_to_copy.append('chai')
        if 'protenix_msa' in protein_info:
            methods_to_copy.append('protenix')
        
        success_count = 0
        for method in methods_to_copy:
            if merge_msa_directories(str(source_path), str(target_path), protein_name, method):
                success_count += 1
        
        if success_count > 0:
            if protein_name not in flagged_proteins:
                added_proteins.append(protein_name)
            print(f"  Successfully merged {success_count}/{len(methods_to_copy)} MSA methods")
        else:
            print(f"  Failed to merge any MSA methods for {protein_name}")
    
    # Save updated target pickle file (append mode)
    if save_pkl_file(target_data, str(target_pkl_path)):
        print(f"\nAppended to target pickle file: {target_pkl_path}")
    else:
        print(f"\nError: Failed to append to target pickle file: {target_pkl_path}")
    
    return added_proteins, skipped_proteins, flagged_proteins


def main():
    """Main function for MSA folder merging."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge newly generated MSA folder with existing MSA folder.')
    parser.add_argument('source_msa_dir', help='Path to newly generated MSA folder')
    parser.add_argument('target_msa_dir', help='Path to existing MSA folder to merge into')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    source_dir = Path(args.source_msa_dir).resolve()
    target_dir = Path(args.target_msa_dir).resolve()
    
    print(f"Merging MSAs from: {source_dir}")
    print(f"Into existing folder: {target_dir}")
    
    # Perform merge
    added, skipped, flagged = merge_msa_folders(str(source_dir), str(target_dir))
    
    # Print summary
    print(f"\n=== Merge Summary ===")
    print(f"Added proteins: {len(added)}")
    if added:
        for protein in added:
            print(f"  + {protein}")
    
    print(f"\nSkipped proteins (identical sequences): {len(skipped)}")
    if skipped:
        for protein in skipped:
            print(f"  = {protein}")
    
    print(f"\nFlagged proteins (different sequences): {len(flagged)}")
    if flagged:
        for protein in flagged:
            print(f"  ! {protein}")
        print("\nWARNING: Some proteins have different sequences between source and target!")
        print("Please review these proteins manually.")
    
    print(f"\nMerge completed! Updated MSA folder: {target_dir}")


if __name__ == "__main__":
    main()

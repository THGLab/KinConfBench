import pandas as pd
import pickle
import argparse
import sys
import yaml
from pathlib import Path

def process_kinase_data(csv_file, proteins_path, msa_folder, ligands_path, output_path, config_output_path=None, skip_protein_check=False):
    """
    Process kinase data from CSV and generate combinations compatible with the config format.
    Automatically separates combinations into apo (no ligands) and holo (with ligands).
    
    Args:
        csv_file: Path to CSV file containing canonical_gene_name and ligands columns
        proteins_path: Path to proteins pkl file
        msa_folder: Path to MSA folder (for reference)
        ligands_path: Path to ligands pkl file
        output_path: Path to save the output pkl file
        config_output_path: Path to save the config.yml file (optional)
        skip_protein_check: If True, skip checking if proteins exist in proteins pkl file
    """
    
    # Read the input CSV file
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    df_subset = df[['canonical_gene_name', 'ligands']].drop_duplicates()
    print(f"Found {len(df_subset)} unique protein-ligand combinations")
    
    # Load proteins and ligands data
    available_proteins = set()
    if not skip_protein_check:
        print(f"Loading proteins from: {proteins_path}")
        with open(proteins_path, 'rb') as f:
            proteins_data = pickle.load(f)
        
        # Extract available protein names
        if isinstance(proteins_data, dict):
            available_proteins = set(proteins_data.keys())
        elif isinstance(proteins_data, list):
            available_proteins = set(proteins_data)
        else:
            print("Warning: Unexpected proteins data format")
            available_proteins = set()
        
        print(f"Available proteins: {len(available_proteins)}")
    else:
        print("Skipping protein validation (skip_protein_check=True)")
    
    print(f"Loading ligands from: {ligands_path}")
    with open(ligands_path, 'rb') as f:
        ligands_data = pickle.load(f)
    
    # Extract available ligand identifiers
    if isinstance(ligands_data, dict):
        available_ligands = set(ligands_data.keys())
    elif isinstance(ligands_data, list):
        available_ligands = set(ligands_data)
    else:
        print("Warning: Unexpected ligands data format")
        available_ligands = set()
    
    print(f"Available ligands: {len(available_ligands)}")
    
    # Process each row and create combinations
    valid_combinations = []
    apo_combinations = []
    holo_combinations = []
    skipped_rows = []
    
    for idx, row in df_subset.iterrows():
        canonical_gene_name = row['canonical_gene_name']
        ligands_str = row['ligands']
        
        # Initialize tracking for this row
        row_valid = True
        missing_items = []
        
        # Check if protein exists (only if not skipping protein check)
        if not skip_protein_check and canonical_gene_name not in available_proteins:
            row_valid = False
            missing_items.append(f"protein:{canonical_gene_name}")
        
        # Process ligands: split by ';' and extract identifier before ':'
        if pd.isna(ligands_str) or ligands_str == '':
            row_valid = False
            missing_items.append("ligands:empty_or_nan")
        else:
            ligand_entries = ligands_str.split(';')
            valid_ligand_ids = []
            
            for ligand_entry in ligand_entries:
                ligand_entry = ligand_entry.strip()
                if ':' in ligand_entry:
                    ligand_id = ligand_entry.split(':')[0].strip()
                else:
                    ligand_id = ligand_entry.strip()
                
                # Check if ligand exists
                if ligand_id == 'No_ligand':
                    # Handle 'No_ligand' case - this will be apo
                    valid_ligand_ids = None
                    break
                elif ligand_id in available_ligands:
                    valid_ligand_ids.append(ligand_id)
                # Don't deal with short peptides for now
                # elif ligand_id.startswith('peptide_'):
                #     # Special case for peptides, which may not be in ligands data
                #     if "("  in ligand_entry or ")" in ligand_entry:
                #         # TODO: fix handling noncanonical peptides later
                #         row_valid = False
                #         missing_items.append(f"ligand:{ligand_id} (contains parentheses)") 
                #     valid_ligand_ids.append(ligand_entry)
                else:
                    row_valid = False
                    missing_items.append(f"ligand:{ligand_id}")
        
        # If row is valid, add to combinations; otherwise, record as skipped
        if row_valid:
            combination = {
                'proteins': [canonical_gene_name],
                'ligands': valid_ligand_ids
            }
            valid_combinations.append(combination)
            
            # Separate into apo and holo based on ligand presence
            if not valid_ligand_ids:
                apo_combinations.append(combination)
            else:
                holo_combinations.append(combination)
        else:
            # Record the skipped row with reason
            skipped_row = row.copy()
            skipped_row['missing_items'] = '; '.join(missing_items)
            skipped_rows.append(skipped_row)
    
    # Save skipped rows to CSV if any
    if skipped_rows:
        skipped_df = pd.DataFrame(skipped_rows)
        skipped_csv_path = output_path.replace('.pkl', '_skipped_rows.csv')
        skipped_df.to_csv(skipped_csv_path, index=False)
    
    # Save separate apo and holo combinations
    apo_output_path = output_path.replace('.pkl', '_apo.pkl')
    holo_output_path = output_path.replace('.pkl', '_holo.pkl')
    
    if len(apo_combinations) > 0:
        with open(apo_output_path, 'wb') as f:
            pickle.dump(apo_combinations, f)

    if len(holo_combinations) > 0:
        with open(holo_output_path, 'wb') as f:
            pickle.dump(holo_combinations, f)
    
    print(f"Saved {len(apo_combinations)} apo combinations to: {apo_output_path}")
    print(f"Saved {len(holo_combinations)} holo combinations to: {holo_output_path}")
    
    # Generate config.yml files
    if config_output_path:
        # Convert all paths to absolute paths
        abs_proteins_path = str(Path(proteins_path).resolve())
        abs_msa_folder = str(Path(msa_folder).resolve())
        abs_ligands_path = str(Path(ligands_path).resolve())
        abs_apo_output_path = str(Path(apo_output_path).resolve())
        abs_holo_output_path = str(Path(holo_output_path).resolve())
        
        # Apo config
        apo_config_path = config_output_path.replace('.yml', '_apo.yml')
        apo_config_data = {
            'proteins': abs_proteins_path,
            'msa_folder': abs_msa_folder,
            'ligands': abs_ligands_path,
            'combinations': abs_apo_output_path
        }

        if len(apo_combinations) == 0:
            print("Warning: No apo combinations found, skipping apo config generation")
        else:
            with open(apo_config_path, 'w') as f:
                yaml.dump(apo_config_data, f, default_flow_style=False, sort_keys=False)            
            print(f"Generated apo config: {apo_config_path}")
        
        # Holo config
        holo_config_path = config_output_path.replace('.yml', '_holo.yml')
        holo_config_data = {
            'proteins': abs_proteins_path,
            'msa_folder': abs_msa_folder,
            'ligands': abs_ligands_path,
            'combinations': abs_holo_output_path
        }
        
        if len(holo_combinations) == 0:
            print("Warning: No holo combinations found, skipping holo config generation")
        else:
            with open(holo_config_path, 'w') as f:
                yaml.dump(holo_config_data, f, default_flow_style=False, sort_keys=False)
            print(f"Generated holo config: {holo_config_path}")
    
    return valid_combinations, len(skipped_rows), apo_combinations, holo_combinations

def main():
    parser = argparse.ArgumentParser(description='Generate kinase protein-ligand combinations from CSV data')
    
    parser.add_argument('--csv_file', required=True, 
                       help='Path to CSV file containing canonical_gene_name and ligands columns')
    parser.add_argument('--proteins_path', required=True,
                       help='Path to proteins pkl file')
    parser.add_argument('--msa_folder', required=True,
                       help='Path to MSA folder')
    parser.add_argument('--ligands_path', required=True,
                       help='Path to ligands pkl file')
    parser.add_argument('--output_path', required=True,
                       help='Path to save the output pkl file with combinations')
    parser.add_argument('--config_output_path', required=False,
                       help='Path to save the config.yml file (optional)')
    parser.add_argument('--skip_protein_check', action='store_true',
                       help='Skip checking if proteins exist in proteins pkl file (useful when MSA is fully cached)')
    
    args = parser.parse_args()
    
    # Validate input files exist (skip proteins_path if skip_protein_check is True)
    files_to_check = [('csv_file', args.csv_file), ('ligands_path', args.ligands_path)]
    if not args.skip_protein_check:
        files_to_check.append(('proteins_path', args.proteins_path))
    
    for path_arg, path_value in files_to_check:
        if not Path(path_value).exists():
            print(f"Error: {path_arg} does not exist: {path_value}")
            sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process the data
    try:
        combinations, skipped_count, apo_combinations, holo_combinations = process_kinase_data(
            csv_file=args.csv_file,
            proteins_path=args.proteins_path,
            msa_folder=args.msa_folder,
            ligands_path=args.ligands_path,
            output_path=args.output_path,
            config_output_path=args.config_output_path,
            skip_protein_check=args.skip_protein_check
        )
        
        print(f"\nProcessing complete:")
        print(f"Total valid combinations: {len(combinations)}")
        print(f"Apo combinations: {len(apo_combinations)}")
        print(f"Holo combinations: {len(holo_combinations)}")
        print(f"Skipped rows: {skipped_count}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()


# python step_1_2_generate_config.py  \
#   --csv_file data/kinase_inferences/gsk3b_human.csv \
#   --proteins_path data/kinase_msas/kinase_msas.pkl \
#   --msa_folder data/kinase_msas \
#   --ligands_path data/kinase_smiles/mapping/kinase_cdd_smiles.pkl \
#   --output_path data/kinase_inferences/gsk3b_apo_inference/combo.pkl \
#   --config_output_path data/kinase_inferences/gsk3b_apo_inference/config.yml \

#!/usr/bin/env python3
"""
Generate MSAs using Boltz, Chai, and Protenix from a FASTA file.
This script creates the MSA generation commands and a master workflow script.
The extraction and organization is handled by cofolding_inference/extract_msas.py.

Final organized structure:
- final_dir/
  - boltz/
    - protein1.csv
    - protein2.csv
    - ...
  - chai/
    - protein1/
      - *.aligned.pqt files
    - protein2/
      - *.aligned.pqt files
    - ...
  - protenix/
    - protein1/
      - MSA files and directories
    - protein2/
      - MSA files and directories
    - ...
  - protein_msa_metadata.pkl
"""

import os
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Import the method classes
from cofolding_inference.boltz import Boltz
from cofolding_inference.chai import Chai
from cofolding_inference.protenix import Protenix
from cofolding_inference.merge_msas import merge_msa_folders, load_pkl_file, compare_sequences


def parse_fasta(fasta_path: str) -> Dict[str, Dict]:
    """
    Parse FASTA file into a dictionary with detailed protein information.
    
    Args:
        fasta_path: Path to FASTA file
        
    Returns:
        Dictionary mapping protein names to dictionaries containing:
        - sequence: The protein sequence
        - full_header: The complete FASTA header
        - index: The position in the FASTA file (0-based)
    """
    proteins = {}
    current_name = None
    current_header = None
    current_seq = []
    index = 0
    
    with open(fasta_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if current_name:
                proteins[current_name] = {
                    'sequence': ''.join(current_seq),
                    'full_header': current_header,
                    'index': index - 1  # Adjust index to be 0-based
                }
            current_header = line
            # Extract name before first | or use the whole header if no | is present
            if '|' in line[1:]:
                current_name = line[1:].split('|')[0]
            else:
                current_name = line[1:]
            current_seq = []
            index += 1
        else:
            current_seq.append(line)
    
    if current_name:
        proteins[current_name] = {
            'sequence': ''.join(current_seq),
            'full_header': current_header,
            'index': index - 1  # Adjust index to be 0-based
        }
    
    return proteins


def generate_boltz_msa(protein_name: str, protein_seq: str, output_dir: str) -> Tuple[str, str, str]:
    """
    Generate MSA for a protein using Boltz method.
    
    Args:
        protein_name: Name of the protein
        protein_seq: Protein sequence
        output_dir: Base output directory
        
    Returns:
        Tuple of (command, output_path, combination_name)
    """
    method = Boltz()
    combo_name = f"{protein_name}_msa_gen"
    
    # Create directories
    protein_input_dir = os.path.join(output_dir, "inputs")
    protein_output_dir = os.path.join(output_dir, "outputs")
    os.makedirs(protein_input_dir, exist_ok=True)
    os.makedirs(protein_output_dir, exist_ok=True)
    
    # Prepare input
    input_path = method.prepare_input(
        protein_names=[protein_name],
        protein_sequences=[protein_seq],
        ligand_names=[],  # No ligands for MSA generation
        ligand_smiles=[],  # No ligands for MSA generation
        combination_name=combo_name,
        input_dir=protein_input_dir,
        msa_paths=None  # Force MSA server usage
    )
    
    # Generate command
    cmd = method.generate_msa_command(
        input_path=input_path,
        output_dir=protein_output_dir
    )
    
    # Add conda activation
    full_cmd = f"{method.get_conda_command()} {cmd}"
    
    return full_cmd, protein_output_dir, combo_name


def generate_chai_msa(protein_name: str, protein_seq: str, output_dir: str) -> Tuple[str, str, str]:
    """
    Generate MSA for a protein using Chai method.
    
    Args:
        protein_name: Name of the protein
        protein_seq: Protein sequence
        output_dir: Base output directory
        
    Returns:
        Tuple of (command, output_path, combination_name)
    """
    method = Chai()
    combo_name = f"{protein_name}_msa_gen"
    
    # Create directories
    protein_input_dir = os.path.join(output_dir, "inputs")
    protein_output_dir = os.path.join(output_dir, "outputs")
    os.makedirs(protein_input_dir, exist_ok=True)
    os.makedirs(protein_output_dir, exist_ok=True)
    
    # Prepare input
    input_path = method.prepare_input(
        protein_names=[protein_name],
        protein_sequences=[protein_seq],
        ligand_names=[],  # No ligands for MSA generation
        ligand_smiles=[],  # No ligands for MSA generation
        combination_name=combo_name,
        input_dir=protein_input_dir,
        msa_paths=None  # Force MSA server usage
    )
    
    # Generate command
    cmd = method.generate_msa_command(
        input_path=input_path,
        output_dir=protein_output_dir,
        protein_name=protein_name
    )
    
    # Add conda activation
    full_cmd = f"{method.get_conda_command()} {cmd}"
    
    return full_cmd, protein_output_dir, combo_name


def generate_protenix_msa(fasta_path: str, output_dir: str) -> Tuple[str, str, str]:
    """
    Generate MSA for a protein using Protenix method.
    
    Args:
        fasta_path: Path to the FASTA file containing protein sequences
        output_dir: Base output directory
        
    Returns:
        Tuple of (command, output_path, combination_name)
    """
    method = Protenix()
    
    protein_output_dir = os.path.join(output_dir, "outputs", "protenix")
    os.makedirs(protein_output_dir, exist_ok=True)
    
    # Generate MSA command
    cmd = method.generate_msa_command(
        input_path=fasta_path,
        output_dir=protein_output_dir
    )
    
    # Add conda activation
    full_cmd = f"{method.get_conda_command()} {cmd}"
    
    return full_cmd, protein_output_dir, ""




def check_existing_msas(merge_path: str, proteins: Dict[str, Dict], methods: List[str]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Check which proteins already exist in the merge path and filter out those with matching sequences.
    
    Args:
        merge_path: Path to existing MSA folder
        proteins: Dictionary of proteins from FASTA file
        methods: List of methods to check for
        
    Returns:
        Tuple of (proteins_to_generate, existing_proteins)
    """
    if not merge_path or not os.path.exists(merge_path):
        print("No existing MSA folder found, will generate MSAs for all proteins")
        return proteins, {}
    
    # Find the pickle file in the merge path directory
    import glob
    pkl_files = glob.glob(os.path.join(merge_path, "*.pkl"))
    
    if not pkl_files:
        print("No pickle file found in merge-path, will generate MSAs for all proteins")
        return proteins, {}
    
    # Use the first (and assumedly only) pickle file found
    existing_pkl_path = pkl_files[0]
    print(f"Loading existing MSA metadata from: {os.path.basename(existing_pkl_path)}")
    
    existing_data = load_pkl_file(existing_pkl_path)
    
    if not existing_data:
        print("No existing MSA metadata found, will generate MSAs for all proteins")
        return proteins, {}
    
    print(f"Found existing MSA data for {len(existing_data)} proteins")
    
    proteins_to_generate = {}
    existing_proteins = {}
    
    for protein_name, protein_info in proteins.items():
        protein_seq = protein_info['sequence']
        
        if protein_name in existing_data:
            existing_seq = existing_data[protein_name].get('sequence', '')
            
            if compare_sequences(protein_seq, existing_seq):
                # Check if all required methods exist
                all_methods_exist = True
                missing_methods = []
                
                for method in methods:
                    method_key = f"{method}_msa"
                    if method_key not in existing_data[protein_name]:
                        all_methods_exist = False
                        missing_methods.append(method)
                    else:
                        # Check if the actual files exist
                        method_path = os.path.join(merge_path, existing_data[protein_name][method_key])
                        if not os.path.exists(method_path):
                            all_methods_exist = False
                            missing_methods.append(method)
                
                if all_methods_exist:
                    print(f"  Skipping {protein_name} - all methods already exist with matching sequence")
                    existing_proteins[protein_name] = protein_info
                    continue
                else:
                    print(f"  Will generate {protein_name} - missing methods: {missing_methods}")
            else:
                print(f"  Will generate {protein_name} - sequence differs from existing")
        else:
            print(f"  Will generate {protein_name} - not found in existing data")
        
        proteins_to_generate[protein_name] = protein_info
    
    print(f"\nSummary:")
    print(f"  Proteins to skip (already exist): {len(existing_proteins)}")
    print(f"  Proteins to generate: {len(proteins_to_generate)}")
    
    return proteins_to_generate, existing_proteins


def generate_bash_script(commands: List[str], output_path: str, script_name: str):
    """
    Generate a bash script with the given commands.
    
    Args:
        commands: List of commands to include in the script
        output_path: Directory to save the script
        script_name: Name of the script (without .sh extension)
    """
    script_path = os.path.join(output_path, f"{script_name}.sh")
    
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Generated bash script for {script_name}\n")
        f.write("set -e  # Exit on any error\n\n")
        
        for i, cmd in enumerate(commands, 1):
            f.write(f"echo \"Running command {i}/{len(commands)}...\"\n")
            f.write(f"{cmd}\n")
            f.write("echo \"Command completed.\"\n\n")
        
        f.write("echo \"All commands completed successfully!\"\n")
    
    # Make script executable
    print(f"Generated: {script_path}")


def main():
    """Main function for standalone script usage."""
    parser = argparse.ArgumentParser(description='Generate MSAs using multiple methods from a FASTA file. Extraction is handled automatically by the generated workflow script.')
    parser.add_argument('fasta', help='Path to FASTA file with protein sequences')
    parser.add_argument('--output-dir', default='tmp-msa/msa_generation',
                       help='Base output directory for intermediate results')
    parser.add_argument('--final-dir', default='tmp-msa/msas',
                       help='Final directory to store organized MSA files')
    parser.add_argument('--script-dir', default='tmp-msa/scripts',
                       help='Directory to save generated bash scripts')
    parser.add_argument('--methods', nargs='+', choices=['boltz'], 
                       default=['boltz'],
                       help='Methods to generate MSAs for (default: all)')
    parser.add_argument('--merge-path', default=None,
                       help='Path to existing MSA folder to merge with. If provided, will merge newly generated MSAs into this folder.')
    
    args = parser.parse_args()
    
    # Convert all paths to absolute paths
    fasta_path = Path(args.fasta).resolve()
    output_dir = Path(args.output_dir).resolve()
    final_dir = Path(args.final_dir).resolve()
    script_dir = Path(args.script_dir).resolve()
    
    # Create base directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(script_dir, exist_ok=True)
    
    # Parse FASTA file
    proteins = parse_fasta(str(fasta_path))
    print(f"Found {len(proteins)} proteins in FASTA file")
    
    # Check existing MSAs if merge-path is provided
    if args.merge_path:
        proteins_to_generate, existing_proteins = check_existing_msas(
            args.merge_path, proteins, args.methods
        )
        
        if not proteins_to_generate:
            print("\nAll proteins already exist with matching sequences and methods!")
            print("No MSA generation needed.")
            return
    else:
        proteins_to_generate = proteins
        existing_proteins = {}
    
    # sequence to name mapping (only for proteins we need to generate)
    seq_to_name_map = {info['sequence']: name for name, info in proteins_to_generate.items()}

    # Process proteins that need MSA generation
    msa_commands = []
    
    # Handle Protenix separately since it processes all proteins at once
    if "protenix" in args.methods and proteins_to_generate:
        # Create a temporary FASTA file with only proteins to generate
        temp_fasta_path = output_dir / "temp_proteins_to_generate.fasta"
        with open(temp_fasta_path, 'w') as f:
            for protein_name, protein_info in proteins_to_generate.items():
                f.write(f"{protein_info['full_header']}\n")
                f.write(f"{protein_info['sequence']}\n")
        
        cmd, protein_output_dir, combo_name = generate_protenix_msa(
            fasta_path=str(temp_fasta_path),
            output_dir=str(output_dir)
        )
        msa_commands.append(cmd)

    for protein_name, protein_info in proteins_to_generate.items():
        protein_seq = protein_info['sequence']
        protein_index = protein_info['index']
        print(f"Processing protein: {protein_name} (index: {protein_index})")
        
        # Generate MSA commands for each method
        for method in args.methods:
            if method == 'boltz':
                cmd, protein_output_dir, combo_name = generate_boltz_msa(
                    protein_name=protein_name,
                    protein_seq=protein_seq,
                    output_dir=str(output_dir)
                )
                msa_commands.append(cmd)
                
            elif method == 'chai':
                cmd, protein_output_dir, combo_name = generate_chai_msa(
                    protein_name=protein_name,
                    protein_seq=protein_seq,
                    output_dir=str(output_dir)
                )
                msa_commands.append(cmd)
                
    
    # Generate bash scripts
    if msa_commands:
        generate_bash_script(msa_commands, str(script_dir), "run_msa_generation")
        print(f"Generated MSA generation script with {len(msa_commands)} commands")
    
    # Generate master script
    master_commands = []
    if msa_commands:
        master_commands.append(f"bash {str(script_dir)}/run_msa_generation.sh")
    
    # Add extraction command using Python script (use temp fasta if we filtered proteins)
    methods_str = " ".join(args.methods)
    if proteins_to_generate != proteins and "protenix" in args.methods:
        # Use the temporary FASTA file for extraction if we filtered proteins
        temp_fasta_path = output_dir / "temp_proteins_to_generate.fasta"
        extract_cmd = f"python {Path(__file__).parent}/cofolding_inference/extract_msas.py {str(temp_fasta_path)} "
    else:
        extract_cmd = f"python {Path(__file__).parent}/cofolding_inference/extract_msas.py {str(fasta_path)} "
    extract_cmd += f"--output-dir {str(output_dir)} --final-dir {str(final_dir)} --methods {methods_str}"
    master_commands.append(extract_cmd)
    
    # Add merge command if merge-path is provided
    if args.merge_path:
        merge_path = Path(args.merge_path).resolve()
        merge_cmd = f"python {Path(__file__).parent}/cofolding_inference/merge_msas.py {str(final_dir)} {str(merge_path)}"
        master_commands.append(merge_cmd)
        print(f"Merge functionality enabled - will merge into: {merge_path}")
    
    # Add cleanup command for temporary FASTA file if it was created
    if proteins_to_generate != proteins and "protenix" in args.methods:
        temp_fasta_path = output_dir / "temp_proteins_to_generate.fasta"
        cleanup_cmd = f"rm -f {str(temp_fasta_path)}"
        master_commands.append(cleanup_cmd)
    
    if master_commands:
        generate_bash_script(master_commands, str(script_dir), "run_complete_workflow")
        print(f"Generated complete workflow script")
        print(f"Run: bash {str(script_dir)}/run_complete_workflow.sh")
        
        # If merge-path is provided, also offer direct merge option
        if args.merge_path:
            print(f"\nAlternatively, you can run the merge separately after MSA generation:")
            print(f"python cofolding_inference/merge_msas.py {str(final_dir)} {str(merge_path)}")
    
    # Print final summary
    if args.merge_path and existing_proteins:
        print(f"\n=== Final Summary ===")
        print(f"Proteins skipped (already exist): {len(existing_proteins)}")
        for protein_name in existing_proteins.keys():
            print(f"  - {protein_name}")
        print(f"Proteins to generate: {len(proteins_to_generate)}")
        for protein_name in proteins_to_generate.keys():
            print(f"  + {protein_name}")


if __name__ == "__main__":
    main()

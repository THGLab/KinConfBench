#!/usr/bin/env python3
"""
Script to run inference on all combinations of proteins and ligands using multiple methods.
This script:
1. Loads protein MSA metadata from a pickle file
2. Loads configuration from a YAML file
3. Generates all combinations of protein-ligand pairs
4. Creates commands for running inference on each pair using all methods
5. Generates bash scripts to execute these commands
6. Includes commands to organize structural files after inference
"""

import os
import argparse
import pickle
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Import the method classes
from cofolding_inference.boltz import Boltz
from cofolding_inference.chai import Chai
from cofolding_inference.protenix import Protenix
from cofolding_inference.rfaa import RoseTTAFoldAA
from cofolding_inference.envs import FOLDER


def load_pickle(pkl_path: str) -> Dict[str, Dict]:
    """
    Load protein MSA metadata from a pickle file.
    
    Args:
        pkl_path: Path to the pickle file
        
    Returns:
        Dictionary mapping protein names to dictionaries containing MSA paths
    """
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_combinations(proteins: Dict[str, Dict], ligands: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Generate all combinations of proteins and ligands.
    Handles both regular ligands (SMILES) and peptide ligands (sequences).
    
    Args:
        proteins: Dictionary mapping protein names to dictionaries containing sequences
        ligands: Dictionary mapping ligand names to SMILES strings or peptide sequences
        
    Returns:
        List of dictionaries containing protein and ligand information for each combination
    """
    combinations = []
    
    for protein_name, protein_info in proteins.items():
        for ligand_name, ligand_data in ligands.items():
            protein_names = [protein_name]
            protein_sequences = [protein_info['sequence']]
            processed_ligand_names = []
            ligand_smiles = []
            
            if len(ligand_data) == 0:
                # No ligands provided
                continue
            # Check if this is a peptide (contains "peptide_" and ":")
            elif isinstance(ligand_data, str) and "peptide_" in ligand_data.lower() and ":" in ligand_data:
                continue
                # # Parse peptide format: "peptide_B: SEQUENCE"
                # parts = ligand_data.split(":", 1)
                # if len(parts) == 2:
                #     peptide_id = parts[0].strip()
                #     peptide_sequence = parts[1].strip()
                    
                #     # Add peptide sequence to protein sequences
                #     protein_names.append(f"{ligand_name}_peptide")
                #     protein_sequences.append(peptide_sequence)
                # else:
                #     # If parsing fails, treat as regular ligand
                #     processed_ligand_names.append(ligand_name)
                #     ligand_smiles.append(ligand_data)
            else:
                # Regular ligand (SMILES)
                processed_ligand_names.append(ligand_name)
                ligand_smiles.append(ligand_data)
            
            name = f"{'_'.join(protein_names)}_{'_'.join(processed_ligand_names)}" if len(processed_ligand_names) > 0 else f"{'_'.join(protein_names)}_apo"

            combination = {
                'proteins': protein_names,
                'protein_sequences': protein_sequences,
                'ligands': processed_ligand_names,
                'ligand_smiles': ligand_smiles,
                'name': name
            }
            combinations.append(combination)
    
    return combinations


def load_combinations_from_pickle(combinations_path: str, proteins_map: Dict[str, Dict], ligands_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Load combinations from pickle file and resolve protein sequences and ligand SMILES.
    Handles both regular ligands (SMILES) and peptide ligands (sequences).
    
    Args:
        combinations_path: Path to combinations pickle file
        proteins_map: Dictionary mapping protein names to protein data (with 'sequence' key)
        ligands_map: Dictionary mapping ligand names to SMILES strings or peptide sequences
        
    Returns:
        List of combination dictionaries with resolved sequences and SMILES
    """
    # Load combinations from pickle file
    combinations_raw = load_pickle(combinations_path)
    
    combinations = []
    
    for combo_raw in combinations_raw:
        protein_names = combo_raw['proteins'].copy()
        ligand_names = combo_raw['ligands']
        
        # Resolve protein sequences
        protein_sequences = []
        for protein_name in protein_names:
            if protein_name not in proteins_map:
                raise ValueError(f"Protein '{protein_name}' not found in proteins data")
            protein_sequences.append(proteins_map[protein_name]['sequence'])
        
        # Resolve ligand SMILES and handle peptides
        ligand_smiles = []
        processed_ligand_names = []

        if not ligand_names:
            name  = f"{'_'.join(protein_names)}_apo"

        else:
            for ligand_name in ligand_names:
                if ligand_name not in ligands_map.keys():
                    print(f"Ligand '{ligand_name}' not found in ligands data")
                    break  # Skip this ligand if it's not found

                # Always get the data from ligands_map
                ligand_data = ligands_map[ligand_name]

                # NOTE: Turn off this function for now for pentameter running
                # TODO: we need to eventually worry about peptides but just not now.     
                # # Check if this is a peptide (contains "peptide_" and ":")
                # if isinstance(ligand_data, str) and "peptide_" in ligand_data.lower() and ":" in ligand_data:
                #     # Parse peptide format: "peptide_B: SEQUENCE"
                #     parts = ligand_data.split(":", 1)
                #     if len(parts) == 2:
                #         peptide_id = parts[0].strip()
                #         peptide_sequence = parts[1].strip()
                #
                #         # Add peptide sequence to protein sequences
                #         protein_names.append(f"{ligand_name}_peptide")
                #         protein_sequences.append(peptide_sequence)
                
                #         # Skip adding this to ligand_smiles since it's now a protein
                #         continue
                
                # # Also check if the ligand name itself indicates it's a peptide
                # elif 'peptide_' in ligand_name.lower() and ':' in ligand_name:
                #     try:
                #         # Try to extract peptide sequence from the name
                #         parts = ligand_name.split(":", 1)
                #         if len(parts) == 2:
                #             peptide_sequence = parts[1].strip()
                            
                #             # Add peptide sequence to protein sequences
                #             protein_names.append(f"{ligand_name}_peptide")
                #             protein_sequences.append(peptide_sequence)
                            
                #             # Skip adding this to ligand_smiles
                #             continue
                #     except:
                #         # If extraction fails, fall through to treating as regular ligand
                #         pass
                
                # Regular ligand (SMILES)
                ligand_smiles.append(ligand_data)
                processed_ligand_names.append(ligand_name)
        
            name = f"{'_'.join(protein_names)}_{'_'.join(processed_ligand_names)}"

        # Create combination dictionary
        combination = {
            'proteins': protein_names,
            'protein_sequences': protein_sequences,
            'ligands': processed_ligand_names,
            'ligand_smiles': ligand_smiles,
            'name': name
        }
        combinations.append(combination)
    
    return combinations

# Define a mapping of method names to their corresponding classes
METHOD_CLASSES = {
    'boltz': Boltz,
    'chai': Chai,
    'protenix': Protenix,
    'rfaa': RoseTTAFoldAA
}

def generate_inference_command(method_name: str, combination: Dict[str, Any], 
                             input_dir: str, output_dir: str, 
                             protein_msa_map: Dict[str, Dict],
                             msa_folder: str = '', num_samples: int = 1,
                             predict_affinity: bool = False) -> Tuple[str, str]:
    """
    Generate inference command for the specified method.
    
    Args:
        method_name: Name of the method (boltz, chai, protenix)
        combination: Dictionary containing protein and ligand information
        input_dir: Directory to store input files
        output_dir: Directory to store output files
        protein_msa_map: Dictionary mapping protein names to dictionaries containing MSA paths
        msa_folder: Base folder for MSA paths
        
    Returns:
        Tuple of (command, input_path)
    """
    # Get the appropriate method class
    method_class = METHOD_CLASSES.get(method_name)
    if not method_class:
        raise ValueError(f"Unknown method: {method_name}")
    
    # Initialize the method
    method = method_class()
    
    # Extract combination details
    protein_names = combination['proteins']
    protein_sequences = combination['protein_sequences']
    ligand_names = combination['ligands']
    ligand_smiles = combination['ligand_smiles']
    combination_name = combination['name']
    
    # Get MSA paths for each protein based on the method
    msa_paths = []
    for protein_name in protein_names:
        msa_path = None
        msa_key = f"{method_name}_msa"
        if msa_key in protein_msa_map[protein_name]:
            msa_path = os.path.join(protein_msa_map[protein_name][msa_key])
            if msa_folder and not os.path.isabs(msa_path):
                msa_path = os.path.join(msa_folder, msa_path)
        msa_paths.append(msa_path)
    
    # Prepare input - pass predict_affinity only for Boltz method
    if method_name == 'boltz':
        input_path = method.prepare_input(
            protein_names=protein_names,
            protein_sequences=protein_sequences,
            ligand_names=ligand_names,
            ligand_smiles=ligand_smiles,
            combination_name=combination_name,
            input_dir=input_dir,
            msa_paths=msa_paths,
            predict_affinity=predict_affinity
        )
    else:
        input_path = method.prepare_input(
            protein_names=protein_names,
            protein_sequences=protein_sequences,
            ligand_names=ligand_names,
            ligand_smiles=ligand_smiles,
            combination_name=combination_name,
            input_dir=input_dir,
            msa_paths=msa_paths
        )
    
    # Generate command with num_samples parameter
    cmd = method.generate_command(
        input_path=input_path,
        output_dir=output_dir,
        combination_name=combination_name,
        num_samples=num_samples
    )
    
    return cmd, input_path


def generate_bash_script(commands: List[str], output_path: str, script_name: str, num_gpus: int = 1):
    """
    Generate bash script(s) with the given commands, optionally parallelized across multiple GPUs.
    
    Args:
        commands: List of commands to include in the script
        output_path: Directory to save the script
        script_name: Name of the script (without .sh extension)
        num_gpus: Number of GPUs to parallelize across (default: 1)
    """
    if num_gpus == 1:
        # Single GPU - generate single script as before
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
        os.chmod(script_path, 0o755)
        print(f"Generated: {script_path}")
    
    else:
        # Multi-GPU - generate parallel scripts
        generate_parallel_bash_scripts(commands, output_path, script_name, num_gpus)


def generate_parallel_bash_scripts(commands: List[str], output_path: str, script_name: str, num_gpus: int):
    """
    Generate multiple bash scripts for parallel execution across GPUs.
    
    Args:
        commands: List of commands to include in the scripts
        output_path: Directory to save the scripts
        script_name: Base name of the scripts (without .sh extension)
        num_gpus: Number of GPUs to parallelize across
    """
    if not commands:
        return
    
    # Split commands across GPUs
    commands_per_gpu = [[] for _ in range(num_gpus)]
    for i, cmd in enumerate(commands):
        gpu_id = i % num_gpus
        commands_per_gpu[gpu_id].append(cmd)
    
    # Generate individual GPU scripts
    gpu_script_paths = []
    for gpu_id in range(num_gpus):
        if not commands_per_gpu[gpu_id]:
            continue
            
        gpu_script_name = f"{script_name}_gpu{gpu_id}"
        gpu_script_path = os.path.join(output_path, f"{gpu_script_name}.sh")
        gpu_script_paths.append(gpu_script_path)
        
        with open(gpu_script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Generated bash script for {gpu_script_name}\n")
            f.write("set -e  # Exit on any error\n")
            f.write(f"export CUDA_VISIBLE_DEVICES={gpu_id}\n")
            f.write(f"echo \"Starting GPU {gpu_id} with {len(commands_per_gpu[gpu_id])} commands...\"\n\n")
            
            for i, cmd in enumerate(commands_per_gpu[gpu_id], 1):
                f.write(f"echo \"GPU {gpu_id}: Running command {i}/{len(commands_per_gpu[gpu_id])}...\"\n")
                f.write(f"{cmd}\n")
                f.write(f"echo \"GPU {gpu_id}: Command completed.\"\n\n")
            
            f.write(f"echo \"GPU {gpu_id}: All commands completed successfully!\"\n")
        
        # Make script executable
        os.chmod(gpu_script_path, 0o755)
        print(f"Generated: {gpu_script_path}")
    
    # Generate master parallel script
    master_script_path = os.path.join(output_path, f"{script_name}.sh")
    with open(master_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Master script for parallel {script_name} execution\n")
        f.write("set -e  # Exit on any error\n\n")
        f.write(f"echo \"Starting parallel execution across {num_gpus} GPUs...\"\n\n")
        
        # Start all GPU scripts in background
        for gpu_script_path in gpu_script_paths:
            f.write(f"bash {gpu_script_path} &\n")
        
        f.write("\n# Wait for all background jobs to complete\n")
        f.write("wait\n\n")
        f.write("echo \"All parallel jobs completed successfully!\"\n")
    
    # Make master script executable
    os.chmod(master_script_path, 0o755)
    print(f"Generated master script: {master_script_path}")


def generate_master_parallel_script(methods: List[str], method_commands: Dict[str, List[str]], 
                                  script_dir: str, skip_organize: bool, num_gpus: int):
    """
    Generate a master script that runs method scripts sequentially (to avoid CUDA memory conflicts)
    but each method uses all GPUs in parallel.
    
    Args:
        methods: List of method names
        method_commands: Dictionary mapping method names to their commands
        script_dir: Directory containing the scripts
        skip_organize: Whether to skip file organization
        num_gpus: Number of GPUs being used
    """
    master_script_path = os.path.join(script_dir, "run_all_inferences.sh")
    
    with open(master_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Master script for sequential method execution with {num_gpus} GPU parallelization per method\n")
        f.write("set -e  # Exit on any error\n\n")
        f.write(f"echo \"Starting inference with {num_gpus} GPUs per method (methods run sequentially)...\"\n\n")
        
        # Run method scripts sequentially to avoid CUDA memory conflicts
        active_methods = []
        for method_name in methods:
            if method_name in method_commands and method_commands[method_name]:
                f.write(f"echo \"Starting {method_name} inference across {num_gpus} GPUs...\"\n")
                f.write(f"bash {script_dir}/run_{method_name}_inference.sh\n")
                f.write(f"echo \"{method_name} inference completed!\"\n\n")
                active_methods.append(method_name)
        
        if active_methods:
            f.write("echo \"All inference methods completed!\"\n\n")
        
        # Run file organization after all inference is complete
        if not skip_organize:
            f.write("# Organize prediction files\n")
            f.write("echo \"Starting file organization...\"\n")
            f.write(f"bash {script_dir}/organize_prediction_files.sh\n")
            f.write("echo \"File organization completed!\"\n\n")
        
        f.write("echo \"All tasks completed successfully!\"\n")
    
    # Make script executable
    os.chmod(master_script_path, 0o755)
    print(f"Generated master script with sequential methods: {master_script_path}")


def generate_file_organization_command(
    source_dir: str, target_dir: str, *, extract_affinities: bool = False
) -> str:
    """
    Generate a command to run the file organization script.

    Args:
        source_dir: Path to the base directory containing model folders
        target_dir: Target directory for organized predictions
        extract_affinities: Whether to extract Boltz affinity predictions

    Returns:
        Command string to run the file organization script
    """
    gather_file = FOLDER / "gather_structural_files.py"
    cmd = f"conda run -n kinconfbench python {str(gather_file)} --source {source_dir} --target {target_dir}"

    if extract_affinities:
        affinity_script = FOLDER / "extract_boltz_affinities.py"
        boltz_output_dir = Path(source_dir) / "boltz"
        affinity_csv = Path(target_dir) / "boltz_affinities.csv"
        cmd += f" && conda run -n kinconfbench python {str(affinity_script)} {str(boltz_output_dir)} --output {str(affinity_csv)}"

    return cmd


def main():
    """Main function for standalone script usage."""
    parser = argparse.ArgumentParser(description='Run inference on all combinations of proteins and ligands')
    parser.add_argument('config', help='Path to YAML configuration file')
    parser.add_argument('--output-folder', default='inference_run',
                       help='Base output folder for all inference files (default: inference_run)')
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["boltz", "chai", "protenix", "rfaa"],
        default=["boltz", "chai", "protenix"],
        help="Methods to run inference for (default: boltz chai protenix)",
    )
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples to generate for each method (default: 5)')
    parser.add_argument('--skip-organize', action='store_true',
                       help='Skip the file organization step')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='Number of GPUs to use for parallel inference (default: 1)')
    parser.add_argument('--predict-affinity', action='store_true',
                       help='Enable affinity prediction for Boltz method (adds affinity properties to YAML)')
    
    args = parser.parse_args()
    
    # Convert config path to absolute path
    config_path = Path(args.config).resolve()
    
    # Create all directories as subfolders under the main output folder
    base_folder = Path(args.output_folder).resolve()
    output_dir = base_folder / 'outputs'
    input_dir = base_folder / 'inputs'
    script_dir = base_folder / 'scripts'
    organized_output_dir = base_folder / 'predictions'
    
    # Create base directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(script_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(str(config_path))
    
    # Load protein MSA metadata
    protein_msa_path = config['proteins']
    protein_msa_map = load_pickle(protein_msa_path)
    
    # Get MSA folder path from config (if provided)
    msa_folder = config.get('msa_folder', '')
    
    # Get ligands from config
    ligands = config['ligands']
    ligands_map = load_pickle(ligands)
    
    # Generate combinations based on configuration
    combinations_spec = config.get('combinations', 'everything')
    if combinations_spec == 'everything':
        # Generate all possible combinations
        combinations = generate_combinations(protein_msa_map, ligands_map)
    else:
        # Load combinations from pickle file
        combinations_path = combinations_spec
        if not os.path.isabs(combinations_path):
            # If relative path, make it relative to config file directory
            config_dir = os.path.dirname(str(config_path))
            combinations_path = os.path.join(config_dir, combinations_path)
        combinations = load_combinations_from_pickle(combinations_path, protein_msa_map, ligands_map)
    
    print(f"Generated {len(combinations)} combinations")
    
    # Display affinity prediction status
    if args.predict_affinity:
        print("Affinity prediction enabled for Boltz method")
    
    # No need to initialize methods here anymore, as we're using separate functions
    
    # Generate commands for each method and combination
    method_commands = {method_name: [] for method_name in args.methods}
    
    if args.num_gpus > 1:
        print(f"Parallelization enabled: Using {args.num_gpus} GPUs")
        print(f"Commands will be distributed across {args.num_gpus} GPUs")
    else:
        print("Single GPU mode: Sequential execution")
    
    for combination in combinations:
        for method_name in args.methods:
            cmd, input_path = generate_inference_command(
                method_name=method_name,
                combination=combination,
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                protein_msa_map=protein_msa_map,
                msa_folder=msa_folder,
                num_samples=args.num_samples,
                predict_affinity=args.predict_affinity,
            )
            method_commands[method_name].append(cmd)
    
    # Generate bash scripts for each method
    for method_name, commands in method_commands.items():
        if commands:
            generate_bash_script(commands, str(script_dir), f"run_{method_name}_inference", args.num_gpus)
            if args.num_gpus > 1:
                print(f"Generated parallel scripts for {method_name}: {len(commands)} commands across {args.num_gpus} GPUs")
    # Generate file organization command unless explicitly skipped
    if not args.skip_organize:
        file_org_cmd = generate_file_organization_command(
            source_dir=str(output_dir),
            target_dir=str(organized_output_dir),
            extract_affinities=args.predict_affinity,
        )
        
        # Create a separate script for file organization
        generate_bash_script([file_org_cmd], str(script_dir), "organize_prediction_files")
    
    # Generate master script based on parallelization mode
    if args.num_gpus == 1:
        # Sequential execution for single GPU
        master_commands = []
        for method_name in args.methods:
            if method_name in method_commands and method_commands[method_name]:
                master_commands.append(f"bash {str(script_dir)}/run_{method_name}_inference.sh")
        
        # Add file organization to master script unless explicitly skipped
        if not args.skip_organize:
            master_commands.append(f"bash {str(script_dir)}/organize_prediction_files.sh")
        
        if master_commands:
            generate_bash_script(master_commands, str(script_dir), "run_all_inferences")
    
    else:
        # Parallel execution for multiple GPUs - methods run in parallel, file organization runs after
        generate_master_parallel_script(args.methods, method_commands, str(script_dir), 
                                      args.skip_organize, args.num_gpus)


if __name__ == "__main__":
    main()


# python step_2_run_all_inferences.py\
#  data/kinase_inferences/gsk3b_apo_inference/config_apo.yml\
#  --output-folder data/kinase_inferences/gsk3b_apo_inference\
#  --num-samples 100\
#  --num-gpus 1\
#  --predict-affinity\

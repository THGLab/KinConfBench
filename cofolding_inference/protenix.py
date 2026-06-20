"""
Protenix co-folding method implementation.

Based on the existing write_protenix_inference.py script, this handles:
- JSON file generation for protein and ligand inputs
- MSA generation with FASTA files
- Protenix predict commands
"""

import os
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from .base import CoFoldingMethod
from .envs import PROTENIX_CONDA_ENV, PROTENIX_VENV

class Protenix(CoFoldingMethod):
    """Protenix co-folding method implementation."""

    def __init__(self):
        if PROTENIX_VENV and os.path.isdir(PROTENIX_VENV):
            super().__init__(name="protenix", venv_path=PROTENIX_VENV)
        else:
            super().__init__(name="protenix", conda_env=PROTENIX_CONDA_ENV)
    
    def prepare_input(self, protein_names: List[str], protein_sequences: List[str], 
                     ligand_names: List[str], ligand_smiles: List[str], combination_name: str,
                     input_dir: str, msa_paths: Optional[List[str]] = None) -> str:
        """
        Prepare JSON input file for Protenix.
        
        Format:
        [
            {
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": "...",
                        "count": 1,
                        "msa": {
                            "precomputed_msa_dir": "path/to/msa",
                            "pairing_db": "uniref100"
                        }
                    }
                },
                {
                    "ligand": {
                        "ligand": "SMILES_STRING",
                        "count": 1
                    }
                }
            ],
            "name": "combination_name"
            }
        ]
        """
        # Create method-specific input directory
        method_input_dir = os.path.join(input_dir, self.name)
        os.makedirs(method_input_dir, exist_ok=True)
        
        # Create JSON structure as a list of objects
        json_data = [
            {
                "sequences": [],
                "name": combination_name
            }
        ]
        
        # Add protein sequences
        for i, (protein_name, protein_seq) in enumerate(zip(protein_names, protein_sequences)):
            protein_entry = {
                "proteinChain": {
                    "sequence": protein_seq,
                    "count": 1
                }
            }
            
            # Add MSA path if available
            if msa_paths and i < len(msa_paths) and msa_paths[i]:
                msa_dir = os.path.dirname(msa_paths[i])
                protein_entry["proteinChain"]["msa"] = {
                    "precomputed_msa_dir": msa_dir,
                    "pairing_db": "uniref100"
                }
            
            json_data[0]["sequences"].append(protein_entry)
        
        # Add ligand SMILES
        for i, (ligand_name, smiles) in enumerate(zip(ligand_names, ligand_smiles)):
            ligand_entry = {
                "ligand": {
                    "ligand": smiles,
                    "count": 1
                }
            }
            json_data[0]["sequences"].append(ligand_entry)
        
        # Write JSON file
        json_filename = f"{combination_name}.json"
        json_path = os.path.join(method_input_dir, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        return json_path
    
    def generate_command(self, input_path: str, output_dir: str, 
                        combination_name: str, num_samples: int = 1) -> str:
        """
        Generate Protenix prediction command.
        
        Uses --use_msa_server if no MSA is provided in the JSON file.
        
        Args:
            input_path: Path to input file/directory
            output_dir: Directory to store outputs
            combination_name: Name for this protein-ligand combination
            num_samples: Number of samples to generate (default: 1)
        """
        method_output_dir = os.path.join(output_dir, self.name, combination_name)
        
        # Check if JSON contains MSA path
        with open(input_path, 'r') as f:
            json_data = json.load(f)
        
        # Check if any protein has MSA information
        has_msa = False
        
        # Handle both list and single object formats
        if isinstance(json_data, list):
            # List of objects format
            for item in json_data:
                for entry in item.get("sequences", []):
                    if "proteinChain" in entry and "msa" in entry["proteinChain"]:
                        has_msa = True
                        break
                if has_msa:
                    break
        else:
            # Single object format (for backward compatibility)
            for entry in json_data.get("sequences", []):
                if "proteinChain" in entry and "msa" in entry["proteinChain"]:
                    has_msa = True
                    break
        
        cmd = self.get_conda_command() + " "
        cmd += f"protenix predict --input {input_path} --out_dir {method_output_dir}"
        if not has_msa:
            cmd += " --use_msa_server"
        
        # Add num_samples parameter
        cmd += f" --sample {num_samples}"
        
        return cmd
    
    def generate_msa_command(self, input_path: str, output_dir: str) -> str:
        """
        Generate command to create MSAs for proteins in a FASTA file.
        
        Args:
            fasta_path: Path to FASTA file with protein sequences
            output_dir: Directory to store MSA outputs
            
        Returns:
            Command string to generate MSAs
        """
        cmd = f"protenix msa --input {input_path} --out_dir {output_dir}"
        return cmd

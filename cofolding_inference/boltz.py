"""
Boltz co-folding inference implementation.

Author: Oliver Sun
Date: 2025-05-23
"""

import os
import yaml
from typing import List, Optional
from pathlib import Path
from .base import CoFoldingMethod
from .envs import BOLTZ_CONDA_ENV, BOLTZ_VENV


class Boltz(CoFoldingMethod):
    """Boltz co-folding method implementation."""

    def __init__(self):
        if BOLTZ_VENV and os.path.isdir(BOLTZ_VENV):
            super().__init__(name="boltz", venv_path=BOLTZ_VENV)
        else:
            super().__init__(name="boltz", conda_env=BOLTZ_CONDA_ENV)
    
    def prepare_input(self, protein_names: List[str], protein_sequences: List[str], 
                     ligand_names: List[str], ligand_smiles: List[str], combination_name: str,
                     input_dir: str, msa_paths: Optional[List[str]] = None, 
                     predict_affinity: bool = False) -> str:
        """
        Prepare YAML input file for Boltz.
        
        New format:
        version: 1
        sequences:
          - protein:
              id: [A, B]
              sequence: <protein_sequence>
              msa: <path_to_msa_file>
          - ligand:
              id: [C]
              smiles: <ligand_smiles>
        properties:  # Optional, only if predict_affinity=True
          - affinity:
              binder: <ligand_id>
        
        Args:
            protein_names: List of protein names
            protein_sequences: List of protein sequences
            ligand_names: List of ligand names
            ligand_smiles: List of ligand SMILES strings
            combination_name: Name for this combination
            input_dir: Directory to store input files
            msa_paths: Optional list of MSA file paths
            predict_affinity: Whether to include affinity prediction properties
        """
        # Create method-specific input directory
        method_input_dir = os.path.join(input_dir, self.name)
        os.makedirs(method_input_dir, exist_ok=True)
        
        # Create YAML file
        yaml_filename = f"{combination_name}.yaml"
        yaml_path = os.path.join(method_input_dir, yaml_filename)
        
        # Build YAML structure
        yaml_data = {
            'version': 1,
            'sequences': []
        }
        
        # Group proteins with same sequence together
        protein_groups = {}
        for i, (protein_name, protein_seq) in enumerate(zip(protein_names, protein_sequences)):
            chain_id = chr(ord('A') + i)  # A, B, C, etc.
            msa_path = msa_paths[i] if msa_paths and i < len(msa_paths) else None
            
            # Create a key for grouping (sequence + msa_path)
            group_key = (protein_seq, msa_path)
            
            if group_key not in protein_groups:
                protein_groups[group_key] = {
                    'ids': [],
                    'sequence': protein_seq,
                    'msa': msa_path
                }
            protein_groups[group_key]['ids'].append(chain_id)
        
        # Add protein groups to sequences
        for group_data in protein_groups.values():
            # Use single string if only one ID, otherwise use list
            id_value = group_data['ids'][0] if len(group_data['ids']) == 1 else group_data['ids']
            
            protein_entry = {
                'protein': {
                    'id': id_value,
                    'sequence': group_data['sequence']
                }
            }
            if group_data['msa']:
                protein_entry['protein']['msa'] = group_data['msa']
            yaml_data['sequences'].append(protein_entry)
        
        # Add ligands (assuming single copy as mentioned)
        ligand_start_chain = chr(ord('A') + len(protein_sequences))
        ligand_chain_ids = []
        for i, smiles in enumerate(ligand_smiles):
            chain_id = chr(ord(ligand_start_chain) + i)
            ligand_chain_ids.append(chain_id)
            ligand_entry = {
                'ligand': {
                    'id': chain_id,  # Single string since it's single copy
                    'smiles': smiles
                }
            }
            yaml_data['sequences'].append(ligand_entry)
        
        # Add affinity prediction properties if requested
        if predict_affinity and ligand_chain_ids:
            yaml_data['properties'] = []
            for ligand_id in ligand_chain_ids:
                affinity_property = {
                    'affinity': {
                        'binder': ligand_id
                    }
                }
                yaml_data['properties'].append(affinity_property)
            
            # Log affinity prediction setup
            print(f"Added affinity prediction for {len(ligand_chain_ids)} ligand(s): {', '.join(ligand_chain_ids)}")
        
        # Write YAML file
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
        return yaml_path
    
    def generate_command(self, input_path: str, output_dir: str, 
                        combination_name: str, num_samples: int = 1) -> str:
        """
        Generate Boltz prediction command.
        
        Uses --use_msa_server if no MSA is provided in the YAML file.
        
        Args:
            input_path: Path to input file/directory
            output_dir: Directory to store outputs
            combination_name: Name for this protein-ligand combination
            num_samples: Number of samples to generate (default: 1)
        """
        method_output_dir = os.path.join(output_dir, self.name, combination_name)
        
        # Check if YAML contains MSA path
        use_msa_server = True
        try:
            with open(input_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
                
            # Check if any protein sequence has an MSA path
            for sequence in yaml_data.get('sequences', []):
                if 'protein' in sequence and 'msa' in sequence['protein']:
                    use_msa_server = False
                    break
        except Exception:
            # If we can't parse the YAML, default to using MSA server
            use_msa_server = True
        
        cmd = self.get_conda_command() + " "
        cmd += f"boltz predict {input_path} --out_dir {method_output_dir}"
        if use_msa_server:
            cmd += " --use_msa_server"
        
        # Add num_samples parameter
        cmd += f" --diffusion_samples {num_samples}"
        
        return cmd
    
    def generate_msa_command(self, input_path, output_dir, protein_name = ''):
        """
        Generate MSA generation command for Boltz.
        """
        method_output_dir = os.path.join(output_dir, self.name)
        os.makedirs(method_output_dir, exist_ok=True)
        
        # Use Boltz's MSA generation command
        cmd = f"boltz predict {input_path} --out_dir {method_output_dir} --use_msa_server --sampling_steps 10"
        
        return cmd

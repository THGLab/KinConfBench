"""
Chai co-folding method implementation.

Author: Oliver Sun
Date: 2025-05-23
"""

import os
from typing import List, Optional
from pathlib import Path
from .base import CoFoldingMethod
from .envs import CHAI_CONDA_ENV, CHAI_VENV


class Chai(CoFoldingMethod):
    """Chai co-folding method implementation."""

    def __init__(self):
        if CHAI_VENV and os.path.isdir(CHAI_VENV):
            super().__init__(name="chai", venv_path=CHAI_VENV)
        else:
            super().__init__(name="chai", conda_env=CHAI_CONDA_ENV)
    
    def prepare_input(self, protein_names: List[str], protein_sequences: List[str], 
                     ligand_names: List[str], ligand_smiles: List[str], combination_name: str,
                     input_dir: str, msa_paths: Optional[List[str]] = None) -> str:
        """
        Prepare FASTA input file and MSA directory for Chai.
        
        Format:
        >protein|<protein_name>
        <protein_sequence>
        >smiles|name=<ligand_name>
        <ligand_smiles>
        """
        # Create method-specific input directory
        method_input_dir = os.path.join(input_dir, self.name)
        os.makedirs(method_input_dir, exist_ok=True)
        
        # Create FASTA file (same format as Boltz)
        fasta_filename = f"{combination_name}.fasta"
        fasta_path = os.path.join(method_input_dir, fasta_filename)
        
        with open(fasta_path, 'w') as f:
            # Protein sections
            for i, (protein_name, protein_seq) in enumerate(zip(protein_names, protein_sequences)):
                if msa_paths and i < len(msa_paths):
                    f.write(f">protein|{protein_name}|{msa_paths[i]}\n")
                else:
                    f.write(f">protein|{protein_name}\n")
                f.write(f"{protein_seq}\n")
            
            # Ligand sections
            for name, smiles in zip(ligand_names, ligand_smiles):
                f.write(f">ligand|name={name}\n")
                f.write(f"{smiles}\n")
        
        return fasta_path
    
    def generate_command(self, input_path: str, output_dir: str, 
                        combination_name: str, num_samples: int = 1, max_sample_per_inferece = 50) -> str:
        """
        Generate Chai prediction command.
        
        If MSA is available, converts a3m to pqt first, then runs fold command.
        Otherwise uses --use-msa-server.
        
        Args:
            input_path: Path to input file/directory
            output_dir: Directory to store outputs
            combination_name: Name for this protein-ligand combination
            num_samples: Number of samples to generate (default: 1)
            max_sample_per_inferece: Maximum samples per inference (default: 50)
        """
        method_output_dir = os.path.join(output_dir, self.name, combination_name)
        input_dir = os.path.dirname(input_path)
        
        # Check if MSA directory is provided
        with open(input_path, 'r') as f:
            first_line = f.readline().strip()
            if len(first_line.split('|')) > 2: msa_folder = first_line.split('|')[2]
            else: msa_folder = None

        if msa_folder:
            # Directly use MSA directory
            with open(input_path, 'r') as f:
                lines = f.readlines()
            with open(input_path, 'w') as f:
                for line in lines:
                    if line.startswith('>protein|'):
                        # Replace MSA path with directory
                        parts = line.split('|')
                        f.write('|'.join(parts[:2]) + '\n')
                    else:
                        f.write(line)

        i = 0
        cmd = ''
        while num_samples >= max_sample_per_inferece:
            cmd += self.get_conda_command() + " "
            method_output_dir_i = os.path.join(method_output_dir, f"batch_{i}")
            if msa_folder:
                cmd += f"chai-lab fold --msa-directory {msa_folder} {input_path} {method_output_dir_i}"
            else:
                # Use native MSA server
                cmd += f"chai-lab fold --use-msa-server {input_path} {method_output_dir_i}"
            # Add num_samples parameter
            cmd += f" --num-diffn-samples {max_sample_per_inferece}\n"
            num_samples -= max_sample_per_inferece
            i += 1
        
        # Handle remaining samples
        if num_samples > 0:
            method_output_dir_i = os.path.join(method_output_dir, f"batch_{i}")
            if msa_folder:
                cmd = self.get_conda_command() + " "
                cmd += f"chai-lab fold --msa-directory {msa_folder} {input_path} {method_output_dir_i}"
            else:
                cmd = self.get_conda_command() + " "
                cmd += f"chai-lab fold --use-msa-server {input_path} {method_output_dir_i}"
            cmd += f" --num-diffn-samples {num_samples}\n"
        
        return cmd
    
    def generate_msa_command(self, input_path: str, output_dir: str, protein_name: str = '') -> str:
        """
        Generate command to create MSAs for proteins in a FASTA file.
        
        Args:
            input_path: Path to FASTA file with protein sequences
            output_dir: Directory to store MSA outputs
            
        Returns:
            Command string to generate MSAs
        """
        method_output_dir = os.path.join(output_dir, self.name, protein_name)
        
        # Use Chai's MSA generation command
        cmd = f"chai-lab fold --use-msa-server {input_path} {method_output_dir} --num-diffn-timesteps 1 --num-diffn-samples 1"

        return cmd

"""
Base class for co-folding methods.

Author: Oliver Sun
Date: 2025-05-23
"""

import os
import yaml
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path


class CoFoldingMethod(ABC):
    """Abstract base class for co-folding methods."""

    def __init__(self, name: str, conda_env: Optional[str] = None,
                 venv_path: Optional[str] = None):
        """
        Initialize the co-folding method.

        Args:
            name: Name of the method (e.g., 'boltz', 'chai', 'protenix')
            conda_env: Name of the conda environment required for this method
            venv_path: Path to a uv/Python venv. If set, takes precedence over
                conda_env and the dispatch uses a PATH prefix instead of
                `conda run`.
        """
        if not conda_env and not venv_path:
            raise ValueError(
                f"{name}: either conda_env or venv_path must be set"
            )
        self.name = name
        self.conda_env = conda_env
        self.venv_path = venv_path
        
    @abstractmethod
    def prepare_input(self, protein_names: List[str], protein_sequences: List[str], 
                     ligand_names: List[str], ligand_smiles: List[str], combination_name: str,
                     input_dir: str, msa_paths: Optional[List[str]] = None) -> str:
        """
        Prepare input files for the method.
        
        Args:
            protein_names: List of protein names
            protein_sequences: List of amino acid sequences for proteins
            ligand_names: List of ligand names
            ligand_smiles: List of SMILES strings for ligands
            combination_name: Name for this protein-ligand combination
            input_dir: Directory to store input files
            msa_paths: Optional list of paths to MSA files (one per protein)
            
        Returns:
            str: Path to the main input file/directory for this method
        """
        pass
    
    @abstractmethod
    def generate_command(self, input_path: str, output_dir: str, 
                        combination_name: str, num_samples: int = 1) -> str:
        """
        Generate the command to run inference.
        
        Args:
            input_path: Path to input file/directory
            output_dir: Directory to store outputs
            combination_name: Name for this protein-ligand combination
            num_samples: Number of samples to generate (default: 1)
            
        Returns:
            str: Command string to execute
        """
        pass

    @abstractmethod
    def generate_msa_command(self, input_path: str, output_dir: str, protein_name: str) -> str:
        """
        Generate the command to run MSA generation.
        
        Args:
            input_path: Path to input file/directory
            output_dir: Directory to store MSA outputs
            protein_name: Name of the protein for which MSA is generated
            
        Returns:
            str: Command string to execute for MSA generation
        """
        pass
    
    def get_conda_command(self) -> str:
        """Return the env-activation prefix to prepend to backend CLI calls."""
        if self.venv_path:
            return f"PATH={self.venv_path}/bin:$PATH"
        return f"conda run -n {self.conda_env}"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def resolve_combination(config: Dict[str, Any], combination: Dict[str, Any]) -> tuple:
    """
    Resolve a combination definition to actual data.
    
    Returns:
        Tuple of (protein_names, protein_sequences, ligand_smiles_list, combination_name, msa_paths)
    """
    protein_names = combination['proteins']
    ligand_names = combination['ligands']
    combination_name = "_".join(protein_names + ligand_names)
    
    # Get protein data
    protein_sequences = []
    msa_paths = []
    for protein_name in protein_names:
        protein_data = config['proteins'][protein_name]
        protein_sequences.append(protein_data['sequence'])
        msa_paths.append(protein_data.get('msa_path', None))
    
    # Get ligand SMILES
    ligand_smiles = []
    for ligand_name in ligand_names:
        ligand_smiles.append(config['ligands'][ligand_name]['smiles'])
    
    return protein_names, protein_sequences, ligand_names, ligand_smiles, combination_name, msa_paths

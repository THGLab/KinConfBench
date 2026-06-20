"""
Co-folding model post-processing utilities
"""

from typing import List, Optional, Tuple, Dict
import os
import re
import shutil
import logging
from Bio import PDB
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.Data.IUPACData import protein_letters_3to1
from utils.constants import OMITTED_LIGANDS, PHOSPHORYLATED_AAS, MODIFIED_AAS


def process_cofolding_files(cif_files: List[str], logger=None) -> List[str]:
    """
    Process a list of co-folding model CIF files to fix ligand chains and sequence representation
    
    Args:
        cif_files: List of paths to CIF files
        output_dir: Directory to save processed files (if None, will overwrite input files)
        logger: Optional logger instance
        
    Returns:
        List of paths to processed CIF files
    """
    if logger:
        logger.info(f"Processing {len(cif_files)} co-folding model files...")
    
    for cif_id, cif_path in cif_files.items():
        if not os.path.exists(cif_path):
            if logger:
                logger.warning(f"File not found: {cif_path}")
            continue
            
        try:
            # Detect model type from filename
            model_type = detect_model_type(cif_path)
            # Fix ligand chains in the mmCIF file
            processed_path = fix_ligand_chains(cif_path, None, model_type, logger)
            
            if logger:
                logger.info(f"Processed file: {os.path.basename(processed_path)}")
                
        except Exception as e:
            if logger:
                logger.error(f"Error processing {os.path.basename(cif_path)}: {str(e)}")
    
    if logger:
        logger.info(f"Completed processing {len(cif_path)} files")
    
    return 


def detect_model_type(cif_path: str) -> str:
    """
    Extract the model type from the filename by splitting on underscore and taking the first part
    
    Args:
        cif_path: Path to CIF file
        
    Returns:
        Model type (first part of the filename before underscore)
    """
    filename = os.path.basename(cif_path)
    # Split by underscore and take the first part
    parts = filename.split('_')
    if parts:
        return parts[0].lower()
    return "unknown"


def fix_ligand_chains(cif_path: str, output_path: Optional[str] = None, model_type: Optional[str] = None, logger=None) -> str:
    """
    Process mmCIF file to move ligands from separate chains to the protein chain
    for compatibility with downstream processing.
    
    Args:
        cif_path: Path to the input mmCIF file
        output_path: Path to save the modified mmCIF file (if None, will overwrite the input file)
        model_type: Type of cofolding model ('boltz', 'chai', 'protenix', or None)
        logger: Optional logger instance
        
    Returns:
        Path to the modified mmCIF file (or original if no changes were needed)
    """
    if logger:
        logger.info(f"Processing ligand chains in {os.path.basename(cif_path)}")
    
    # Use the input path as output path if not provided
    if not output_path:
        output_path = cif_path
        # Create a backup of the original file
        backup_path = os.path.join(os.path.dirname(cif_path), "raw", f"{os.path.basename(cif_path)}.bak")
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        if not os.path.exists(backup_path):
            shutil.copy(cif_path, backup_path)
    
    try:
        # Parse the mmCIF file
        parser = PDB.MMCIFParser(QUIET=True)
        structure = parser.get_structure('structure', cif_path)
        
        # Find the main protein chain and ligand chains
        main_chain_id, ligand_chains = identify_chains(structure)
        
        if not ligand_chains:
            # No separate ligand chains found
            if logger:
                logger.info(f"No separate ligand chains found in {os.path.basename(cif_path)}")
            
            # For Boltz models, we still need to fix the sequence representation
            if model_type and model_type.lower() == 'boltz':
                if logger:
                    logger.info(f"Processing Boltz model sequence representation in {os.path.basename(cif_path)}")
                return fix_boltz_sequence_representation(cif_path, output_path, logger)
            
            return cif_path
        
        # Process ligands and move them to the main chain
        structure = move_ligands_to_main_chain(structure, main_chain_id, ligand_chains, logger)
        
        # Save the modified structure
        io = PDB.MMCIFIO()
        io.set_structure(structure)
        io.save(output_path)
        
        if logger:
            logger.info(f"Saved modified structure to {os.path.basename(output_path)}")
        
        # For Boltz models, we also need to fix the sequence representation
        if model_type and model_type.lower() == 'boltz':
            if logger:
                logger.info(f"Processing Boltz model sequence representation in {os.path.basename(output_path)}")
            return fix_boltz_sequence_representation(output_path, output_path, logger)
        
        return output_path
    
    except Exception as e:
        if logger:
            logger.error(f"Error processing {os.path.basename(cif_path)}: {str(e)}")
        # Return original file path if processing failed
        return cif_path


def identify_chains(structure) -> Tuple[str, Dict[str, List]]:
    """
    Identify the main protein chain and ligand chains in a structure
    
    Args:
        structure: Biopython Structure object
        
    Returns:
        Tuple of (main_chain_id, ligand_chains_dict)
    """
    main_chain_id = None
    main_chain_residue_count = 0
    ligand_chains = {}
    
    # First pass: identify the main protein chain (the one with the most residues)
    for model in structure:
        for chain in model:
            chain_id = chain.id
            residue_count = 0
            protein_residues = 0
            
            for residue in chain:
                residue_count += 1
                if PDB.is_aa(residue):
                    protein_residues += 1
            
            # If this chain has more protein residues than the current main chain, update
            if protein_residues > main_chain_residue_count:
                main_chain_residue_count = protein_residues
                main_chain_id = chain_id
    
    # Second pass: identify ligand chains (chains with non-protein residues)
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if chain_id == main_chain_id:
                continue
            
            ligands = []
            for residue in chain:
                # Skip water molecules, common ions, and modified amino acids
                if residue.resname in OMITTED_LIGANDS or residue.resname in MODIFIED_AAS or residue.resname in PHOSPHORYLATED_AAS:
                    continue
                
                # If not a standard amino acid, consider it a ligand
                if not PDB.is_aa(residue):
                    ligands.append((residue.resname, residue.id[1]))
            
            if ligands:
                ligand_chains[chain_id] = ligands
    
    return main_chain_id, ligand_chains


def move_ligands_to_main_chain(structure, main_chain_id: str, ligand_chains: Dict[str, List], logger=None) -> PDB.Structure:
    """
    Move ligands from separate chains to the main protein chain
    
    Args:
        structure: Biopython Structure object
        main_chain_id: ID of the main protein chain
        ligand_chains: Dictionary mapping chain IDs to lists of ligand residues
        logger: Optional logger instance
        
    Returns:
        Modified structure with ligands moved to the main chain
    """
    # Create a new structure to hold the modified model
    new_structure = PDB.Structure.Structure('modified_structure')
    
    for model in structure:
        # Create a new model
        new_model = PDB.Model.Model(model.id)
        new_structure.add(new_model)
        
        # First, copy the main chain
        main_chain = None
        for chain in model:
            if chain.id == main_chain_id:
                main_chain = chain
                break
        
        if not main_chain:
            if logger:
                logger.error(f"Main chain {main_chain_id} not found in model {model.id}")
            continue
        
        # Create a new chain for the main protein
        new_chain = PDB.Chain.Chain(main_chain_id)
        new_model.add(new_chain)
        
        # Copy all residues from the main chain
        for residue in main_chain:
            new_chain.add(residue.copy())
        
        # Find the highest residue number in the main chain
        max_resid = 0
        for residue in main_chain:
            if isinstance(residue.id[1], int) and residue.id[1] > max_resid:
                max_resid = residue.id[1]
        
        # Process each ligand chain
        for chain_id, ligands in ligand_chains.items():
            chain = model[chain_id]
            
            for residue in chain:
                # Skip water molecules, common ions, and modified amino acids
                if residue.resname in OMITTED_LIGANDS or residue.resname in MODIFIED_AAS or residue.resname in PHOSPHORYLATED_AAS:
                    continue
                
                # If not a standard amino acid, consider it a ligand
                if not PDB.is_aa(residue):
                    # Create a new residue ID with 'H_' prefix for the ligand name
                    # This format is expected by classify_ligands.py
                    max_resid += 1
                    new_id = ('H_' + residue.resname, max_resid, residue.id[2])
                    
                    # Create a new residue with the modified ID
                    new_residue = PDB.Residue.Residue(new_id, residue.resname, residue.segid)
                    
                    # Copy all atoms from the original residue
                    for atom in residue:
                        new_residue.add(atom.copy())
                    
                    # Add the new residue to the main chain
                    new_chain.add(new_residue)
                    
                    if logger:
                        logger.debug(f"Moved ligand {residue.resname} from chain {chain_id} to main chain {main_chain_id}")
        
        # Add other chains that are not ligand chains (e.g., other protein chains)
        for chain in model:
            if chain.id != main_chain_id and chain.id not in ligand_chains:
                new_chain = PDB.Chain.Chain(chain.id)
                new_model.add(new_chain)
                
                for residue in chain:
                    new_chain.add(residue.copy())
    
    return new_structure


def fix_boltz_sequence_representation(cif_path: str, output_path: Optional[str] = None, logger=None) -> str:
    """
    Fix the sequence representation in Boltz model mmCIF files.
    Converts three-letter amino acid codes in the sequence field to one-letter codes.
    
    Args:
        cif_path: Path to the input mmCIF file
        output_path: Path to save the modified mmCIF file (if None, will overwrite input)
        logger: Optional logger instance
        
    Returns:
        Path to the modified mmCIF file
    """
    if not output_path:
        output_path = cif_path
    
    try:
        # Parse the mmCIF file as a dictionary
        mmcif_dict = MMCIF2Dict(cif_path)
        
        # Check if the file has entity_poly.pdbx_seq_one_letter_code field
        if '_entity_poly.pdbx_seq_one_letter_code' in mmcif_dict:
            # Get the current one-letter code sequences
            one_letter_codes = mmcif_dict['_entity_poly.pdbx_seq_one_letter_code']
            
            # Get the three-letter code sequences if available
            if '_entity_poly.pdbx_seq_three_letter_code' in mmcif_dict:
                three_letter_codes = mmcif_dict['_entity_poly.pdbx_seq_three_letter_code']
                
                # Process each sequence
                for i, three_letter_seq in enumerate(three_letter_codes):
                    if '(' in three_letter_seq:  # Check if it uses (XXX) format
                        # Extract three-letter codes from the format (XXX)(XXX)...
                        three_letter_residues = re.findall(r'\(([A-Za-z0-9]+)\)', three_letter_seq)
                        
                        # Convert to one-letter codes
                        one_letter_seq = ''
                        for res in three_letter_residues:
                            # Convert standard amino acids
                            if res.upper() in protein_letters_3to1:
                                one_letter_seq += protein_letters_3to1[res.upper()]
                            else:
                                # For non-standard residues, use X
                                one_letter_seq += 'X'
                        
                        # Update the one-letter code sequence
                        if i < len(one_letter_codes):
                            one_letter_codes[i] = one_letter_seq
            
            # Write the modified mmCIF file
            with open(cif_path, 'r') as f:
                mmcif_content = f.readlines()
            
            with open(output_path, 'w') as f:
                in_entity_poly = False
                one_letter_idx = -1
                
                for line in mmcif_content:
                    if line.startswith('_entity_poly.pdbx_seq_one_letter_code'):
                        in_entity_poly = True
                        one_letter_idx = 0
                        f.write(line)  # Write the header line
                    elif in_entity_poly and one_letter_idx >= 0 and one_letter_idx < len(one_letter_codes):
                        # Write the updated one-letter code
                        f.write(f"{one_letter_codes[one_letter_idx]}\n")
                        one_letter_idx += 1
                    else:
                        f.write(line)
            
            if logger:
                logger.info(f"Updated sequence representation in {os.path.basename(output_path)}")
        
        return output_path
    
    except Exception as e:
        if logger:
            logger.error(f"Error fixing Boltz sequence representation in {os.path.basename(cif_path)}: {str(e)}")
        return cif_path

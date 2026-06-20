"""
Unified CIF extractor that consolidates sequence, ligand, phosphorylation, 
and mutation extraction from mmCIF files using native metadata.
"""

import gzip
import re
import sys
from typing import Dict, List, Optional, Tuple, Any
from Bio import PDB
from Bio.PDB import MMCIFParser, MMCIF2Dict
from Bio.SeqUtils import seq1
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Ensure the parent directory is in the path
from utils.constants import OMITTED_LIGANDS, MODIFIED_AAS, PHOSPHORYLATED_AAS


class MMCIFExtractor:
    """
    Unified extractor for mmCIF files that extracts all relevant information
    for kinase analysis in a single pass.
    """
    
    def __init__(self, logger=None):
        self.parser = MMCIFParser(QUIET=True)
        self.logger = logger
    
    def _open_file(self, filename: str):
        """Get file handle for gzipped or regular files."""
        if filename.endswith(".gz"):
            return gzip.open(filename, "rt")
        else:
            return open(filename, "r")
    
    def extract_all_chains(self, cif_filepath: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract all information for all protein chains in the CIF file.
        
        Args:
            cif_filepath: Path to the CIF file
            
        Returns:
            Dictionary with chain_id as key and extracted data as value
        """
        try:
            # Parse CIF file
            with self._open_file(cif_filepath) as file:
                mmcif_dict = MMCIF2Dict.MMCIF2Dict(file)
            
            with self._open_file(cif_filepath) as file:
                structure = self.parser.get_structure("structure_id", file)
            
            # Get all protein chains
            protein_chains = self._identify_protein_chains(mmcif_dict, structure)
            
            # Extract data for each chain
            chain_data = {}
            for chain_id in protein_chains:
                chain_data[chain_id] = self._extract_chain_data(
                    mmcif_dict, structure, chain_id, cif_filepath
                )
            
            return chain_data
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing CIF file {cif_filepath}: {e}")
            else:
                print(f"Error processing CIF file {cif_filepath}: {e}")
            return {}
    
    def extract_basic_chains(self, cif_filepath: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract only basic information (sequence, s2c_mapping, ligands) for co-folding structures.
        This is a lightweight version for structures that don't need full PDB metadata.
        
        Args:
            cif_filepath: Path to the CIF file
            
        Returns:
            Dictionary with chain_id as key and basic extracted data as value
        """
        try:
            # Parse CIF file
            with self._open_file(cif_filepath) as file:
                mmcif_dict = MMCIF2Dict.MMCIF2Dict(file)
            
            with self._open_file(cif_filepath) as file:
                structure = self.parser.get_structure("structure_id", file)
            
            # Get all protein chains
            protein_chains = self._identify_protein_chains(mmcif_dict, structure)
            
            # Extract basic data for each chain
            chain_data = {}
            for chain_id in protein_chains:
                chain_data[chain_id] = self._extract_basic_chain_data(
                    mmcif_dict, structure, chain_id
                )
            
            return chain_data
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing CIF file {cif_filepath}: {e}")
            else:
                print(f"Error processing CIF file {cif_filepath}: {e}")
            return {}
    
    def _identify_protein_chains(self, mmcif_dict: Dict, structure) -> List[str]:
        """Identify all protein chains in the structure."""
        protein_chains = set()
        
        # From entity_poly information
        if "_entity_poly.pdbx_strand_id" in mmcif_dict:
            chain_list = mmcif_dict["_entity_poly.pdbx_strand_id"]
            entity_types = mmcif_dict.get("_entity_poly.type", [])
            
            for i, chains_str in enumerate(chain_list):
                if i < len(entity_types) and "polypeptide" in entity_types[i].lower():
                    chains = chains_str.split(",")
                    for chain in chains:
                        chain_id = chain.strip()
                        # Check if chain is long enough to be a kinase
                        if self._is_kinase_length_chain(structure, chain_id):
                            protein_chains.add(chain_id)
        
        # Fallback: check structure directly
        if not protein_chains:
            for model in structure:
                for chain in model:
                    # Check if chain has amino acid residues and is long enough
                    if self._is_kinase_length_chain(structure, chain.id):
                        protein_chains.add(chain.id)
                break  # Only check first model
        
        return list(protein_chains)
    
    def _is_kinase_length_chain(self, structure, chain_id: str) -> bool:
        """Check if a chain has >= 20 amino acid residues (kinase length)."""
        for model in structure:
            if chain_id in model:
                chain = model[chain_id]
                aa_count = sum(1 for residue in chain if PDB.is_aa(residue))
                return aa_count >= 20
        return False
    
    def _extract_peptide_sequence_with_modifications(self, peptide_chain, mmcif_dict: Dict, chain_id: str) -> str:
        """Extract peptide sequence with proper handling of modified amino acids."""
        sequence_parts = []
        
        # First try to get sequence from mmCIF metadata if available
        if mmcif_dict and "_entity_poly.pdbx_seq_one_letter_code" in mmcif_dict:
            chain_list = mmcif_dict["_entity_poly.pdbx_strand_id"]
            sequence_list = mmcif_dict["_entity_poly.pdbx_seq_one_letter_code"]
            
            for i, chains_str in enumerate(chain_list):
                chains = [c.strip() for c in chains_str.split(",")]
                if chain_id in chains and i < len(sequence_list):
                    # Get the raw sequence with modifications in parentheses
                    raw_sequence = sequence_list[i]
                    raw_sequence = ''.join(raw_sequence.splitlines())
                    return raw_sequence  # Return as-is with modifications like (ZXW)
        
        # Fallback: extract from coordinates
        for residue in peptide_chain:
            if PDB.is_aa(residue):
                res_name = residue.get_resname()
                if res_name in MODIFIED_AAS or res_name in PHOSPHORYLATED_AAS:
                    # For modified residues, just show the modification in parentheses
                    sequence_parts.append(f"({res_name})")
                else:
                    sequence_parts.append(seq1(res_name))
        
        return ''.join(sequence_parts)
    
    def _extract_peptide_ligands(self, structure, chain_id: str, mmcif_dict: Dict = None) -> List[str]:
        """Extract short peptide chains (< 30 AA) as ligands if they interact with the target chain."""
        peptide_ligands = []
        
        for model in structure:
            if chain_id not in model:
                continue
                
            target_chain = model[chain_id]
            
            # Look for short peptide chains
            for chain in model:
                if chain.id == chain_id:  # Skip the target chain itself
                    continue
                    
                # Check if this is a short peptide chain
                aa_residues = [residue for residue in chain if PDB.is_aa(residue)]
                if len(aa_residues) < 30:  # Short peptide (1-29 amino acids)
                    # Check if peptide interacts with target chain
                    if self._peptide_interacts_with_chain(chain, target_chain):
                        # Extract peptide sequence with proper handling of modified residues
                        peptide_sequence = self._extract_peptide_sequence_with_modifications(chain, mmcif_dict, chain.id)
                        peptide_entry = f"peptide_{chain.id}:{peptide_sequence}"
                        peptide_ligands.append(peptide_entry)
            break  # Only process first model
        
        return peptide_ligands
    
    def _peptide_interacts_with_chain(self, peptide_chain, target_chain, cutoff: float = 4.0) -> bool:
        """Check if peptide chain makes contact with target chain within cutoff distance."""
        for peptide_residue in peptide_chain:
            if PDB.is_aa(peptide_residue):
                for peptide_atom in peptide_residue:
                    for protein_residue in target_chain:
                        if PDB.is_aa(protein_residue):
                            for protein_atom in protein_residue:
                                try:
                                    distance = peptide_atom - protein_atom
                                    if distance <= cutoff:
                                        return True
                                except:
                                    continue
        return False
    
    def _extract_basic_chain_data(self, mmcif_dict: Dict, structure, chain_id: str) -> Dict[str, Any]:
        """Extract only basic information for a specific chain (sequence, s2c_mapping, ligands)."""
        data = {
            'sequence': None,
            'first_residue': None,
            's2c_mapping': None,
            'ligands': [],
            'metadata': {
                'chain_length': 0,
                'has_coordinates': False,
                'sequence_source': None,
                'extraction_type': 'basic'
            }
        }
        
        # Extract sequence information (basic version)
        sequence_data = self._extract_sequence_data(mmcif_dict, structure, chain_id)
        data['sequence'] = sequence_data['sequence']
        data['first_residue'] = sequence_data['first_residue']
        data['s2c_mapping'] = sequence_data['s2c_mapping']
        data['metadata']['sequence_source'] = sequence_data['metadata']['sequence_source']
        
        # Extract ligand information
        ligands = self._extract_ligands(structure, chain_id, mmcif_dict)
        data['ligands'] = ligands
        
        # Update metadata
        data['metadata']['chain_length'] = len(data['sequence']) if data['sequence'] else 0
        data['metadata']['has_coordinates'] = self._has_coordinates(structure, chain_id)
        
        return data
    
    def _extract_chain_data(self, mmcif_dict: Dict, structure, chain_id: str, 
                           cif_filepath: str) -> Dict[str, Any]:
        """Extract all data for a specific chain."""
        data = {
            'sequence': None,
            'first_residue': None,
            's2c_mapping': None,
            'ligands': [],
            'gene_name': None,
            'uniprot_id': None,
            'reference_sequence': None,
            'alignment': None,
            'phosphorylations': [],
            'mutations': [],
            'metadata': {
                'chain_length': 0,
                'model_id': 0,
                'has_coordinates': False,
                'sequence_source': None
            }
        }
        
        # Extract sequence information
        sequence_data = self._extract_sequence_data(mmcif_dict, structure, chain_id)
        data.update(sequence_data)
        
        # Extract ligand information
        ligands = self._extract_ligands(structure, chain_id, mmcif_dict)
        data['ligands'] = ligands
        
        # Extract covalent linkage information
        covalent_data = self._extract_covalent_linkages(mmcif_dict, ligands, chain_id)
        data['is_covalent_binder'] = covalent_data['is_covalent_binder']
        data['covalent_info'] = covalent_data['covalent_info']
        
        # Extract UniProt information
        uniprot_data = self._extract_uniprot_data(mmcif_dict, chain_id)
        data.update(uniprot_data)
        
        # Extract phosphorylations and mutations
        modifications = self._extract_modifications(structure, chain_id, mmcif_dict)
        data['phosphorylations'] = modifications['phosphorylations']
        data['mutations'] = modifications['mutations']
        
        # Update metadata
        data['metadata']['chain_length'] = len(data['sequence']) if data['sequence'] else 0
        data['metadata']['has_coordinates'] = self._has_coordinates(structure, chain_id)
        
        return data
    
    def _extract_sequence_data(self, mmcif_dict: Dict, structure, chain_id: str) -> Dict[str, Any]:
        """Extract sequence and related data."""
        sequence = None
        first_res = None
        s2c_mapping = None
        sequence_source = None
        
        # Try to get sequence from CIF metadata first
        if "_entity_poly.pdbx_seq_one_letter_code" in mmcif_dict:
            chain_list = mmcif_dict["_entity_poly.pdbx_strand_id"]
            sequence_list = mmcif_dict["_entity_poly.pdbx_seq_one_letter_code"]
            
            for i, chains_str in enumerate(chain_list):
                chains = [c.strip() for c in chains_str.split(",")]
                if chain_id in chains and i < len(sequence_list):
                    sequence = sequence_list[i]
                    sequence = ''.join(sequence.splitlines())
                    # Replace modified residues with X
                    sequence = re.sub(r'\([A-Z0-9][A-Z0-9][A-Z0-9]\)', "X", sequence)
                    sequence_source = "cif_metadata"
                    break
        
        # Get sequence from coordinates
        sequence_from_coords, first_res = self._extract_sequence_from_coordinates(
            structure, chain_id
        )
        
        # Build sequence-to-coordinate mapping
        s2c_mapping = self._build_s2c_mapping(mmcif_dict, chain_id)
        
        # Use metadata sequence if available, otherwise use coordinate sequence
        if sequence is None:
            sequence = sequence_from_coords
            sequence_source = "coordinates"
        else:
            # Align sequences to get proper first_res
            if sequence_from_coords:
                frag_index = self._align_sequences(sequence, sequence_from_coords)
                if frag_index >= 0:
                    first_res = first_res - frag_index
        
        return {
            'sequence': sequence,
            'first_residue': first_res,
            's2c_mapping': s2c_mapping,
            'metadata': {'sequence_source': sequence_source}
        }
    
    def _extract_sequence_from_coordinates(self, structure, chain_id: str) -> Tuple[str, int]:
        """Extract sequence from atomic coordinates."""
        sequence = []
        first_res = 0
        prev_residue_num = None
        
        for model in structure:
            if chain_id in model:
                chain = model[chain_id]
                for residue in chain:
                    if PDB.is_aa(residue):
                        curr_residue_num = residue.get_id()[1]
                        if curr_residue_num > 50000:  # Skip PDBrenum files
                            continue
                        
                        if prev_residue_num is not None:
                            missing_residues = curr_residue_num - prev_residue_num - 1
                            sequence.extend(["X"] * missing_residues)
                        
                        sequence.append(seq1(residue.get_resname()))
                        prev_residue_num = curr_residue_num
                        
                        if len(sequence) == 1:  # First residue
                            first_res = curr_residue_num
                break  # Only process first model
        
        return ''.join(sequence), first_res
    
    def _build_s2c_mapping(self, mmcif_dict: Dict, chain_id: str) -> Optional[Dict[int, int]]:
        """Build sequence-to-coordinate mapping from CIF metadata."""
        if "_pdbx_poly_seq_scheme.pdb_seq_num" not in mmcif_dict:
            return None
        
        s2c_dict = {}
        chain_ids = mmcif_dict["_pdbx_poly_seq_scheme.pdb_strand_id"]
        seq_ids = mmcif_dict["_pdbx_poly_seq_scheme.seq_id"]
        pdb_seq_nums = mmcif_dict["_pdbx_poly_seq_scheme.pdb_seq_num"]
        
        for i, cid in enumerate(chain_ids):
            if cid == chain_id and i < len(seq_ids) and i < len(pdb_seq_nums):
                try:
                    seq_id = int(seq_ids[i])
                    pdb_seq_num = int(pdb_seq_nums[i])
                    s2c_dict[seq_id] = pdb_seq_num
                except (ValueError, TypeError):
                    continue
        
        return s2c_dict if s2c_dict else None
    
    def _align_sequences(self, seq1: str, seq2: str) -> int:
        """Simple sequence alignment to find offset."""
        if not seq1 or not seq2:
            return -1
        
        # Try with first 10 residues
        frag = seq2[:10].replace("X", ".")
        match = re.search(frag, seq1)
        if match:
            return match.start()
        
        # Try with first 5 residues
        frag = seq2[:5].replace("X", ".")
        match = re.search(frag, seq1)
        if match:
            return match.start()
        
        return -1
    
    def _extract_ligands(self, structure, chain_id: str, mmcif_dict: Dict = None) -> List[str]:
        """Extract ligands that interact with the specified chain.
        
        First tries to extract from mmCIF metadata (_pdbx_entity_nonpoly),
        then falls back to structure search if no ligands found.
        """
        ligands = []
        
        # First try: Extract from mmCIF metadata
        if mmcif_dict and "_pdbx_entity_nonpoly.entity_id" in mmcif_dict:
            ligands = self._extract_ligands_from_mmcif(mmcif_dict, chain_id)
            if self.logger:
                self.logger.debug(f"Extracted {ligands} ligands from mmCIF metadata for chain {chain_id}")
            else:
                print(f"Extracted {ligands} ligands from mmCIF metadata for chain {chain_id}")
        
        # Fallback: Extract from structure if no ligands found in metadata
        if not ligands:
            ligands = self._extract_ligands_from_structure(structure, chain_id)
            if self.logger:
                self.logger.debug(f"Extracted {len(ligands)} ligands from structure coordinates for chain {chain_id}")
            else:
                print(f"Extracted {len(ligands)} ligands from structure coordinates for chain {chain_id}")
        # Add short peptide chains as ligands
        peptide_ligands = self._extract_peptide_ligands(structure, chain_id, mmcif_dict)
        ligands.extend(peptide_ligands)
        
        return ligands if ligands else ["No_ligand"]
    
    def _extract_ligands_from_mmcif(self, mmcif_dict: Dict, chain_id: str) -> List[str]:
        """Extract ligands from mmCIF metadata using _pdbx_entity_nonpoly."""
        ligands = []
        
        try:
            if "_pdbx_entity_nonpoly.entity_id" not in mmcif_dict:
                return ligands
            
            entity_ids = mmcif_dict["_pdbx_entity_nonpoly.entity_id"]
            comp_ids = mmcif_dict.get("_pdbx_entity_nonpoly.comp_id", [])
            names = mmcif_dict.get("_pdbx_entity_nonpoly.name", [])

            ligand_entity_list = []
            for i, entity_id in enumerate(entity_ids):
                if i < len(comp_ids):
                    comp_id = comp_ids[i]
                    
                    # Skip common solvents, ions, and modified amino acids
                    if comp_id in OMITTED_LIGANDS or comp_id in MODIFIED_AAS:
                        continue
                    ligand_entity_list.append(entity_id)
            
            # Try to get residue numbers from _pdbx_nonpoly_scheme if available
            ligand_residue_nums = {}
            if "_pdbx_nonpoly_scheme.entity_id" in mmcif_dict:
                scheme_entity_ids = mmcif_dict["_pdbx_nonpoly_scheme.entity_id"]
                scheme_pdb_seq_nums = mmcif_dict.get("_pdbx_nonpoly_scheme.pdb_seq_num", [])
                scheme_pdb_strand_ids = mmcif_dict.get("_pdbx_nonpoly_scheme.pdb_strand_id", [])
                
                for i, ent_id in enumerate(scheme_entity_ids):
                    if ent_id in ligand_entity_list and scheme_pdb_strand_ids[i] == chain_id:
                        try:
                            ligand_residue_nums[ent_id] = int(scheme_pdb_seq_nums[i])
                        except (ValueError, TypeError):
                            continue
            
            # Extract non-polymer entities (ligands)
            for i, entity_id in enumerate(entity_ids):
                if i < len(comp_ids):
                    comp_id = comp_ids[i]
                    
                    # Skip common solvents, ions, and modified amino acids
                    if comp_id in OMITTED_LIGANDS or comp_id in MODIFIED_AAS:
                        continue
                    
                    # Use actual residue number if available, otherwise default to 1
                    residue_num = ligand_residue_nums.get(entity_id, 1)
                    ligand_entry = f"{comp_id}:{residue_num}"
                    if ligand_entry not in ligands:
                        ligands.append(ligand_entry)
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error extracting ligands from mmCIF metadata: {e}")
            else:
                print(f"Error extracting ligands from mmCIF metadata: {e}")
        
        return ligands
    
    def _extract_ligands_from_structure(self, structure, chain_id: str) -> List[str]:
        """Extract ligands from structure coordinates (fallback method)."""
        ligands = []
        
        for model in structure:
            if chain_id not in model:
                continue
                
            target_chain = model[chain_id]
            
            # Look for heteroatoms in all chains
            for chain in model:
                for residue in chain:
                    if 'H_' in residue.id[0]:  # Heteroatom
                        ligand_name = residue.id[0][2:].strip()
                        ligand_id = residue.id[1]  # This is the actual residue number
                        
                        # Skip common solvents and modified amino acids
                        if ligand_name in OMITTED_LIGANDS or ligand_name in MODIFIED_AAS:
                            continue
                        
                        # Check if ligand interacts with target chain
                        if self._ligand_interacts_with_chain(residue, target_chain):
                            ligand_entry = f"{ligand_name}:{ligand_id}"
                            if ligand_entry not in ligands:
                                ligands.append(ligand_entry)
            break  # Only process first model
        
        return ligands
    
    def _ligand_interacts_with_chain(self, ligand_residue, target_chain, cutoff: float = 4.0) -> bool:
        """Check if ligand makes contact with target chain within cutoff distance."""
        for ligand_atom in ligand_residue:
            for protein_residue in target_chain:
                if PDB.is_aa(protein_residue):
                    for protein_atom in protein_residue:
                        try:
                            distance = ligand_atom - protein_atom
                            if distance <= cutoff:
                                return True
                        except:
                            continue
        return False
    
    def _extract_uniprot_data(self, mmcif_dict: Dict, chain_id: str) -> Dict[str, Any]:
        """Extract UniProt ID and reference sequence from CIF metadata."""
        gene_name = None
        uniprot_id = None
        reference_sequence = None
        alignment = None
        
        try:
            # Extract UniProt ID from struct_ref
            if "_struct_ref.db_name" in mmcif_dict:
                db_names = mmcif_dict["_struct_ref.db_name"]
                db_codes = mmcif_dict.get("_struct_ref.db_code", [])
                uniprot_ids = mmcif_dict.get("_struct_ref.pdbx_db_accession", [])
                ref_sequences = mmcif_dict.get("_struct_ref.pdbx_seq_one_letter_code", [])
                
                # Ensure db_names is a list
                if isinstance(db_names, str):
                    db_names = [db_names]
                if isinstance(db_codes, str):
                    db_codes = [db_codes]
                if isinstance(uniprot_ids, str):
                    uniprot_ids = [uniprot_ids]
                if isinstance(ref_sequences, str):
                    ref_sequences = [ref_sequences]
                
                for i, db_name in enumerate(db_names):
                    if db_name == "UNP" and i < len(db_codes):
                        # Check if this reference applies to our chain
                        if self._reference_applies_to_chain(mmcif_dict, i + 1, chain_id):
                            gene_name = db_codes[i] if i < len(db_codes) else None
                            uniprot_id = uniprot_ids[i] if i < len(uniprot_ids) else None
                            if i < len(ref_sequences):
                                reference_sequence = ''.join(ref_sequences[i].splitlines())
                            break
            
            # Extract alignment information
            if uniprot_id:
                alignment = self._extract_alignment_info(mmcif_dict, chain_id)
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error extracting UniProt data: {e}")
            else:
                print(f"Error extracting UniProt data: {e}")
        
        return {
            'gene_name': gene_name,
            'uniprot_id': uniprot_id,
            'reference_sequence': reference_sequence,
            'alignment': alignment
        }
    
    def _reference_applies_to_chain(self, mmcif_dict: Dict, ref_id: int, chain_id: str) -> bool:
        """Check if a struct_ref entry applies to the given chain."""
        if "_struct_ref_seq.pdbx_strand_id" not in mmcif_dict:
            return False
        
        ref_ids = mmcif_dict.get("_struct_ref_seq.ref_id", [])
        strand_ids = mmcif_dict["_struct_ref_seq.pdbx_strand_id"]
        
        for i, rid in enumerate(ref_ids):
            if str(rid) == str(ref_id) and i < len(strand_ids):
                if strand_ids[i] == chain_id:
                    return True
        return False
    
    def _extract_alignment_info(self, mmcif_dict: Dict, chain_id: str) -> Optional[Dict[str, Any]]:
        """Extract alignment information from struct_ref_seq."""
        if "_struct_ref_seq.pdbx_strand_id" not in mmcif_dict:
            return None
        
        strand_ids = mmcif_dict["_struct_ref_seq.pdbx_strand_id"]
        seq_align_begs = mmcif_dict.get("_struct_ref_seq.seq_align_beg", [])
        seq_align_ends = mmcif_dict.get("_struct_ref_seq.seq_align_end", [])
        db_align_begs = mmcif_dict.get("_struct_ref_seq.db_align_beg", [])
        db_align_ends = mmcif_dict.get("_struct_ref_seq.db_align_end", [])
        
        for i, sid in enumerate(strand_ids):
            if sid == chain_id:
                try:
                    return {
                        'pdb_start': int(seq_align_begs[i]) if i < len(seq_align_begs) else None,
                        'pdb_end': int(seq_align_ends[i]) if i < len(seq_align_ends) else None,
                        'uniprot_start': int(db_align_begs[i]) if i < len(db_align_begs) else None,
                        'uniprot_end': int(db_align_ends[i]) if i < len(db_align_ends) else None,
                    }
                except (ValueError, TypeError, IndexError):
                    continue
        
        return None
    
    def _extract_modifications(self, structure, chain_id: str, mmcif_dict: Dict = None) -> Dict[str, List[Dict[str, Any]]]:
        """Extract phosphorylations and mutations from mmCIF metadata first, then structure.
        
        First tries to extract from mmCIF metadata (_pdbx_modification_feature and _entity.pdbx_mutation),
        then falls back to structure search if no modifications found.
        """
        phosphorylations = []
        mutations = []
        
        # First try: Extract from mmCIF metadata
        if mmcif_dict:
            mmcif_modifications = self._extract_modifications_from_mmcif(mmcif_dict, chain_id)
            phosphorylations.extend(mmcif_modifications['phosphorylations'])
            mutations.extend(mmcif_modifications['mutations'])
        
        # Fallback: Extract from structure if no modifications found in metadata
        if not phosphorylations and not mutations:
            structure_modifications = self._extract_modifications_from_structure(structure, chain_id)
            phosphorylations.extend(structure_modifications['phosphorylations'])
            mutations.extend(structure_modifications['mutations'])
        
        return {
            'phosphorylations': phosphorylations,
            'mutations': mutations
        }
    
    def _extract_modifications_from_mmcif(self, mmcif_dict: Dict, chain_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract modifications from mmCIF metadata using _pdbx_modification_feature and _entity.pdbx_mutation."""
        phosphorylations = []
        mutations = []
        
        try:
            # Extract from _pdbx_modification_feature (for phosphorylations and other modifications)
            if "_pdbx_modification_feature.type" in mmcif_dict:
                feature_types = mmcif_dict["_pdbx_modification_feature.type"]
                auth_comp_ids = mmcif_dict.get("_pdbx_modification_feature.auth_comp_id", [])
                auth_asym_ids = mmcif_dict.get("_pdbx_modification_feature.auth_asym_id", [])
                auth_seq_ids = mmcif_dict.get("_pdbx_modification_feature.auth_seq_id", [])
                ref_comp_ids = mmcif_dict.get("_pdbx_modification_feature.ref_comp_id", [])
                
                for i, feature_type in enumerate(feature_types):
                    # Check if this modification applies to our chain
                    if i < len(auth_asym_ids) and auth_asym_ids[i] == chain_id:
                        auth_comp_id = auth_comp_ids[i] if i < len(auth_comp_ids) else None
                        seq_id = auth_seq_ids[i] if i < len(auth_seq_ids) else None
                        ref_comp_id = ref_comp_ids[i] if i < len(ref_comp_ids) else None
                        
                        # Check if it's a phosphorylation
                        if (feature_type.lower() in ['phosphorylation', 'modified residue'] or 
                            (auth_comp_id and auth_comp_id in PHOSPHORYLATED_AAS)):
                            
                            phospho_info = {
                                'residue': ref_comp_id if ref_comp_id else auth_comp_id,
                                'position': None,  # Would need sequence mapping
                                'pdb_residue_num': int(seq_id) if seq_id else None,
                                'uniprot_position': None,
                                'modification_type': auth_comp_id or feature_type,
                                'description': self._get_phospho_description(auth_comp_id) if auth_comp_id else feature_type
                            }
                            phosphorylations.append(phospho_info)
            
            # Extract from _entity.pdbx_mutation (for sequence mutations)
            if "_entity.pdbx_mutation" in mmcif_dict:
                entity_ids = mmcif_dict.get("_entity.id", [])
                mutation_descriptions = mmcif_dict["_entity.pdbx_mutation"]
                
                # Find entity ID for our chain
                target_entity_id = self._get_entity_id_for_chain(mmcif_dict, chain_id)
                
                # Ensure mutation_descriptions is a list
                if isinstance(mutation_descriptions, str):
                    mutation_descriptions = [mutation_descriptions]
                
                for i, entity_id in enumerate(entity_ids):
                    if (target_entity_id and str(entity_id) == str(target_entity_id) and 
                        i < len(mutation_descriptions) and 
                        mutation_descriptions[i] not in ['?', '.', 'None', '']):
                        
                        # Just store the mutation description as a simple string
                        mutations.extend(mutation_descriptions[i].split(', '))
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error extracting modifications from mmCIF metadata: {e}")
            else:
                print(f"Error extracting modifications from mmCIF metadata: {e}")
        
        return {
            'phosphorylations': phosphorylations,
            'mutations': mutations
        }
    
    def _extract_modifications_from_structure(self, structure, chain_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract modifications from structure coordinates (fallback method)."""
        phosphorylations = []
        mutations = []
        
        for model in structure:
            if chain_id not in model:
                continue
                
            chain = model[chain_id]
            position = 1
            
            for residue in chain:
                if PDB.is_aa(residue):
                    res_name = residue.get_resname()
                    pdb_num = residue.get_id()[1]
                    
                    # Check for phosphorylations
                    if res_name in PHOSPHORYLATED_AAS:
                        phospho_info = {
                            'residue': res_name,
                            'position': position,
                            'pdb_residue_num': pdb_num,
                            'uniprot_position': None,  # Would need alignment mapping
                            'modification_type': res_name,
                            'description': self._get_phospho_description(res_name)
                        }
                        phosphorylations.append(phospho_info)
                    
                    # Check for mutations (modified amino acids)
                    elif res_name in MODIFIED_AAS:
                        mutation_info = {
                            'pdb_residue': res_name,
                            'position': position,
                            'pdb_residue_num': pdb_num,
                            'uniprot_position': None,  # Would need alignment mapping
                            'mutation_type': 'modified_amino_acid',
                        }
                        mutations.append(mutation_info)
                    
                    position += 1
            break  # Only process first model
        
        return {
            'phosphorylations': phosphorylations,
            'mutations': mutations
        }
    
    def _get_entity_id_for_chain(self, mmcif_dict: Dict, chain_id: str) -> Optional[str]:
        """Get entity ID for a given chain ID."""
        if "_entity_poly.pdbx_strand_id" in mmcif_dict:
            poly_entity_ids = mmcif_dict["_entity_poly.entity_id"]
            poly_strand_ids = mmcif_dict["_entity_poly.pdbx_strand_id"]
            
            for i, strand_ids_str in enumerate(poly_strand_ids):
                chains = [c.strip() for c in strand_ids_str.split(",")]
                if chain_id in chains and i < len(poly_entity_ids):
                    return poly_entity_ids[i]
        
        return None
    
    def _get_phospho_description(self, res_name: str) -> str:
        """Get description for phosphorylated residue."""
        descriptions = {
            'SEP': 'phosphoserine',
            'TPO': 'phosphothreonine',
            'PTR': 'phosphotyrosine',
            'PHD': 'phosphohistidine',
            'NED': 'phosphoaspartate',
        }
        return descriptions.get(res_name, f'phosphorylated {res_name}')
    
    def _has_coordinates(self, structure, chain_id: str) -> bool:
        """Check if chain has coordinate data."""
        for model in structure:
            if chain_id in model:
                chain = model[chain_id]
                return any(PDB.is_aa(residue) for residue in chain)
        return False
    
    def _extract_covalent_linkages(self, mmcif_dict: Dict, ligands: List[str], chain_id: str) -> Dict[str, Any]:
        """
        Extract covalent linkage information from mmCIF _struct_conn data.
        
        Args:
            mmcif_dict: Parsed mmCIF dictionary
            ligands: List of ligands in format ["LIGAND:residue_num", ...]
            chain_id: Current chain ID being processed
            
        Returns:
            Dictionary with is_covalent_binder (bool) and covalent_info (list of tuples)
        """
        covalent_info = {
            'is_covalent_binder': False,
            'covalent_info': []
        }
        
        # Extract ligand identifiers and their chain associations from the ligands list
        ligand_comp_ids = set()
        peptide_chain_ids = set()
        ligand_chain_mapping = {}  # Maps ligand comp_id to the chain it's associated with
        
        for ligand in ligands:
            if ligand != "No_ligand":
                if ligand.startswith("peptide_"):
                    # Extract peptide chain ID: "peptide_B:sequence" -> "B"
                    peptide_chain_id = ligand.split(':')[0].replace('peptide_', '')
                    peptide_chain_ids.add(peptide_chain_id)
                else:
                    # Regular ligand: "LIGAND:residue_num" -> "LIGAND"
                    comp_id = ligand.split(':')[0]
                    ligand_comp_ids.add(comp_id)
                    # Map this ligand to the current kinase chain (since it was extracted for this chain)
                    ligand_chain_mapping[comp_id] = chain_id
        
        if not ligand_comp_ids and not peptide_chain_ids:
            return covalent_info
        
        try:
            # Check if _struct_conn data exists
            if "_struct_conn.conn_type_id" not in mmcif_dict:
                return covalent_info
            
            conn_types = mmcif_dict["_struct_conn.conn_type_id"]
            
            # Ensure conn_types is a list
            if isinstance(conn_types, str):
                conn_types = [conn_types]
            
            # Get all relevant fields - using label_asym_ids as requested
            ptnr1_label_comp_ids = mmcif_dict.get("_struct_conn.ptnr1_label_comp_id", [])
            ptnr2_label_comp_ids = mmcif_dict.get("_struct_conn.ptnr2_label_comp_id", [])
            ptnr1_auth_comp_ids = mmcif_dict.get("_struct_conn.ptnr1_auth_comp_id", [])
            ptnr2_auth_comp_ids = mmcif_dict.get("_struct_conn.ptnr2_auth_comp_id", [])
            ptnr1_auth_seq_ids = mmcif_dict.get("_struct_conn.ptnr1_auth_seq_id", [])
            ptnr2_auth_seq_ids = mmcif_dict.get("_struct_conn.ptnr2_auth_seq_id", [])
            ptnr1_label_atom_ids = mmcif_dict.get("_struct_conn.ptnr1_label_atom_id", [])
            ptnr2_label_atom_ids = mmcif_dict.get("_struct_conn.ptnr2_label_atom_id", [])
            ptnr1_label_asym_ids = mmcif_dict.get("_struct_conn.ptnr1_label_asym_id", [])
            ptnr2_label_asym_ids = mmcif_dict.get("_struct_conn.ptnr2_label_asym_id", [])
            
            # Ensure all fields are lists
            for field_name, field_data in [
                ("ptnr1_label_comp_ids", ptnr1_label_comp_ids),
                ("ptnr2_label_comp_ids", ptnr2_label_comp_ids),
                ("ptnr1_auth_comp_ids", ptnr1_auth_comp_ids),
                ("ptnr2_auth_comp_ids", ptnr2_auth_comp_ids),
                ("ptnr1_auth_seq_ids", ptnr1_auth_seq_ids),
                ("ptnr2_auth_seq_ids", ptnr2_auth_seq_ids),
                ("ptnr1_label_atom_ids", ptnr1_label_atom_ids),
                ("ptnr2_label_atom_ids", ptnr2_label_atom_ids),
                ("ptnr1_label_asym_ids", ptnr1_label_asym_ids),
                ("ptnr2_label_asym_ids", ptnr2_label_asym_ids)
            ]:
                if isinstance(field_data, str):
                    locals()[field_name] = [field_data]
            
            # Process each connection
            for i, conn_type in enumerate(conn_types):
                if conn_type.lower() == "covale":
                    # Get component IDs and chain IDs for both partners using label_asym_ids
                    ptnr1_comp = ptnr1_label_comp_ids[i] if i < len(ptnr1_label_comp_ids) else None
                    ptnr2_comp = ptnr2_label_comp_ids[i] if i < len(ptnr2_label_comp_ids) else None
                    ptnr1_chain = ptnr1_label_asym_ids[i] if i < len(ptnr1_label_asym_ids) else None
                    ptnr2_chain = ptnr2_label_asym_ids[i] if i < len(ptnr2_label_asym_ids) else None
                    
                    # Create set of all relevant chains for this kinase (kinase chain + its peptide ligands)
                    relevant_chains = {chain_id} | peptide_chain_ids
                    
                    # Check if this covalent connection involves our kinase system
                    # At least one partner must be our ligand/peptide AND at least one must be in relevant chains
                    ptnr1_is_our_ligand = ptnr1_comp in ligand_comp_ids or ptnr1_chain in peptide_chain_ids
                    ptnr2_is_our_ligand = ptnr2_comp in ligand_comp_ids or ptnr2_chain in peptide_chain_ids
                    ptnr1_in_relevant_chains = ptnr1_chain in relevant_chains
                    ptnr2_in_relevant_chains = ptnr2_chain in relevant_chains
                    
                    # Include if: (has our ligand) AND (involves our chain system)
                    has_our_ligand = ptnr1_is_our_ligand or ptnr2_is_our_ligand
                    involves_our_system = ptnr1_in_relevant_chains or ptnr2_in_relevant_chains
                    
                    if has_our_ligand and involves_our_system:
                        partner1 = self._format_covalent_partner_with_chain(
                            ptnr1_auth_comp_ids, ptnr1_auth_seq_ids, ptnr1_label_atom_ids, 
                            ptnr1_label_asym_ids, i
                        )
                        partner2 = self._format_covalent_partner_with_chain(
                            ptnr2_auth_comp_ids, ptnr2_auth_seq_ids, ptnr2_label_atom_ids,
                            ptnr2_label_asym_ids, i
                        )
                        
                        if partner1 and partner2:
                            covalent_info['is_covalent_binder'] = True
                            covalent_info['covalent_info'].append((partner1, partner2))
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error extracting covalent linkages: {e}")
            else:
                print(f"Error extracting covalent linkages: {e}")
        
        return covalent_info
    
    def _format_covalent_partner_with_chain(self, auth_comp_ids: List[str], auth_seq_ids: List[str], 
                                           label_atom_ids: List[str], label_asym_ids: List[str], 
                                           index: int) -> Optional[str]:
        """
        Format covalent partner information with chain ID in the format: CHAIN_COMP_SEQ_ATOM
        
        Args:
            auth_comp_ids: List of auth component IDs
            auth_seq_ids: List of auth sequence IDs  
            label_atom_ids: List of label atom IDs
            label_asym_ids: List of label asymmetric unit IDs (chain IDs)
            index: Index to extract from lists
            
        Returns:
            Formatted string like "A_CYS_805_SG" or None if data missing
        """
        try:
            if (index < len(auth_comp_ids) and index < len(auth_seq_ids) and 
                index < len(label_atom_ids) and index < len(label_asym_ids)):
                
                chain_id = label_asym_ids[index]
                comp_id = auth_comp_ids[index]
                seq_id = auth_seq_ids[index]
                atom_id = label_atom_ids[index]
                
                return f"{chain_id}_{comp_id}_{seq_id}_{atom_id}"
        except (IndexError, TypeError):
            pass
        
        return None
    
    def _format_covalent_partner(self, auth_comp_ids: List[str], auth_seq_ids: List[str], 
                                label_atom_ids: List[str], index: int) -> Optional[str]:
        """
        Format covalent partner information in the format: COMP_SEQ_ATOM
        
        Args:
            auth_comp_ids: List of auth component IDs
            auth_seq_ids: List of auth sequence IDs  
            label_atom_ids: List of label atom IDs
            index: Index to extract from lists
            
        Returns:
            Formatted string like "CYS_805_SG" or None if data missing
        """
        try:
            if (index < len(auth_comp_ids) and index < len(auth_seq_ids) and 
                index < len(label_atom_ids)):
                
                comp_id = auth_comp_ids[index]
                seq_id = auth_seq_ids[index]
                atom_id = label_atom_ids[index]
                
                return f"{comp_id}_{seq_id}_{atom_id}"
        except (IndexError, TypeError):
            pass
        
        return None

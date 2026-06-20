"""
Enhanced Kinase analyzer that integrates mmcif_extractor with kinase_state.py functionality
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any
from Bio import PDB
from warnings import simplefilter

sys.path.append(str(Path(__file__).parent.parent))  # Ensure the parent directory is in the path
from kincore_funcs.modules.assign_default_values import assign_default_values
from kincore_funcs.modules.run_hmmsearch import run_hmmsearch
from kincore_funcs.modules.identify_group import identify_group
from kincore_funcs.modules.identify_residues import identify_residues, get_real_residue_number
from kincore_funcs.modules.compute_distance import compute_distance
from kincore_funcs.modules.compute_dihedrals import compute_dihedrals
from kincore_funcs.modules.spatial_label import spatial_label
from kincore_funcs.modules.dihedral_label import dihedral_label
from kincore_funcs.modules.chelix import chelix_conformation
from kincore_funcs.modules.active_labels import active_labels
from kincore_funcs.modules.classify_ligands import classify_ligands
from kincore_funcs.modules.delete_files import delete_files

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class KinaseAnalyzer:
    """Enhanced wrapper that integrates mmcif_extractor with kinase analysis"""
    
    def __init__(self, logger=None):
        """Initialize the analyzer"""
        self.logger = logger
        
        # Find HMM location
        self.hmm_loc = self._find_hmm_location()
        
        # Initialize parser
        self.parser = PDB.MMCIFParser(QUIET=True)
        
    def _find_hmm_location(self) -> Optional[Path]:
        """Find the HMM directory"""
        # Look in the kinase_pipeline HMMs directory first
        hmm_path = Path(__file__).parent.parent / "kincore_funcs" / "HMMs"
        
        if hmm_path.exists():
            return str(hmm_path)
        
        self.logger.warning("Warning: HMM directory not found")
        return None
    
    def analyze_structure_with_data(self, cif_path: str, chain_data: Dict[str, Dict[str, Any]], 
                                   run_hmm: bool = True, kinase_family: str = None) -> List[Dict[str, Any]]:
        """
        Analyze a structure using pre-extracted chain data and kinase analysis
        
        Args:
            cif_path: Path to CIF file
            chain_data: Pre-extracted chain data from mmcif_extractor
            
        Returns:
            List of dictionaries with kinase analysis results for each chain
        """
        if not self.hmm_loc:
            if self.logger:
                self.logger.error("HMM directory not available")
            else:
                print("HMM directory not available")
            return []
        
        try:
            if not chain_data:
                if self.logger:
                    self.logger.warning(f"No protein chains found in provided data")
                else:
                    print(f"No protein chains found in provided data")
                return []
            
            # Parse structure for kinase analysis
            if cif_path.endswith(".gz"):
                import gzip
                handle = gzip.open(cif_path, 'rt')
            else:
                handle = open(cif_path, 'r')
            
            structure = self.parser.get_structure("structure_id", handle)
            handle.close()
            
            # Analyze each protein chain
            results = []
            for chain_id, data in chain_data.items():
                # Skip chains that are too short to be kinases
                if data['metadata']['chain_length'] <= 200:
                    if self.logger:
                        self.logger.info(f"Chain {chain_id} too short ({data['metadata']['chain_length']} residues)")
                    continue
                
                result = self._analyze_chain(cif_path, structure, chain_id, data, run_hmm, kinase_family)
                if result:
                    results.append(result)
            
            return results
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error analyzing structure {cif_path}: {e}")
            else:
                print(f"Error analyzing structure {cif_path}: {e}")
            return []
    
    def _analyze_chain(self, cif_path: str, structure, chain_id: str, chain_data: Dict[str, Any], 
                      run_hmm: bool = True, kinase_family: str = None) -> Optional[Dict[str, Any]]:
        """Analyze a single chain using kinase analysis kincore_funcs.modules"""
        
        try:
            # Initialize results dataframe (mimicking kinase_state.py structure)
            conf_df = pd.DataFrame()
            index = 0
            
            # Set basic information
            conf_df.at[index, 'Model_id'] = 0  # Assuming first model
            conf_df.at[index, 'Chain_id'] = chain_id
            
            # Assign default values
            conf_df = assign_default_values(index, conf_df)
            
            # Use sequence and s2c_mapping from mmcif_extractor
            sequence = chain_data['sequence']
            s2c_mapping = chain_data['s2c_mapping']
            
            if not sequence:
                if self.logger:
                    self.logger.warning(f"No sequence found for chain {chain_id}")
                else:
                    print(f"No sequence found for chain {chain_id}")
                return None
            
            # Set sequence information in dataframe
            conf_df.at[index, 'Sequence'] = sequence
            conf_df.at[index, 'First_res'] = chain_data['first_residue'] or 1
            # Handle kinase family identification
            if run_hmm and self.hmm_loc:
                # Create FASTA file from mmcif_extractor sequence for HMM search
                self._create_fasta_file(cif_path, chain_id, sequence)

                # Run HMM search to identify kinase family
                try:
                    run_hmmsearch(self.hmm_loc, cif_path, index, conf_df)
                    conf_df = identify_group(cif_path, index, conf_df)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error in HMM search for chain {chain_id}: {e}")
                    else:
                        print(f"Error in HMM search for chain {chain_id}: {e}")
                    # Clean up any partial HMM files
                    delete_files(cif_path, conf_df.at[index, 'Model_id'], 
                               conf_df.at[index, 'Chain_id'], 'None')
                    return None
                
                group = conf_df.at[index, 'Group']
                if group == 'None':
                    # Clean up HMM files
                    delete_files(cif_path, conf_df.at[index, 'Model_id'], 
                               conf_df.at[index, 'Chain_id'], conf_df.at[index, 'Group'])
                    if self.logger:
                        self.logger.info(f"Chain {chain_id} is probably not a protein kinase")
                    else:
                        print(f"Chain {chain_id} is probably not a protein kinase")
                    return None
            elif kinase_family:
                # Use provided kinase family
                group = kinase_family
                conf_df.at[index, 'Group'] = group
                conf_df.at[index, 'Score'] = 999.0  # High score for manually specified family
            else:
                # Skip HMM search, assume it's a kinase but unknown family
                self.logger.warning(f"Skipping HMM search for chain {chain_id}, assuming kinase family unknown")
            
            # Identify key residues using s2c_mapping from mmcif_extractor
            try:
                conf_df = identify_residues(cif_path, index, conf_df, s2c_mapping)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error identifying residues for chain {chain_id}: {e}")
                else:
                    print(f"Error identifying residues for chain {chain_id}: {e}")
                # Clean up HMM files and return None - residue identification is critical
                if run_hmm and self.hmm_loc:
                    delete_files(cif_path, conf_df.at[index, 'Model_id'], 
                               conf_df.at[index, 'Chain_id'], conf_df.at[index, 'Group'])
                return None
            
            # Calculate distances, dihedrals, and labels
            try:
                conf_df = compute_distance(cif_path, index, conf_df, structure)
                conf_df = compute_dihedrals(cif_path, index, conf_df, structure)
                conf_df = spatial_label(index, conf_df)
                conf_df = dihedral_label(index, conf_df, 0.45)
                conf_df = chelix_conformation(index, conf_df)
                conf_df = active_labels(index, conf_df)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in structural calculations for chain {chain_id}: {e}")
                else:
                    print(f"Error in structural calculations for chain {chain_id}: {e}")
                # Clean up HMM files and continue - some analysis may still be possible
                delete_files(cif_path, conf_df.at[index, 'Model_id'], 
                           conf_df.at[index, 'Chain_id'], conf_df.at[index, 'Group'])
                return None
            
            # Use ligands from mmcif_extractor and classify them
            ligands = chain_data['ligands']
            if ligands and ligands != ["No_ligand"]:
                # Set ligand information in dataframe
                conf_df.at[index, 'Ligand'] = ','.join(ligands)
                # Classify ligands using the kinase module
                conf_df = classify_ligands(cif_path, index, conf_df, structure)
            else:
                conf_df.at[index, 'Ligand'] = "No_ligand"
                conf_df.at[index, 'Ligand_label'] = "No_ligand"
            
            # Get PDB residue numbers for identified residues
            model_id = conf_df.at[index, "Model_id"]
            chain_id_val = conf_df.at[index, "Chain_id"]
            
            for resname in ("XDFG", "Lys", "Glu", "Glu4", "Asp", "Phe", "Gly", "HRD", "XHRD", "Arg", "APE", "APE9", "DFG6", "HPN7"):
                res_num = conf_df.at[index, resname + "_num"]
                if pd.notna(res_num):
                    res1 = int(res_num)
                    if s2c_mapping is not None:
                        conf_df.at[index, f'real_{resname}_num'] = conf_df.at[index, f'{resname}_num']
                    else:
                        conf_df.at[index, "real_" + resname + "_num"] = get_real_residue_number(
                            cif_path, model_id, chain_id_val, res1, structure)
                    
                    # Create residue string
                    real_num = conf_df.at[index, "real_" + resname + "_num"]
                    restype = conf_df.at[index, resname + "_restype"]
                    conf_df.at[index, resname + "_string"] = f"{resname}.{restype}{int(real_num)}"
            
            # Determine final group
            finalgroup = self._determine_final_group(group, conf_df.at[index, 'Score'])
            
            # Create structured result
            result = self._create_structured_result(conf_df, index, chain_data, finalgroup)
            
            # Clean up HMM files (only if HMM search was run)
            if run_hmm and self.hmm_loc:
                delete_files(cif_path, conf_df.at[index, 'Model_id'], 
                            conf_df.at[index, 'Chain_id'], conf_df.at[index, 'Group'])
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error analyzing chain {chain_id}: {e}")
            else:
                print(f"Error analyzing chain {chain_id}: {e}")
            return None
    
    def _determine_final_group(self, group: str, score: float) -> str:
        """Determine final kinase group based on HMM group and score"""
        if group == "None":
            return "None"
        
        # Groups that should be classified as OTHER
        other_groups = {
            "CDC7", "PAN3", "TP53RK", "PIK3R4", "RNASEL", "PKDCC", "BUB", "ULK", 
            "WNK", "PINK1", "EIF2AK41", "PEAK", "POMK", "HASP", "SCYL", "STK31", 
            "PXK", "MOS", "TBCK", "RPS6KC1"
        }
        
        if group in other_groups:
            return "OTHER"
        
        if group == "MAP3K123":
            return "TKL"
        
        if score < 190.0:
            return "OTHER"
        
        return group
    
    def _create_structured_result(self, conf_df: pd.DataFrame, index: int, 
                                chain_data: Dict[str, Any], finalgroup: str) -> Dict[str, Any]:
        """Create structured result dictionary"""
        
        # Extract key labels
        result = {
            'chain_id': conf_df.at[index, 'Chain_id'],
            'kinase_family': finalgroup,
            'hmm_group': conf_df.at[index, 'Group'],
            'hmm_score': conf_df.at[index, 'Score'],
            
            # Main kinase state labels
            'activity_label': conf_df.at[index, 'Activity_label'],
            'spatial_label': conf_df.at[index, 'Spatial_label'],
            'dihedral_label': conf_df.at[index, 'Dihedral_label'],
            'chelix_label': conf_df.at[index, 'Chelix_label'],
            'saltbr_label': conf_df.at[index, 'SaltBr_label'],
            'actloop_nt_label': conf_df.at[index, 'ActLoopNT_label'],
            'actloop_ct_label': conf_df.at[index, 'ActLoopCT_label'],
            'spine_label': conf_df.at[index, 'Spine_label'],
            
            # Ligand information (enhanced from mmcif_extractor)
            'ligands': chain_data['ligands'],
            'ligand_classification': conf_df.at[index, 'Ligand_label'],
            
            # Key residues
            'key_residues': self._extract_key_residues(conf_df, index),
            
            # Distances and dihedrals
            'distances': self._extract_distances(conf_df, index),
            'dihedrals': self._extract_dihedrals(conf_df, index),
            
            # Enhanced information from mmcif_extractor
            'phosphorylations': chain_data.get('phosphorylations', []),
            'mutations': chain_data.get('mutations', []),
            'gene_name': chain_data.get('gene_name'),
            'uniprot_id': chain_data.get('uniprot_id'),
            
            # Metadata
            'sequence_length': len(chain_data['sequence']) if chain_data['sequence'] else 0,
            'first_residue': chain_data['first_residue'],
            'has_coordinates': chain_data['metadata']['has_coordinates'],
            'sequence_source': chain_data['metadata']['sequence_source']
        }
        
        return result
    
    def _extract_key_residues(self, conf_df: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Extract key residue information"""
        residues = {}
        for resname in ("XDFG", "Lys", "Glu", "Glu4", "Asp", "Phe", "Gly", "HRD", "XHRD", "Arg", "APE", "APE9", "DFG6", "HPN7"):
            residues[resname] = {
                'restype': conf_df.at[index, f'{resname}_restype'],
                'sequence_num': conf_df.at[index, f'{resname}_num'],
                'pdb_num': conf_df.at[index, f'real_{resname}_num'],
                'string': conf_df.at[index, f'{resname}_string']
            }
        return residues
    
    def _extract_distances(self, conf_df: pd.DataFrame, index: int) -> Dict[str, float]:
        """Extract distance measurements"""
        return {
            'Glu4_Phe': conf_df.at[index, 'Glu4-Phe-dis'],
            'Lys_Phe': conf_df.at[index, 'Lys-Phe-dis'],
            'Lys_Glu': conf_df.at[index, 'Lys-Glu-dis'],
            'SaltBr': conf_df.at[index, 'LysNZ-GluOE-dis'],
            'DFG6_XHRD': conf_df.at[index, 'DFG6-XHRD-dis'],
            'APE9_Arg': conf_df.at[index, 'APE9-Arg-dis'],
            'Spine1': conf_df.at[index, 'Spine1-dis'],
            'Spine2': conf_df.at[index, 'Spine2-dis'],
            'Spine3': conf_df.at[index, 'Spine3-dis'],
            'Spine_total': conf_df.at[index, 'Spine-dis']
        }
    
    def _extract_dihedrals(self, conf_df: pd.DataFrame, index: int) -> Dict[str, float]:
        """Extract dihedral angle measurements"""
        return {
            'XDFG_Phi': conf_df.at[index, 'XDFG_Phi'],
            'XDFG_Psi': conf_df.at[index, 'XDFG_Psi'],
            'Asp_Phi': conf_df.at[index, 'Asp_Phi'],
            'Asp_Psi': conf_df.at[index, 'Asp_Psi'],
            'Asp_Chi1': conf_df.at[index, 'Asp_Chi1'],
            'Asp_Chi2': conf_df.at[index, 'Asp_Chi2'],
            'Phe_Phi': conf_df.at[index, 'Phe_Phi'],
            'Phe_Psi': conf_df.at[index, 'Phe_Psi'],
            'Phe_Chi1': conf_df.at[index, 'Phe_Chi1'],
            'Phe_Chi2': conf_df.at[index, 'Phe_Chi2'],
            'Gly_Phi': conf_df.at[index, 'Gly_Phi'],
            'Gly_Psi': conf_df.at[index, 'Gly_Psi']
        }

    # Legacy methods for backward compatibility
    def _parse_kinase_output(self, output: str) -> List[Dict]:
        """Legacy method - kept for backward compatibility"""
        if self.logger:
            self.logger.warning("Using legacy parsing method. Consider using the new integrated analysis.")
        else:
            print("Warning: Using legacy parsing method. Consider using the new integrated analysis.")
        return []
    
    def _parse_labels_line(self, line: str) -> Dict:
        """Legacy method - kept for backward compatibility"""
        return {}
    
    def _parse_residues_line(self, line: str) -> str:
        """Legacy method - kept for backward compatibility"""
        return ""
    
    def _parse_ligands_line(self, line: str) -> tuple:
        """Legacy method - kept for backward compatibility"""
        return "No_ligand", "No_ligand"
    
    def _create_fasta_file(self, cif_path: str, chain_id: str, sequence: str) -> str:
        """Create FASTA file from sequence for HMM search"""
        
        # Generate FASTA filename based on CIF path and chain ID
        fasta_path = f"{cif_path[0:-4]}_{chain_id}.fasta"
        
        # Write sequence to FASTA file
        with open(fasta_path, 'w') as f:
            f.write(f">{cif_path[0:-4]}_{chain_id}\n")
            f.write(sequence)
        
        return fasta_path

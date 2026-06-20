#!/usr/bin/env python3
"""
Main kinase analysis pipeline that processes PDB IDs or co-folding models through separate workflows:
1. PDB workflow: Download -> Extract -> Kinase Analysis -> PDB API -> CSV
2. Co-folding workflow: Process -> Extract -> Kinase Analysis -> CSV (skip PDB API)
"""

import os
import sys
import pandas as pd
import csv
from pathlib import Path
from typing import List, Dict, Optional, Any
import time
import logging

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from workers.cif_downloader import CIFDownloader
from workers.mmcif_extractor import MMCIFExtractor
from workers.kinase_analyzer import KinaseAnalyzer
from workers.pdb_api_extractor import PDBAPIExtractor
from utils.cofolding_processor import process_cofolding_files


class KinaseAnalysisPipeline:
    """Complete pipeline for kinase structure analysis"""
    
    def __init__(self, work_dir: str = "kinase_pipeline_work", verbose: bool = False):
        """
        Initialize the pipeline
        
        Args:
            work_dir: Working directory for temporary files
            verbose: Enable verbose logging
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logger(verbose)
        
        # Initialize components with logger
        self.cif_downloader = CIFDownloader(self.work_dir / "cif_files", logger=self.logger)
        self.mmcif_extractor = MMCIFExtractor(logger=self.logger)
        self.kinase_analyzer = KinaseAnalyzer(logger=self.logger)
        self.pdb_api_extractor = PDBAPIExtractor()
        
        # Track whether CIF files should be preserved
        self.keep_cif_files = False
    
    def _setup_logger(self, verbose: bool = False) -> logging.Logger:
        """Set up the main pipeline logger"""
        logger = logging.getLogger("kinase_pipeline")
        logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create formatters
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_level = logging.INFO if verbose else logging.WARNING
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.work_dir / "pipeline.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def process_pdb_structures(self, pdb_ids: List[str], output_csv: str, 
                              minimal_output: bool = False, 
                              delete_cif_after: bool = False) -> pd.DataFrame:
        """
        Process PDB structures through the experimental structure workflow
        
        Args:
            pdb_ids: List of PDB IDs to process
            output_csv: Output CSV file path
            minimal_output: If True, create minimal CSV with essential columns only
            delete_cif_after: If True, delete downloaded CIF files after processing
            
        Returns:
            DataFrame with results
        """
        self.logger.info(f"Starting PDB structure pipeline for {len(pdb_ids)} PDB IDs")
        
        # Step 1: Download CIF files
        cif_files = self._download_cif_files(pdb_ids)
        if not cif_files:
            return pd.DataFrame()
        
        # Step 2: Extract mmCIF data
        chain_data_dict = self._extract_pdb_chain_data(cif_files)
        
        # Step 3: Run kinase analysis
        kinase_results = self._run_kinase_analysis(chain_data_dict, is_cofolding=False)
        if not kinase_results:
            return pd.DataFrame()
        
        # Step 4: Query PDB API for metadata
        api_data = self._query_pdb_api(kinase_results)
        
        # Step 5: Create final CSV
        df = self._create_final_csv(kinase_results, api_data, output_csv, minimal_output)
        
        # Step 6: Delete CIF files if requested
        if delete_cif_after:
            self._delete_cif_files(cif_files)
        
        return df
    
    def process_cofolding_structures(self, cif_folder: str, output_csv: str, 
                                   minimal_output: bool = False) -> pd.DataFrame:
        """
        Process co-folding structures through the computational structure workflow
        
        Args:
            cif_folder: Folder containing co-folding CIF files
            output_csv: Output CSV file path
            minimal_output: If True, create minimal CSV with essential columns only
            
        Returns:
            DataFrame with results
        """
        self.logger.info(f"Starting co-folding structure pipeline for folder: {cif_folder}")
        
        # Step 1: Find CIF files
        cif_files = self._find_cif_files_in_folder(cif_folder)
        if not cif_files:
            raise ValueError(f"No CIF files found in folder: {cif_folder}")
        
        # Step 2: Pre-process co-folding models
        self._preprocess_cofolding_files(cif_files)
        
        # Step 3: Extract mmCIF data (basic extraction for co-folding)
        chain_data_dict = self._extract_cofolding_chain_data(cif_files)
        
        # Step 4: Run kinase analysis
        kinase_results = self._run_kinase_analysis(chain_data_dict, is_cofolding=True)
        if not kinase_results:
            return pd.DataFrame()
        
        # Step 5: Create final CSV (skip PDB API for co-folding)
        df = self._create_final_csv(kinase_results, {}, output_csv, minimal_output)
        
        return df
    
    # Step-specific methods
    def _download_cif_files(self, pdb_ids: List[str]) -> Dict[str, str]:
        """Step 1: Download CIF files for PDB IDs"""
        self.logger.info("Step 1: Downloading CIF files...")
        downloaded_files = self.cif_downloader.download_batch(pdb_ids)
        self.logger.info(f"Downloaded {len(downloaded_files)} CIF files")
        return downloaded_files
    
    def _find_cif_files_in_folder(self, folder_path: str) -> Dict[str, str]:
        """Find all CIF files in a folder"""
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        cif_files = {}
        
        # Look for .cif and .cif.gz files
        for cif_file in folder.glob('*.cif'):
            # Use filename without extension as structure ID
            structure_id = cif_file.stem.split('.')[0]  # Remove any .cif suffix
            if structure_id.endswith('.cif'):  # Handle .cif.gz case
                structure_id = structure_id[:-4]
            
            cif_files[structure_id] = str(cif_file)
        
        self.logger.info(f"Found {len(cif_files)} CIF files in folder")
        return cif_files
    
    def _preprocess_cofolding_files(self, cif_files: Dict[str, str]) -> Dict[str, str]:
        """Step 2: Pre-process co-folding CIF files"""
        self.logger.info("Step 2: Pre-processing co-folding models...")
        processed_cofolding_files = process_cofolding_files(cif_files, self.logger)
        return processed_cofolding_files
    
    def _extract_pdb_chain_data(self, cif_files: Dict[str, str]) -> Dict[str, Dict]:
        """Step 2/3: Extract chain data from PDB CIF files (full extraction)"""
        self.logger.info("Step 2: Extracting PDB chain data...")
        chain_data_dict = {}
        
        for structure_id, cif_path in cif_files.items():
            try:
                chain_data = self.mmcif_extractor.extract_all_chains(cif_path)
                if chain_data:
                    chain_data_dict[structure_id] = {
                        'cif_path': cif_path,
                        'chain_data': chain_data
                    }
                    self.logger.info(f"  ✓ Extracted data for {structure_id}: {len(chain_data)} chains")
                else:
                    self.logger.warning(f"  ✗ No chain data extracted for {structure_id}")
            except Exception as e:
                self.logger.error(f"  ✗ Error extracting {structure_id}: {e}")
        
        return chain_data_dict
    
    def _extract_cofolding_chain_data(self, cif_files: Dict[str, str]) -> Dict[str, Dict]:
        """Step 3: Extract chain data from co-folding CIF files (basic extraction)"""
        self.logger.info("Step 3: Extracting co-folding chain data...")
        chain_data_dict = {}
        
        for structure_id, cif_path in cif_files.items():
            try:
                chain_data = self.mmcif_extractor.extract_basic_chains(cif_path)
                if chain_data:
                    chain_data_dict[structure_id] = {
                        'cif_path': cif_path,
                        'chain_data': chain_data
                    }
                    self.logger.info(f"  ✓ Extracted data for {structure_id}: {len(chain_data)} chains")
                else:
                    self.logger.warning(f"  ✗ No chain data extracted for {structure_id}")
            except Exception as e:
                self.logger.error(f"  ✗ Error extracting {structure_id}: {e}")
        
        return chain_data_dict
    
    def _run_kinase_analysis(self, chain_data_dict: Dict[str, Dict], is_cofolding: bool = False) -> List[Dict[str, Any]]:
        """Step 3/4: Run kinase analysis on extracted chain data"""
        step_num = 4 if is_cofolding else 3
        model_type = "co-folding" if is_cofolding else "PDB"
        self.logger.info(f"Step {step_num}: Running kinase analysis on {model_type} structures...")
        
        all_results = []
        
        for structure_id, data in chain_data_dict.items():
            cif_path = data['cif_path']
            chain_data = data['chain_data']
            
            try:
                kinase_results = self.kinase_analyzer.analyze_structure_with_data(
                    cif_path, chain_data, run_hmm=True
                )
                
                if kinase_results:
                    self.logger.info(f"  ✓ Analyzed {structure_id}: {len(kinase_results)} kinase chains")
                    for result in kinase_results:
                        result['structure_id'] = structure_id
                        result['pdb_id'] = structure_id  # For backward compatibility
                        result['is_cofolding'] = is_cofolding
                        result['cif_path'] = cif_path
                        
                        # Add UniProt ID and alignment information from chain data if available
                        chain_id = result['chain_id']
                        if chain_id in chain_data:
                            chain_info = chain_data[chain_id]
                            # Transfer UniProt ID if it exists in the chain data (rename to uniprot_id_mmcif)
                            if 'uniprot_id' in chain_info and chain_info['uniprot_id']:
                                result['uniprot_id_mmcif'] = chain_info['uniprot_id']
                            # Also transfer gene name if available
                            if 'gene_name' in chain_info and chain_info['gene_name']:
                                result['gene_name'] = chain_info['gene_name']
                            # Transfer alignment information if available
                            if 'alignment' in chain_info and chain_info['alignment']:
                                result['alignment'] = chain_info['alignment']
                            # Transfer reference sequence if available
                            if 'reference_sequence' in chain_info and chain_info['reference_sequence']:
                                result['reference_sequence'] = chain_info['reference_sequence']
                            # Transfer covalent linkage information if available
                            if 'is_covalent_binder' in chain_info:
                                result['is_covalent_binder'] = chain_info['is_covalent_binder']
                            if 'covalent_info' in chain_info:
                                result['covalent_info'] = chain_info['covalent_info']
                        
                        all_results.append(result)
                else:
                    self.logger.warning(f"  ✗ No kinase chains found in {structure_id}")
                    
            except Exception as e:
                self.logger.error(f"  ✗ Error analyzing {structure_id}: {e}")
        
        self.logger.info(f"Successfully analyzed {len(all_results)} kinase chains")
        return all_results
    
    def _query_pdb_api(self, kinase_results: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """Step 4: Query PDB API for structure metadata (PDB structures only)"""
        self.logger.info("Step 4: Querying PDB API for structure metadata...")
        
        # Get unique PDB IDs
        pdb_ids = list(set(r['structure_id'] for r in kinase_results if len(r.get('structure_id', '')) == 4))
        
        if pdb_ids:
            api_data = self.pdb_api_extractor.get_batch_data(pdb_ids)
            self.logger.info(f"Retrieved API data for {len(api_data)} PDB structures")
            return api_data
        else:
            self.logger.info("No PDB structures found, skipping API queries")
            return {}
    
    def _create_final_csv(self, kinase_results: List[Dict[str, Any]], api_data: Dict[str, Dict], 
                         output_csv: str, minimal_output: bool = False) -> pd.DataFrame:
        """Step 5: Create final CSV output"""
        self.logger.info(f"Step 5: Creating final CSV...")
        
        # Add UniProt IDs from PDB API to kinase results
        self._add_pdb_api_uniprot_ids(kinase_results, api_data)
        
        if minimal_output:
            df = self._create_minimal_csv(kinase_results, api_data, output_csv)
        else:
            df = self._create_comprehensive_csv(kinase_results, api_data, output_csv)
        
        if output_csv:
            self.logger.info(f"Results written to {output_csv}")
        
        return df
    
    def _add_pdb_api_uniprot_ids(self, kinase_results: List[Dict[str, Any]], api_data: Dict[str, Dict]) -> None:
        """Add UniProt IDs from PDB API to kinase results"""
        for result in kinase_results:
            structure_id = result.get('structure_id', result.get('pdb_id', ''))
            chain_id = result.get('chain_id', '')
            
            # Get UniProt ID from PDB API data if available
            if structure_id in api_data and 'uniprot_id_pdb_api' in api_data[structure_id]:
                uniprot_mapping = api_data[structure_id]['uniprot_id_pdb_api']
                if chain_id in uniprot_mapping:
                    result['uniprot_id_pdb_api'] = uniprot_mapping[chain_id]
    
    def _create_minimal_csv(self, kinase_results: List[Dict[str, Any]], 
                           api_data: Dict[str, Dict], output_csv: str) -> pd.DataFrame:
        """Create minimal CSV focused on kinase-specific information"""
        rows = []
        
        # Check if we're dealing with cofolding models
        has_cofolding = any(result.get('is_cofolding', False) for result in kinase_results)
        
        for result in kinase_results:
            structure_id = result.get('structure_id', result.get('pdb_id', ''))
            chain_id = result['chain_id']
            is_cofolding = result.get('is_cofolding', False)
            api_info = api_data.get(structure_id, {})
            
            # Create kinase-focused row
            row = {
                'structure_id': structure_id,
                'chain_id': chain_id,
                'model_type': 'cofolding' if is_cofolding else 'experimental',
                'gene_name': result.get('gene_name', ''),
                'uniprot_id_mmcif': result.get('uniprot_id_mmcif', ''),
                'uniprot_id_pdb_api': result.get('uniprot_id_pdb_api', ''),
                'organism': api_info.get('Organism', result.get('organism', 'Homo sapiens')),
                'kinase_family': result.get('kinase_family', ''),
                'hmm_score': result.get('hmm_score', ''),
                'confidence_level': result.get('confidence_level', ''),
                
                # All kinase state labels
                'activity_label': result.get('activity_label', ''),
                'spatial_label': result.get('spatial_label', ''),
                'dihedral_label': result.get('dihedral_label', ''),
                'chelix_label': result.get('chelix_label', ''),
                'saltbr_label': result.get('saltbr_label', ''),
                'actloop_nt_label': result.get('actloop_nt_label', ''),
                'actloop_ct_label': result.get('actloop_ct_label', ''),
                'spine_label': result.get('spine_label', ''),
                
                # Ligand information
                'ligands': ';'.join(result.get('ligands', [])),
                'ligand_type': result.get('ligand_classification', ''),
                'is_covalent_binder': result.get('is_covalent_binder', False),
                'covalent_info': self._format_covalent_info(result.get('covalent_info', [])),
                
                # Post-translational modifications
                'phosphorylations': self._format_phosphorylations(result.get('phosphorylations', [])),
                'mutations': self._format_mutations(result.get('mutations', [])),
                
                # Only include experimental info for PDB structures
                'method': api_info.get('Method', 'Computational' if is_cofolding else ''),
                'resolution': api_info.get('Resolution', 'N/A' if is_cofolding else ''),
                'deposition_date': api_info.get('Deposition_Date', '') if not is_cofolding else ''
            }
            
            # Add alignment information only for experimental structures
            if not is_cofolding:
                row.update({
                    'pdb_align_start': result.get('alignment', {}).get('pdb_start', ''),
                    'pdb_align_end': result.get('alignment', {}).get('pdb_end', ''),
                    'uniprot_align_start': result.get('alignment', {}).get('uniprot_start', ''),
                    'uniprot_align_end': result.get('alignment', {}).get('uniprot_end', ''),
                })
            
            # For cofolding models, exclude num_ligands, num_phosphorylations, num_mutations
            if not has_cofolding:
                row['num_ligands'] = len(result.get('ligands', []))
                row['num_phosphorylations'] = len(result.get('phosphorylations', []))
                row['num_mutations'] = len(result.get('mutations', []))
            
            rows.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        
        # Define column order for kinase-focused CSV
        if has_cofolding:
            # Skip unnecessary columns for cofolding models
            column_order = [
                'structure_id', 'chain_id', 
                'kinase_family', 'hmm_score', 'confidence_level',
                'activity_label', 'spatial_label', 'dihedral_label',
                'chelix_label', 'saltbr_label', 'actloop_nt_label', 'actloop_ct_label', 'spine_label',
                'ligands', 'ligand_type', 'is_covalent_binder', 'covalent_info',
                'phosphorylations', 'mutations'
            ]
            # Keep uniprot_id_mmcif if it was manually provided
            if any(result.get('uniprot_id_mmcif') for result in kinase_results):
                column_order.insert(2, 'uniprot_id_mmcif')
        else:
            # Full columns for experimental structures
            column_order = [
                'structure_id', 'chain_id', 'model_type', 'gene_name', 'uniprot_id_mmcif', 'uniprot_id_pdb_api', 'organism',
                'kinase_family', 'hmm_score', 'confidence_level',
                'activity_label', 'spatial_label', 'dihedral_label',
                'chelix_label', 'saltbr_label', 'actloop_nt_label', 'actloop_ct_label', 'spine_label',
                'ligands', 'ligand_type', 'is_covalent_binder', 'covalent_info',
                'phosphorylations', 'mutations',
                'pdb_align_start', 'pdb_align_end', 'uniprot_align_start', 'uniprot_align_end',
                'method', 'resolution', 'deposition_date'
            ]
        
        # Add count columns only if not dealing with cofolding models
        if not has_cofolding:
            # Insert count columns after their respective data columns
            ligand_idx = column_order.index('ligand_type') + 1
            column_order.insert(ligand_idx, 'num_ligands')
            
            phos_idx = column_order.index('phosphorylations') + 1
            column_order.insert(phos_idx, 'num_phosphorylations')
            
            mut_idx = column_order.index('mutations') + 1
            column_order.insert(mut_idx, 'num_mutations')
        
        # Reorder columns (only include columns that exist in the DataFrame)
        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns]
        if 'deposition_date' in df.columns:
            df['deposition_date'] = pd.to_datetime(df['deposition_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        
        # Save to CSV if path provided
        if output_csv:
            df = df.sort_values(by=['structure_id', 'chain_id'])
            df.to_csv(output_csv, index=False)
        
        return df

    def _create_comprehensive_csv(self, kinase_results: List[Dict[str, Any]], 
                                 api_data: Dict[str, Dict], output_csv: str) -> pd.DataFrame:
        """Create comprehensive CSV with kinase-focused information"""
        rows = []
        
        # Check if we're dealing with cofolding models
        has_cofolding = any(result.get('is_cofolding', False) for result in kinase_results)
        
        for result in kinase_results:
            structure_id = result.get('structure_id', result.get('pdb_id', ''))
            chain_id = result['chain_id']
            is_cofolding = result.get('is_cofolding', False)
            api_info = api_data.get(structure_id, {})
            
            # Create comprehensive kinase-focused row
            row = {
                # Basic identifiers
                'structure_id': structure_id,
                'chain_id': chain_id,
                'structure_chain': f"{structure_id}_{chain_id}",
                'model_type': 'cofolding' if is_cofolding else 'experimental',
                'gene_name': result.get('gene_name', ''),
                'organism': api_info.get('Organism', result.get('organism', 'Homo sapiens')),
                
                # Kinase classification and confidence
                'kinase_family': result.get('kinase_family', ''),
                'hmm_group': result.get('hmm_group', ''),
                'hmm_score': result.get('hmm_score', ''),
                'confidence_score': result.get('confidence_score', result.get('hmm_score', '')),
                'confidence_level': result.get('confidence_level', ''),
                
                # UniProt information (mainly for experimental structures)
                'uniprot_id_mmcif': result.get('uniprot_id_mmcif', ''),
                'uniprot_id_pdb_api': result.get('uniprot_id_pdb_api', ''),
                'reference_sequence': result.get('reference_sequence', ''),
                'pdb_align_start': result.get('alignment', {}).get('pdb_start', ''),
                'pdb_align_end': result.get('alignment', {}).get('pdb_end', ''),
                'uniprot_align_start': result.get('alignment', {}).get('uniprot_start', ''),
                'uniprot_align_end': result.get('alignment', {}).get('uniprot_end', ''),
                
                # All kinase state labels
                'activity_label': result.get('activity_label', ''),
                'spatial_label': result.get('spatial_label', ''),
                'dihedral_label': result.get('dihedral_label', ''),
                'chelix_label': result.get('chelix_label', ''),
                'saltbr_label': result.get('saltbr_label', ''),
                'actloop_nt_label': result.get('actloop_nt_label', ''),
                'actloop_ct_label': result.get('actloop_ct_label', ''),
                'spine_label': result.get('spine_label', ''),
                
                # Ligand information
                'ligands': ';'.join(result.get('ligands', [])),
                'ligand_type': result.get('ligand_classification', ''),
                'num_ligands': len(result.get('ligands', [])),
                'is_covalent_binder': result.get('is_covalent_binder', False),
                'covalent_info': self._format_covalent_info(result.get('covalent_info', [])),
                
                # Post-translational modifications
                'phosphorylations': self._format_phosphorylations(result.get('phosphorylations', [])),
                'num_phosphorylations': len(result.get('phosphorylations', [])),
                'mutations': self._format_mutations(result.get('mutations', [])),
                'num_mutations': len(result.get('mutations', [])),
                
                # Key residues with PDB numbering
                'lys_residue': self._format_key_residue(result, 'Lys'),
                'glu_residue': self._format_key_residue(result, 'Glu'),
                'asp_residue': self._format_key_residue(result, 'Asp'),
                'phe_residue': self._format_key_residue(result, 'Phe'),
                'gly_residue': self._format_key_residue(result, 'Gly'),
                'dfg_residue': self._format_key_residue(result, 'XDFG'),
                'hrd_residue': self._format_key_residue(result, 'HRD'),
                
                # Key structural measurements
                'lys_glu_distance': result.get('distances', {}).get('Lys_Glu', ''),
                'saltbridge_distance': result.get('distances', {}).get('SaltBr', ''),
                'spine_distance': result.get('distances', {}).get('Spine_total', ''),
                'dfg_hrd_distance': result.get('distances', {}).get('DFG6_XHRD', ''),
                
                # Key dihedral angles
                'asp_phi': result.get('dihedrals', {}).get('Asp_Phi', ''),
                'asp_psi': result.get('dihedrals', {}).get('Asp_Psi', ''),
                'asp_chi1': result.get('dihedrals', {}).get('Asp_Chi1', ''),
                'asp_chi2': result.get('dihedrals', {}).get('Asp_Chi2', ''),
                'phe_phi': result.get('dihedrals', {}).get('Phe_Phi', ''),
                'phe_psi': result.get('dihedrals', {}).get('Phe_Psi', ''),
                'phe_chi1': result.get('dihedrals', {}).get('Phe_Chi1', ''),
                'phe_chi2': result.get('dihedrals', {}).get('Phe_Chi2', ''),
                
                # Sequence information
                'sequence_length': result.get('sequence_length', ''),
                'first_residue': result.get('first_residue', ''),
                'sequence_source': result.get('sequence_source', ''),
                
                # Experimental structure metadata (only for PDB structures)
                'method': api_info.get('Method', 'Computational' if is_cofolding else ''),
                'resolution': api_info.get('Resolution', 'N/A' if is_cofolding else ''),
                'r_factor': api_info.get('Rfac', 'N/A' if is_cofolding else ''),
                'free_r_factor': api_info.get('FreeRfac', 'N/A' if is_cofolding else ''),
                'deposition_date': api_info.get('Deposition_Date', '') if not is_cofolding else '',
                'protein_name': api_info.get('ProteinName', '') if not is_cofolding else '',
                
                # Co-folding specific metadata
                'computational_model': result.get('computational_model', is_cofolding),
                'needs_experimental_validation': result.get('needs_experimental_validation', is_cofolding),
                
                # File path for reference
                'cif_path': result.get('cif_path', ''),
                
                # Analysis metadata
                'analysis_successful': True,
                'error_message': ''
            }
            
            rows.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        
        # Define column order prioritizing kinase information
        if has_cofolding:
            # Simplified column order for cofolding models
            column_order = [
                # Core identifiers
                'structure_id', 'chain_id', 'structure_chain',
                
                # Kinase classification
                'kinase_family', 'hmm_group', 'hmm_score', 'confidence_score', 'confidence_level',
                
                # Kinase state labels
                'activity_label', 'spatial_label', 'dihedral_label', 'chelix_label',
                'saltbr_label', 'actloop_nt_label', 'actloop_ct_label', 'spine_label',
                
                # Ligands and modifications
                'ligands', 'ligand_type', 'num_ligands',
                'phosphorylations', 'num_phosphorylations', 'mutations', 'num_mutations',
                
                # Key residues
                'lys_residue', 'glu_residue', 'asp_residue', 'phe_residue', 'gly_residue',
                'dfg_residue', 'hrd_residue',
                
                # Structural measurements
                'lys_glu_distance', 'saltbridge_distance', 'spine_distance', 'dfg_hrd_distance',
                
                # Dihedral angles
                'asp_phi', 'asp_psi', 'asp_chi1', 'asp_chi2',
                'phe_phi', 'phe_psi', 'phe_chi1', 'phe_chi2',
                
                # Sequence info
                'sequence_length', 'first_residue', 'sequence_source',
                
                # Co-folding specific metadata
                'computational_model', 'needs_experimental_validation',
                
                # File reference and analysis metadata
                'cif_path', 'analysis_successful', 'error_message'
            ]
            
            # Keep uniprot_id_mmcif if it was manually provided
            if any(result.get('uniprot_id_mmcif') for result in kinase_results):
                column_order.insert(3, 'uniprot_id_mmcif')
        else:
            # Full columns for experimental structures
            column_order = [
                # Core identifiers
                'structure_id', 'chain_id', 'structure_chain', 'model_type', 'gene_name', 'organism',
                
                # Kinase classification
                'kinase_family', 'hmm_group', 'hmm_score', 'confidence_score', 'confidence_level',
                'uniprot_id_mmcif', 'uniprot_id_pdb_api', 'reference_sequence', 'pdb_align_start', 'pdb_align_end', 
                'uniprot_align_start', 'uniprot_align_end',
            
            # Kinase state labels (most important)
            'activity_label', 'spatial_label', 'dihedral_label', 'chelix_label',
            'saltbr_label', 'actloop_nt_label', 'actloop_ct_label', 'spine_label',
            
            # Ligands and modifications
            'ligands', 'ligand_type', 'num_ligands',
            'phosphorylations', 'num_phosphorylations', 'mutations', 'num_mutations',
            
            # Key residues
            'lys_residue', 'glu_residue', 'asp_residue', 'phe_residue', 'gly_residue',
            'dfg_residue', 'hrd_residue',
            
            # Structural measurements
            'lys_glu_distance', 'saltbridge_distance', 'spine_distance', 'dfg_hrd_distance',
            
            # Dihedral angles
            'asp_phi', 'asp_psi', 'asp_chi1', 'asp_chi2',
            'phe_phi', 'phe_psi', 'phe_chi1', 'phe_chi2',
            
            # Sequence info
            'sequence_length', 'first_residue', 'sequence_source',
            
            # Experimental metadata (less important for co-folding)
            'method', 'resolution', 'r_factor', 'free_r_factor', 'deposition_date', 'protein_name',
            
            # Co-folding metadata
            'computational_model', 'needs_experimental_validation',
            
            # File reference and analysis metadata
            'cif_path', 'analysis_successful', 'error_message'
        ]
        
        # Reorder columns
        available_columns = [col for col in column_order if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in column_order]
        final_columns = available_columns + remaining_columns
        
        df = df[final_columns]
        if 'deposition_date' in df.columns:
            df['deposition_date'] = pd.to_datetime(df['deposition_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        
        # Save to CSV if path provided
        if output_csv:
            df = df.sort_values(by=['structure_id', 'chain_id'])
            df.to_csv(output_csv, index=False)
        
        return df
    
    def _format_phosphorylations(self, phosphorylations: List) -> str:
        """Format phosphorylations for CSV output"""
        if not phosphorylations:
            return ''
        
        formatted = []
        for p in phosphorylations:
            if isinstance(p, dict):
                residue = p.get('residue', 'X')
                pdb_num = p.get('pdb_residue_num', 'UNK')
                formatted.append(f"{residue}{pdb_num}")
            else:
                formatted.append(str(p))
        
        return ';'.join(formatted)
    
    def _format_mutations(self, mutations: List) -> str:
        """Format mutations for CSV output"""
        if not mutations:
            return ''
        
        formatted = []
        for m in mutations:
            if isinstance(m, dict):
                residue = m.get('pdb_residue', m.get('canonical_residue', 'UNK'))
                formatted.append(str(residue))
            else:
                formatted.append(str(m))
        
        return ';'.join(formatted)
    
    def _format_covalent_info(self, covalent_info: List) -> str:
        """Format covalent linkage information for CSV output"""
        if not covalent_info:
            return ''
        
        formatted = []
        for cov_link in covalent_info:
            if isinstance(cov_link, (tuple, list)) and len(cov_link) == 2:
                protein_partner, ligand_partner = cov_link
                formatted.append(f"{protein_partner}-{ligand_partner}")
            else:
                formatted.append(str(cov_link))
        
        return ';'.join(formatted)
    
    def _format_key_residue(self, result: Dict, residue_name: str) -> str:
        """Format key residue information for CSV output"""
        key_residues = result.get('key_residues', {})
        if residue_name not in key_residues:
            return ''
        
        residue_info = key_residues[residue_name]
        restype = residue_info.get('restype', '')
        pdb_num = residue_info.get('pdb_num', '')
        
        if restype and pdb_num:
            return f"{restype}{pdb_num}"
        return ''
    
    def process_from_file(self, input_file: str, output_csv: str, 
                         file_type: str = 'pdb_list', minimal_output: bool = False,
                         delete_cif_after: bool = False, batch_size: int = 0) -> pd.DataFrame:
        """
        Process PDB IDs from various input file formats
        
        Args:
            input_file: Input file path
            output_csv: Output CSV file path
            file_type: Type of input file ('pdb_list', 'csv', 'results_file')
            minimal_output: If True, create minimal CSV with essential columns only
            delete_cif_after: If True, delete downloaded CIF files after processing
            batch_size: If > 0, process PDB IDs in batches of this size and append results
            
        Returns:
            DataFrame with results
        """
        if file_type == 'pdb_list':
            pdb_ids = self._read_pdb_list(input_file)
        elif file_type == 'csv':
            pdb_ids = self._extract_pdb_ids_from_csv(input_file)
        elif file_type == 'results_file':
            pdb_ids = self._extract_pdb_ids_from_results(input_file)
        else:
            raise ValueError(f"Unknown file type: {file_type}")
        
        self.logger.info(f"Extracted {len(pdb_ids)} unique PDB IDs from {input_file}")
        
        if batch_size > 0 and len(pdb_ids) > batch_size:
            return self.process_pdb_structures_in_batches(pdb_ids, output_csv, minimal_output, delete_cif_after, batch_size)
        else:
            return self.process_pdb_structures(pdb_ids, output_csv, minimal_output, delete_cif_after)
    
    def _read_pdb_list(self, file_path: str) -> List[str]:
        """Read PDB IDs from a simple text file (one per line)"""
        pdb_ids = []
        with open(file_path, 'r') as f:
            for line in f:
                pdb_id = line.strip().upper()
                if len(pdb_id) == 4 and pdb_id.isalnum():
                    pdb_ids.append(pdb_id)
        return list(set(pdb_ids))
    
    def _extract_pdb_ids_from_csv(self, csv_file: str) -> List[str]:
        """Extract PDB IDs from CSV file (assumes 'pdb_id' column exists)"""
        pdb_ids = set()
        df = pd.read_csv(csv_file)
        
        # Look for common PDB ID column names
        pdb_columns = ['pdb_id', 'PDB', 'pdb', 'PDB_ID', 'structure_id']
        pdb_column = None
        
        for col in pdb_columns:
            if col in df.columns:
                pdb_column = col
                break
        
        if pdb_column:
            for pdb_id in df[pdb_column]:
                if isinstance(pdb_id, str) and len(pdb_id) >= 4:
                    clean_id = pdb_id[:4].upper()
                    if clean_id.isalnum():
                        pdb_ids.add(clean_id)
        
        return list(pdb_ids)
    
    def _extract_pdb_ids_from_results(self, results_file: str) -> List[str]:
        """Extract PDB IDs from kinase analysis results file"""
        pdb_ids = set()
        
        try:
            with open(results_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("# Analysis for"):
                        # Extract PDB ID from header
                        pdb_id = line.split()[-1].upper()
                        if len(pdb_id) == 4:
                            pdb_ids.add(pdb_id)
                    elif line and not line.startswith("#"):
                        # Extract PDB ID from data lines
                        parts = line.split()
                        if parts:
                            pdb_id = parts[0].upper()
                            if len(pdb_id) == 4:
                                pdb_ids.add(pdb_id)
        except Exception as e:
            self.logger.error(f"Error reading results file: {e}")
        
        return list(pdb_ids)
    
    def _delete_cif_files(self, cif_files: Dict[str, str]):
        """Delete downloaded CIF files
        
        Args:
            cif_files: Dictionary mapping structure IDs to CIF file paths
        """
        import os
        deleted_count = 0
        for structure_id, cif_path in cif_files.items():
            try:
                if os.path.exists(cif_path):
                    os.remove(cif_path)
                    deleted_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to delete CIF file for {structure_id}: {e}")
        
        self.logger.info(f"Deleted {deleted_count} CIF files")
    
    def process_pdb_structures_in_batches(self, pdb_ids: List[str], output_csv: str, 
                                         minimal_output: bool = False, 
                                         delete_cif_after: bool = False,
                                         batch_size: int = 100) -> pd.DataFrame:
        """
        Process PDB structures in batches, saving results incrementally
        
        Args:
            pdb_ids: List of PDB IDs to process
            output_csv: Output CSV file path
            minimal_output: If True, create minimal CSV with essential columns only
            delete_cif_after: If True, delete downloaded CIF files after processing
            batch_size: Number of PDB IDs to process in each batch
            
        Returns:
            DataFrame with combined results
        """
        self.logger.info(f"Processing {len(pdb_ids)} PDB IDs in batches of {batch_size}")
        
        # Split PDB IDs into batches
        batches = [pdb_ids[i:i + batch_size] for i in range(0, len(pdb_ids), batch_size)]
        self.logger.info(f"Split into {len(batches)} batches")
        
        all_results = []
        file_exists = False
        
        # Process each batch
        for i, batch in enumerate(batches):
            self.logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} PDB IDs)")
            
            # Process this batch
            batch_output = "" if i == 0 else None  # Only write header for first batch
            df_batch = self.process_pdb_structures(batch, batch_output, minimal_output, delete_cif_after)
            
            if len(df_batch) > 0:
                # Append results to the output file
                if output_csv:
                    mode = 'w' if i == 0 else 'a'
                    header = i == 0  # Only write header for first batch
                    df_batch.to_csv(output_csv, mode=mode, header=header, index=False)
                    self.logger.info(f"Batch {i+1} results {'written to' if i == 0 else 'appended to'} {output_csv}")
                
                # Add to combined results
                all_results.append(df_batch)
        
        # Combine all results
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _extract_chain_from_cif(self, cif_file: str, chain_id: str, output_file: str) -> bool:
        """
        Extract a single chain from a CIF file and save it
        
        Args:
            cif_file: Path to input CIF file
            chain_id: Chain ID to extract
            output_file: Path to output CIF file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from Bio import PDB
            from Bio.PDB.mmcifio import MMCIFIO
            
            parser = PDB.MMCIFParser(QUIET=True)
            structure = parser.get_structure("structure", cif_file)
            
            # Create a new structure with only the requested chain
            new_structure = PDB.Structure.Structure("structure")
            new_model = PDB.Model.Model(0)
            new_structure.add(new_model)
            
            # Find and copy the chain
            for model in structure:
                if chain_id in model:
                    chain = model[chain_id]
                    new_model.add(chain)
                    break
            else:
                self.logger.warning(f"Chain {chain_id} not found in {cif_file}")
                return False
            
            # Write the new structure
            io = MMCIFIO()
            io.set_structure(new_structure)
            io.save(str(output_file))
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to extract chain {chain_id} from {cif_file}: {e}")
            return False
    
    def download_chain_files(self, output_csv: str):
        """
        Download and extract chain files from CSV results
        
        Args:
            output_csv: Path to output CSV file to extract chain information from
        """
        try:
            df = pd.read_csv(output_csv)
            if 'structure_id' not in df.columns or 'chain_id' not in df.columns:
                self.logger.warning("CSV file missing 'structure_id' or 'chain_id' columns, skipping chain download")
                return
            
            # Get unique structure-chain pairs
            chains_to_extract = df[['structure_id', 'chain_id']].drop_duplicates()
            
            # Create output directory for chain CIF files
            parent_dir = self.work_dir.parent
            preserved_cif_dir = parent_dir / "cif_files"
            preserved_cif_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"Downloading and extracting {len(chains_to_extract)} chains from CSV results...")
            
            # Temporary directory for full CIF downloads
            temp_download_dir = self.work_dir.parent / "temp_cif_downloads"
            temp_download_dir.mkdir(exist_ok=True)
            temp_downloader = CIFDownloader(temp_download_dir, logger=self.logger)
            
            # Download and extract each chain
            for idx, row in chains_to_extract.iterrows():
                structure_id = str(row['structure_id']).upper()
                chain_id = str(row['chain_id'])
                
                # Download full CIF
                cif_file = temp_downloader.download_single(structure_id)
                if not cif_file:
                    self.logger.warning(f"Failed to download {structure_id}")
                    continue
                
                # Extract chain and save with naming format {structure_id}_{chain_id}.cif
                output_file = preserved_cif_dir / f"{structure_id}_{chain_id}.cif"
                if self._extract_chain_from_cif(cif_file, chain_id, str(output_file)):
                    self.logger.info(f"  ✓ Extracted {structure_id}_{chain_id}.cif")
                else:
                    self.logger.warning(f"  ✗ Failed to extract {structure_id}_{chain_id}")
            
            # Clean up temporary download directory
            import shutil
            if temp_download_dir.exists():
                shutil.rmtree(temp_download_dir)
            
            self.logger.info(f"Preserved {len(list(preserved_cif_dir.glob('*.cif')))} chain CIF files in: {preserved_cif_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to download chain files from CSV: {e}")
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
            self.logger.info(f"Cleaned up work directory: {self.work_dir}")


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Kinase Analysis Pipeline")
    parser.add_argument("-id", "--pdb-ids", nargs="+", help="List of PDB IDs to process")
    parser.add_argument("-ff", "--pdb-folder", help="Folder containing CIF files to process directly")
    parser.add_argument("-f", "--input-file", help="Input file with PDB IDs")
    parser.add_argument("--file-type", choices=['pdb_list', 'csv', 'results_file'], 
                       default='pdb_list', help="Type of input file")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    parser.add_argument("--work-dir", default="kinase_pipeline_work", help="Working directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--minimal", action="store_false", dest="comprehensive", 
                       help="Create comprehensive CSV with all columns (default: minimal)")
    parser.add_argument("-c", "--cofolding", action="store_true", dest="cofolding_folder",
                       help="Treat pdb-folder contents as co-folding models")
    parser.add_argument("--no-cleanup", action="store_false", dest="cleanup", 
                       help="Don't clean up temporary files after processing (default: cleanup)")
    parser.add_argument("--keep-cif", action="store_false", dest="delete_cif", 
                       help="Keep downloaded CIF files after processing (default: delete)")
    parser.add_argument("--batch-size", type=int, default=0, help="Process PDB IDs in batches of this size (0 = no batching)")
    
    args = parser.parse_args()
    
    if not args.pdb_ids and not args.pdb_folder and not args.input_file:
        print("Error: Must provide -id/--pdb-ids, -ff/--pdb-folder, or -f/--input-file")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = KinaseAnalysisPipeline(args.work_dir, verbose=args.verbose)
    
    # Set keep_cif_files based on delete_cif flag (inverted logic)
    # If delete_cif is False (--keep-cif was used), we want to keep CIF files
    pipeline.keep_cif_files = not args.delete_cif
    
    try:
        # Use comprehensive flag (which defaults to False) to determine if we want minimal output (True)
        minimal_output = not args.comprehensive
        
        if args.input_file:
            df = pipeline.process_from_file(args.input_file, args.output, args.file_type, minimal_output, args.delete_cif, args.batch_size)
        elif args.pdb_folder and args.cofolding_folder:
            # Process co-folding structures
            df = pipeline.process_cofolding_structures(args.pdb_folder, args.output, minimal_output)
        elif args.pdb_folder:
            # Process regular PDB structures from folder
            cif_files = pipeline._find_cif_files_in_folder(args.pdb_folder)
            pdb_ids = list(cif_files.keys())
            df = pipeline.process_pdb_structures(pdb_ids, args.output, minimal_output, args.delete_cif)
        elif args.pdb_ids:
            # Process PDB IDs
            df = pipeline.process_pdb_structures(args.pdb_ids, args.output, minimal_output, args.delete_cif)
        else:
            print("Error: No input provided")
            sys.exit(1)
        
        print(f"\nPipeline completed successfully!")
        print(f"Processed {len(df)} kinase chains")
        print(f"Results saved to: {args.output}")
        
        # Print summary statistics
        if len(df) > 0:
            print(f"\nSummary:")
            print(f"- Unique structures: {df['structure_id'].nunique()}")
            if 'kinase_family' in df.columns:
                print(f"- Kinase families found: {df['kinase_family'].nunique()}")
            if 'model_type' in df.columns:
                experimental = sum(df['model_type'] == 'experimental')
                cofolding = sum(df['model_type'] == 'cofolding')
                print(f"- Experimental structures: {experimental}")
                print(f"- Co-folding models: {cofolding}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)
    
    finally:
        if args.cleanup:
            pipeline.cleanup()
            # If keep_cif_files is True, download chain files after cleanup
            if pipeline.keep_cif_files:
                pipeline.download_chain_files(args.output)


if __name__ == "__main__":
    main()

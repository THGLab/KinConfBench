"""
PDB API extractor for structure-level metadata
"""

import requests
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union


class PDBAPIExtractor:
    """Extract structure-level metadata from PDB API"""
    
    def __init__(self, logger=None):
        """
        Initialize the extractor
        
        Args:
            logger: Logger instance for logging messages
        """
        self.graphql_url = "https://data.rcsb.org/graphql"
        self.logger = logger
    
    def get_batch_data(self, pdb_ids: List[str]) -> Dict[str, Dict]:
        """
        Get PDB data for multiple PDB IDs
        
        Args:
            pdb_ids: List of PDB identifiers
            
        Returns:
            Dict with keys as pdb_id and values as metadata dicts
        """
        results = {}
        chunk_size = 50
        for i in range(0, len(pdb_ids), chunk_size):
            chunk = pdb_ids[i:i + chunk_size]
            chunk_results = self._get_chunk_data(chunk)
            results.update(chunk_results)
        
        return results
    
    def get_single_data(self, pdb_id: str) -> Dict:
        """
        Get PDB data for a single PDB ID
        
        Args:
            pdb_id: PDB identifier
            
        Returns:
            Metadata dictionary for the structure
        """
        result = self.get_batch_data([pdb_id])
        return result.get(pdb_id.upper(), {})
    
    def _get_chunk_data(self, pdb_ids: List[str]) -> Dict[str, Dict]:
        """Get data for a chunk of PDB IDs"""
        query = """
        query GetPDBData($entry_ids: [String!]!) {
          entries(entry_ids: $entry_ids) {
            rcsb_id
            rcsb_accession_info {
              initial_release_date
              deposit_date
            }
            exptl {
              method
            }
            refine {
              ls_d_res_high
              ls_R_factor_R_work
              ls_R_factor_R_free
            }
            em_3d_reconstruction {
                resolution
            }
            struct {
              title
            }
            polymer_entities {
              entity_poly {
                pdbx_strand_id
              }
              rcsb_entity_source_organism {
                scientific_name
              }
              uniprots {
                rcsb_id
                rcsb_uniprot_accession
              }
            }
          }
        }
        """
        
        variables = {"entry_ids": [pdb_id.upper() for pdb_id in pdb_ids]}
        
        try:
            response = requests.post(
                self.graphql_url,
                json={"query": query, "variables": variables},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            results = {}
            
            if 'data' in data and 'entries' in data['data']:
                for entry in data['data']['entries']:
                    pdb_id = entry['rcsb_id']
                    
                    result = {
                        'Method': '',
                        'Resolution': '',
                        'Rfac': '',
                        'FreeRfac': '',
                        'Deposition_Date': '',
                        'ProteinName': '',
                        'Organism': '',
                        'uniprot_id_pdb_api': {}  # Dictionary to store chain_id -> uniprot_id mapping
                    }
                    
                    # Extract data
                    if entry.get('rcsb_accession_info'):
                        date = (entry['rcsb_accession_info'].get('initial_release_date') or 
                               entry['rcsb_accession_info'].get('deposit_date'))
                        if date:
                            result['Deposition_Date'] = date
                    
                    if entry.get('exptl') and len(entry['exptl']) > 0:
                        result['Method'] = entry['exptl'][0].get('method', '')
                    
                    if entry.get('refine') and len(entry['refine']) > 0:
                        refine_data = entry['refine'][0]
                        result['Resolution'] = refine_data.get('ls_d_res_high', '')
                        result['Rfac'] = refine_data.get('ls_R_factor_R_work', '')
                        result['FreeRfac'] = refine_data.get('ls_R_factor_R_free', '')
                    
                    if entry.get('em_3d_reconstruction') and len(entry['em_3d_reconstruction']) > 0:
                        result['Resolution'] = entry['em_3d_reconstruction'][0].get('resolution', '')
                    
                    if entry.get('struct') and entry['struct'].get('title'):
                        result['ProteinName'] = entry['struct']['title']
                    
                    if entry.get('polymer_entities'):
                        for entity in entry['polymer_entities']:
                            # Extract organism information
                            if entity.get('rcsb_entity_source_organism'):
                                organism = entity['rcsb_entity_source_organism'][0].get('scientific_name', '')
                                if organism:
                                    result['Organism'] = organism
                            
                            # Extract UniProt IDs and map to chains
                            if entity.get('uniprots') and entity.get('entity_poly'):
                                chains = entity.get('entity_poly', {}).get('pdbx_strand_id', '')
                                if chains:
                                    # Convert comma-separated chain list to individual chains
                                    chain_list = [c.strip() for c in chains.split(',')]
                                    
                                    # Get UniProt ID from entity
                                    uniprot_id = None
                                    for uniprot_entry in entity.get('uniprots', []):
                                        if uniprot_entry.get('rcsb_uniprot_accession'):
                                            uniprot_id = uniprot_entry.get('rcsb_uniprot_accession')
                                            break
                                    
                                    # Map UniProt ID to each chain
                                    if uniprot_id:
                                        for chain_id in chain_list:
                                            result['uniprot_id_pdb_api'][chain_id] = uniprot_id
                    
                    results[pdb_id] = result
            
            return results
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error querying PDB API: {e}")
            else:
                print(f"Error querying PDB API: {e}")
            return {}

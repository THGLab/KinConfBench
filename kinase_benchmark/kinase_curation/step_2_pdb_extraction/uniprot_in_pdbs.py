#!/usr/bin/env python
"""
Example usage of the uniprot_pdb_search module
"""

import os
import sys
import requests
import pandas as pd
from typing import List, Dict, Any

def search_pdb_by_uniprot(uniprot_id: str) -> List[Dict[str, Any]]:
    """
    Search for PDB structures that correspond to a given UniProt ID
    
    Args:
        uniprot_id: UniProt accession ID
            
    Returns:
        List of dictionaries containing basic PDB structure information
    """
    # Construct the query to find structures with the given UniProt ID
    query = """{
  "query": {
    "type": "group",
    "logical_operator": "and",
    "nodes": [
      {
        "type": "terminal",
        "service": "text",
        "parameters": {
          "operator": "exact_match",
          "value": "UNIPROT_ID",
          "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession"
        }
      },
      {
        "type": "terminal",
        "service": "text",
        "parameters": {
          "operator": "exact_match",
          "value": "UniProt",
          "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name"
        }
      }
    ]
  },
  "request_options": {
    "return_all_hits": true
  },
  "return_type": "polymer_entity"
}"""
    # Convert the query to a JSON string and then URL encode it
    query_url = f"https://search.rcsb.org/rcsbsearch/v2/query?json={query.replace('UNIPROT_ID', uniprot_id)}"
    
    try:
        # Execute the query
        response = requests.get(query_url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        all_pdb_ids = []
        
        if 'result_set' in data:
            all_pdb_ids = [entry['identifier'] for entry in data['result_set']]
        
        if not all_pdb_ids:
            print(f"No PDB structures found for UniProt ID: {uniprot_id}")
            return []
        
        return all_pdb_ids
    
    except requests.RequestException as e:
        print(f"Error querying PDB for UniProt ID {uniprot_id}: {e}")
        return []

def process_uniprot_csv(csv_file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process a CSV file containing UniProt IDs and search for PDB structures for each ID
    
    Args:
        csv_file_path: Path to the CSV file containing UniProt IDs in a column named 'uniprot_id'
            
    Returns:
        Dictionary with UniProt IDs as keys and lists of PDB structures as values
    """
    # Read the CSV file using pandas
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {}
    
    # Check if 'uniprot_id' column exists
    if 'uniprot_id' not in df.columns:
        print("Error: CSV file must contain a column named 'uniprot_id'")
        return {}
    
    # Extract unique UniProt IDs
    uniprot_ids = df['uniprot_id'].unique()
    print(f"Found {len(uniprot_ids)} unique UniProt IDs in the CSV file")
    
    # Search for PDB structures for each UniProt ID
    results = {}
    for uniprot_id in uniprot_ids:
        print(f"Searching for structures for UniProt ID: {uniprot_id}")
        structures = search_pdb_by_uniprot(uniprot_id)
        results[uniprot_id] = structures
        print(f"Found {len(structures)} structures for {uniprot_id}")
    
    return results

if __name__ == "__main__":
    # Check if we should run a specific example
    results = process_uniprot_csv("../step_1_uniprot_extraction/step_1_kinase_domain_info.csv")
    with open("kinase_pdb_ids.txt", "w") as f:
        for pdb_ids in results.values():
            for pdb_id in pdb_ids:
                f.write(f"{pdb_id.split("_")[0]}\n")
    
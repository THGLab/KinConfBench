#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to extract kinase sequences from the dunbrack_active_kinase_mapping.txt file.
Uses SwissProt_ID|UniProt_ID as the FASTA header.
Extracts kinase domain sequences that are fully contained within the reference range.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Union, Tuple

def get_kinase_domains(uniprot_id: str) -> List[Tuple[str, int, int]]:
    """
    Get all kinase domains for a protein.
    
    Args:
        uniprot_id (str): The UniProt ID of the protein.
    
    Returns:
        List[Tuple[str, int, int]]: List of tuples containing (domain_name, start_pos, end_pos)
    """
    # Common names for kinase domains in UniProt
    kinase_domain_names = [
        "protein kinase", 
        "kinase", 
        "catalytic domain", 
        "tyrosine-protein kinase", 
        "serine/threonine-protein kinase"
    ]
    
    domains = []
    
    # UniProt API URL
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    
    try:
        # Make request to UniProt API
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse JSON response
        data = response.json()
        
        # Extract kinase domains
        for feature in data.get("features", []):
            feature_type = feature.get("type", "")
            
            # Check if the feature is a domain
            if feature_type.lower() in ["domain", "region"]:
                domain_description = feature.get("description", "Unknown domain")
                
                # Check if domain is a kinase domain
                is_kinase_domain = False
                for kinase_name in kinase_domain_names:
                    if kinase_name.lower() in domain_description.lower():
                        is_kinase_domain = True
                        break
                
                if is_kinase_domain:
                    # Get domain boundaries
                    start = feature.get("location", {}).get("start", {}).get("value")
                    end = feature.get("location", {}).get("end", {}).get("value")
                    
                    if start is not None and end is not None:
                        domains.append((domain_description, start, end))
        
        return domains
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from UniProt: {e}")
        return []
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing UniProt response: {e}")
        return []

def extract_sequence_range(uniprot_id: str, start_pos: int, end_pos: int) -> Optional[str]:
    """
    Extract a protein sequence within a specific range of residues.
    
    Args:
        uniprot_id (str): The UniProt ID of the protein.
        start_pos (int): The starting position (1-based, inclusive).
        end_pos (int): The ending position (1-based, inclusive).
    
    Returns:
        str or None: The sequence within the specified range if successful, None otherwise.
    """
    # UniProt API URL
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    
    try:
        # Make request to UniProt API
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse JSON response
        data = response.json()
        
        # Extract full protein sequence
        full_sequence = data.get("sequence", {}).get("value", "")
        if not full_sequence:
            print(f"No sequence found for UniProt ID: {uniprot_id}")
            return None
        
        # Validate positions
        if start_pos < 1 or end_pos > len(full_sequence) or start_pos > end_pos:
            print(f"Invalid range: {start_pos}-{end_pos}. Protein length is {len(full_sequence)}.")
            return None
        
        # Extract sequence within range (UniProt positions are 1-based)
        return full_sequence[start_pos-1:end_pos]
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from UniProt: {e}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing UniProt response: {e}")
        return None

def is_range_contained(inner_start: int, inner_end: int, outer_start: int, outer_end: int) -> bool:
    """
    Check if the first range is fully contained within the second range.
    
    Args:
        inner_start (int): Start of inner range
        inner_end (int): End of inner range
        outer_start (int): Start of outer range
        outer_end (int): End of outer range
    
    Returns:
        bool: True if inner range is fully contained within outer range, False otherwise
    """
    return inner_start >= outer_start and inner_end <= outer_end

def process_dunbrack_kinase_file(input_file: str, output_file: str) -> None:
    """
    Process the dunbrack_active_kinase_mapping.txt file and extract kinase domain sequences.
    Uses SwissProt_ID|UniProt_ID as the FASTA header.
    Only extracts kinase domains that are fully contained within the reference range.
    
    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output FASTA file.
    """
    results = {}
    success_count = 0
    failure_count = 0
    not_contained_count = 0
    first_entry = True
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    # Read input file
    with open(input_file, 'r') as f:
        # Read the first line as header
        header = f.readline().strip().split()
        
        # Find column indices
        try:
            uniprot_idx = header.index("Uniprot")
            swissprot_idx = header.index("SwissProt_ID")
            start_idx = header.index("Kinase_Start")
            end_idx = header.index("Kinase_End")
        except ValueError as e:
            print(f"Error: Required column not found in header: {e}")
            print(f"Expected header to contain 'Uniprot', 'SwissProt_ID', 'Kinase_Start', and 'Kinase_End'")
            print(f"Found header: {header}")
            return
        
        # Process each data line
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # Parse line
            parts = line.split()
            if len(parts) <= max(uniprot_idx, swissprot_idx, start_idx, end_idx):
                print(f"  Warning: Line {i+1} has insufficient columns")
                failure_count += 1
                continue
            
            uniprot_id = parts[uniprot_idx]
            swissprot_id = parts[swissprot_idx]
            try:
                ref_start_pos = int(parts[start_idx])
                ref_end_pos = int(parts[end_idx])
            except ValueError:
                print(f"  Warning: Line {i+1} has invalid positions. Expected integers for Kinase_Start and Kinase_End")
                failure_count += 1
                continue
            
            print(f"Processing {i+1}: {swissprot_id}|{uniprot_id} (reference range: {ref_start_pos}-{ref_end_pos})")
            
            # Get all kinase domains
            kinase_domains = get_kinase_domains(uniprot_id)
            
            if not kinase_domains:
                print(f"  Warning: No kinase domains found for {swissprot_id}|{uniprot_id}")
                failure_count += 1
                continue
            
            # Find kinase domains fully contained within the reference range
            contained_domains = []
            for domain_name, domain_start, domain_end in kinase_domains:
                if is_range_contained(domain_start, domain_end, ref_start_pos, ref_end_pos):
                    contained_domains.append((domain_name, domain_start, domain_end))
            
            if not contained_domains:
                print(f"  Warning: No kinase domains fully contained within reference range {ref_start_pos}-{ref_end_pos} for {swissprot_id}|{uniprot_id}")
                not_contained_count += 1
                continue
            
            # Extract sequences for contained domains
            for domain_name, domain_start, domain_end in contained_domains:
                print(f"  Found contained kinase domain: {domain_name} ({domain_start}-{domain_end})")
                
                # Extract sequence for kinase domain
                sequence = extract_sequence_range(uniprot_id, domain_start, domain_end)
                
                if sequence:
                    # Write to output file with header as SwissProt_ID|UniProt_ID
                    fasta_header = f">{swissprot_id}|{uniprot_id}"
                    
                    # Write to file
                    mode = 'w' if first_entry else 'a'
                    with open(output_file, mode) as out_f:
                        out_f.write(f"{fasta_header}\n")
                        # Write sequence in one line
                        out_f.write(f"{sequence}\n")
                    
                    first_entry = False
                    if uniprot_id not in results:
                        results[uniprot_id] = []
                    results[uniprot_id].append((domain_name, sequence))
                    success_count += 1
                else:
                    print(f"  Warning: Failed to extract sequence for {swissprot_id}|{uniprot_id} domain {domain_name} ({domain_start}-{domain_end})")
                    failure_count += 1
    
    print(f"\nProcessing complete:")
    print(f"  {success_count} kinase domain sequences extracted")
    print(f"  {not_contained_count} entries had no kinase domains fully contained within reference range")
    print(f"  {failure_count} sequences failed to extract")
    print(f"Results written to: {output_file}")

def main():
    input_file = "dunbrack_active_kinase_mapping.txt"
    output_file = "dunbrack_active_kinases.fasta"
    
    print(f"Processing {input_file} to extract kinase domain sequences...")
    process_dunbrack_kinase_file(input_file, output_file)
    print(f"Done! Sequences written to {output_file}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to extract kinase domain information from the dunbrack_active_kinase_mapping.txt file.
Creates a CSV file with UniProt ID, start position, and end position of kinase domains.
Only includes kinase domains that are contained within the reference range.
"""

import os
import json
import requests
import csv
from typing import List, Tuple, Optional

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

def is_domain_contained_in_range(domain_start: int, domain_end: int, range_start: int, range_end: int) -> bool:
    """
    Check if the domain is fully contained within the reference range.
    
    Args:
        domain_start (int): Start of domain
        domain_end (int): End of domain
        range_start (int): Start of reference range
        range_end (int): End of reference range
    
    Returns:
        bool: True if domain is fully contained within reference range, False otherwise
    """
    return domain_start >= range_start and domain_end <= range_end

def extract_kinase_domain_info(input_file: str, output_file: str) -> None:
    """
    Extract kinase domain information from the dunbrack_active_kinase_mapping.txt file.
    Creates a CSV file with UniProt ID, start position, and end position of kinase domains.
    Only includes kinase domains that are contained within the reference range.
    
    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output CSV file.
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open output file and write header
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['uniprot_id', 'gene_name', 'uniprot_start', 'uniprot_end'])
        
        # Read input file
        with open(input_file, 'r') as f:
            # Read the first line as header
            header = f.readline().strip().split()
            
            # Find column indices
            try:
                uniprot_idx = header.index("Uniprot")
                gene_name_idx = header.index("SwissProt_ID")
                start_idx = header.index("Kinase_Start")
                end_idx = header.index("Kinase_End")
            except ValueError as e:
                print(f"Error: Required column not found in header: {e}")
                print(f"Expected header to contain 'Uniprot', 'SwissProt_ID', 'Kinase_Start', and 'Kinase_End'")
                print(f"Found header: {header}")
                return
            
            # Process each data line
            success_count = 0
            failure_count = 0
            not_contained_count = 0
            
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                # Parse line
                parts = line.split()
                if len(parts) <= max(uniprot_idx, start_idx, end_idx):
                    print(f"  Warning: Line {i+1} has insufficient columns")
                    failure_count += 1
                    continue
                
                uniprot_id = parts[uniprot_idx]
                gene_name = parts[gene_name_idx]
                try:
                    ref_start_pos = int(parts[start_idx])
                    ref_end_pos = int(parts[end_idx])
                except ValueError:
                    print(f"  Warning: Line {i+1} has invalid positions. Expected integers for Kinase_Start and Kinase_End")
                    failure_count += 1
                    continue
                
                print(f"Processing {i+1}: {uniprot_id} (reference range: {ref_start_pos}-{ref_end_pos})")
                
                # Get all kinase domains
                kinase_domains = get_kinase_domains(uniprot_id)
                
                if not kinase_domains:
                    print(f"  Warning: No kinase domains found for {uniprot_id}")
                    failure_count += 1
                    continue
                
                print(f"  Found {len(kinase_domains)} kinase domains for {uniprot_id}:")
                for domain_name, domain_start, domain_end in kinase_domains:
                    print(f"    - {domain_name}: {domain_start}-{domain_end}")
                
                # Find kinase domains that are contained within the reference range
                contained_domains = []
                for domain_name, domain_start, domain_end in kinase_domains:
                    if is_domain_contained_in_range(domain_start, domain_end, ref_start_pos, ref_end_pos):
                        contained_domains.append((domain_name, domain_start, domain_end))
                
                if not contained_domains:
                    print(f"  Warning: No kinase domains are contained within reference range {ref_start_pos}-{ref_end_pos} for {uniprot_id}")
                    not_contained_count += 1
                else:
                    print(f"  Found {len(contained_domains)} kinase domains contained within reference range")
                    # Write domain information to CSV
                    for domain_name, domain_start, domain_end in contained_domains:
                        print(f"  Writing to CSV: {uniprot_id}, {domain_start}, {domain_end}")
                        writer.writerow([uniprot_id, gene_name, domain_start, domain_end])
                        success_count += 1
    
    print(f"\nProcessing complete:")
    print(f"  {success_count} kinase domains extracted")
    print(f"  {not_contained_count} entries had no kinase domains contained within reference range")
    print(f"  {failure_count} entries failed to process")
    print(f"Results written to: {output_file}")

def main():
    input_file = "dunbrack_active_kinase_mapping.txt"
    output_file = "step_1_kinase_domain_info.csv"
    
    print(f"Processing {input_file} to extract kinase domain information...")
    extract_kinase_domain_info(input_file, output_file)
    print(f"Done! Information written to {output_file}")

if __name__ == "__main__":
    main()

import pandas as pd
import pickle
import requests
import os
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem

def detect_csv_format(df):
    """
    Detect the format of the CSV file and return appropriate column mappings.
    Requires either:
    1. 'SMILES' and 'names' columns
    2. 'ligands' column
    
    Returns:
        dict: Contains 'format', 'identifier_col', 'smiles_col', 'ligand_col' keys
    """
    columns = df.columns.tolist()
    
    # Check for exact SMILES + names format (format 1)
    if 'SMILES' in columns and 'names' in columns:
        return {
            'format': 'smiles_direct',
            'identifier_col': 'names',
            'smiles_col': 'SMILES',
            'ligand_col': None
        }
    
    # Check for ligands column (format 2: CCD codes)
    if 'ligands' in columns:
        return {
            'format': 'ccd_codes',
            'identifier_col': None,
            'smiles_col': None,
            'ligand_col': 'ligands'
        }
    
    raise ValueError("CSV must contain either ('SMILES' and 'names' columns) or ('ligands' column).")

def fetch_smiles_from_rcsb(ccd_code):
    """Fetch SMILES from RCSB API for a given CCD code."""
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ccd_code}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data["rcsb_chem_comp_descriptor"]["smiles"]
    except Exception as e:
        print(f"❌ Failed to fetch {ccd_code}: {e}")
        return None

def extract_ccd_codes_from_ligands(df, ligand_col):
    """Extract CCD codes from ligands column."""
    all_ligand_codes = set()
    
    for entry in df[ligand_col].dropna():
        if str(entry).strip() == "No_ligand":
            continue
        for ligand in str(entry).split(";"):
            ligand_code = ligand.strip().split(":")[0].upper()
            if ligand_code and "peptide" not in ligand_code.lower():  # Only add non-empty codes
                all_ligand_codes.add(ligand_code)
    
    return all_ligand_codes

def process_smiles_direct_format(df, format_info, existing_dict=None):
    """Process CSV with direct SMILES column."""
    if existing_dict is None:
        existing_dict = {}
    
    new_ligands = {}
    identifier_col = format_info['identifier_col']
    smiles_col = format_info['smiles_col']
    rdkit_failed = []
    flagged_entries = []
    skipped_entries = []
    
    print(f"📊 Processing direct SMILES format using columns: {identifier_col} -> {smiles_col}")
    
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        if pd.isna(smiles) or not str(smiles).strip():
            continue
            
        # Use identifier if available, otherwise use index
        if identifier_col and not pd.isna(row[identifier_col]):
            identifier = str(row[identifier_col]).strip()
        else:
            identifier = f"compound_{idx}"
        
        smiles_clean = str(smiles).strip()
        
        # Check if identifier already exists in existing dictionary
        if identifier in existing_dict:
            if existing_dict[identifier] == smiles_clean:
                # Same SMILES, skip
                skipped_entries.append(identifier)
                print(f"  = Skipping {identifier}: identical SMILES")
                continue
            else:
                # Different SMILES, flag it
                flagged_entries.append((identifier, existing_dict[identifier], smiles_clean))
                print(f"  ! Flagged {identifier}: different SMILES")
                print(f"    Existing: {existing_dict[identifier][:50]}{'...' if len(existing_dict[identifier]) > 50 else ''}")
                print(f"    New:      {smiles_clean[:50]}{'...' if len(smiles_clean) > 50 else ''}")
                # Still proceed with the new value
        
        # Basic RDKit validation: load molecule and try to embed
        try:
            mol = Chem.MolFromSmiles(smiles_clean)
            if mol is None:
                rdkit_failed.append((identifier, "Cannot parse SMILES"))
                print(f"⚠️ RDKit failed for {identifier}: Cannot parse SMILES")
            else:
                # Try to embed the molecule
                try:
                    mol_h = Chem.AddHs(mol)
                    result = AllChem.EmbedMolecule(mol_h, randomSeed=42)
                    if result == -1:
                        rdkit_failed.append((identifier, "Cannot embed molecule"))
                        print(f"⚠️ RDKit failed for {identifier}: Cannot embed molecule")
                    else:
                        new_ligands[identifier] = smiles_clean
                except Exception as e:
                    rdkit_failed.append((identifier, f"Embedding error: {str(e)}"))
                    print(f"⚠️ RDKit failed for {identifier}: {str(e)}")
        except Exception as e:
            rdkit_failed.append((identifier, f"RDKit error: {str(e)}"))
            print(f"⚠️ RDKit failed for {identifier}: {str(e)}")
    
    if rdkit_failed:
        print(f"\n🚨 RDKit validation failed for {len(rdkit_failed)} ligands:")
        for identifier, error in rdkit_failed[:5]:  # Show first 5
            print(f"   - {identifier}: {error}")
        if len(rdkit_failed) > 5:
            print(f"   ... and {len(rdkit_failed) - 5} more")
    
    if skipped_entries:
        print(f"\n✅ Skipped {len(skipped_entries)} entries with identical SMILES")
    
    if flagged_entries:
        print(f"\n🚨 Flagged {len(flagged_entries)} entries with different SMILES:")
        for identifier, old_smiles, new_smiles in flagged_entries[:3]:  # Show first 3
            print(f"   - {identifier}")
            print(f"     Old: {old_smiles[:40]}{'...' if len(old_smiles) > 40 else ''}")
            print(f"     New: {new_smiles[:40]}{'...' if len(new_smiles) > 40 else ''}")
        if len(flagged_entries) > 3:
            print(f"   ... and {len(flagged_entries) - 3} more")
        print("   WARNING: New values will overwrite existing ones!")
    
    return new_ligands

def process_ccd_codes_format(df, format_info, existing_dict=None):
    """Process CSV with CCD codes in ligands column."""
    if existing_dict is None:
        existing_dict = {}
    
    ligand_col = format_info['ligand_col']
    print(f"📊 Processing CCD codes format using column: {ligand_col}")
    
    # Extract all CCD codes
    all_ligand_codes = extract_ccd_codes_from_ligands(df, ligand_col)
    
    # Separate existing and new codes
    existing_codes = all_ligand_codes & existing_dict.keys()
    new_ligands = sorted(all_ligand_codes - existing_dict.keys())
    
    print(f"🔍 Found {len(existing_codes)} existing CCD codes (will skip)")
    print(f"🔍 Found {len(new_ligands)} new CCD codes to query from RCSB")
    
    # Report skipped existing codes
    if existing_codes:
        print(f"✅ Skipping {len(existing_codes)} existing CCD codes")
        for code in sorted(list(existing_codes))[:5]:  # Show first 5
            print(f"  = {code}")
        if len(existing_codes) > 5:
            print(f"  ... and {len(existing_codes) - 5} more")
    
    # Query RCSB for new ligands
    added_count = 0
    rdkit_failed = []
    flagged_entries = []
    
    for code in new_ligands:
        smiles = fetch_smiles_from_rcsb(code)
        if smiles:
            # Check if we somehow already have this code with different SMILES (shouldn't happen but safety check)
            if code in existing_dict and existing_dict[code] != smiles:
                flagged_entries.append((code, existing_dict[code], smiles))
                print(f"  ! Flagged {code}: different SMILES from RCSB")
                print(f"    Existing: {existing_dict[code][:50]}{'...' if len(existing_dict[code]) > 50 else ''}")
                print(f"    RCSB:     {smiles[:50]}{'...' if len(smiles) > 50 else ''}")
            
            # Basic RDKit validation: load molecule and try to embed
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    rdkit_failed.append((code, "Cannot parse SMILES"))
                    print(f"⚠️ RDKit failed for {code}: Cannot parse SMILES")
                else:
                    # Try to embed the molecule
                    try:
                        mol_h = Chem.AddHs(mol)
                        result = AllChem.EmbedMolecule(mol_h, randomSeed=42)
                        if result == -1:
                            rdkit_failed.append((code, "Cannot embed molecule"))
                            print(f"⚠️ RDKit failed for {code}: Cannot embed molecule")
                        else:
                            existing_dict[code] = smiles
                            added_count += 1
                    except Exception as e:
                        rdkit_failed.append((code, f"Embedding error: {str(e)}"))
                        print(f"⚠️ RDKit failed for {code}: {str(e)}")
            except Exception as e:
                rdkit_failed.append((code, f"RDKit error: {str(e)}"))
                print(f"⚠️ RDKit failed for {code}: {str(e)}")
    
    print(f"✅ Added {added_count} new ligands from RCSB.")
    
    if rdkit_failed:
        print(f"\n🚨 RDKit validation failed for {len(rdkit_failed)} CCD codes:")
        for code, error in rdkit_failed[:5]:  # Show first 5
            print(f"   - {code}: {error}")
        if len(rdkit_failed) > 5:
            print(f"   ... and {len(rdkit_failed) - 5} more")
    
    if flagged_entries:
        print(f"\n🚨 Flagged {len(flagged_entries)} CCD codes with different SMILES:")
        for code, old_smiles, new_smiles in flagged_entries[:3]:  # Show first 3
            print(f"   - {code}")
            print(f"     Old: {old_smiles[:40]}{'...' if len(old_smiles) > 40 else ''}")
            print(f"     New: {new_smiles[:40]}{'...' if len(new_smiles) > 40 else ''}")
        if len(flagged_entries) > 3:
            print(f"   ... and {len(flagged_entries) - 3} more")
        print("   WARNING: New values from RCSB will overwrite existing ones!")
    
    return existing_dict

def main():
    """Main processing function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate ligand SMILES dictionary from CSV')
    parser.add_argument('csv_file', help='Path to input CSV file')
    parser.add_argument('--output', '-o', default='examples/kinase_ligands.pkl',
                       help='Output pickle file path (default: examples/kinase_ligands.pkl)')
    
    args = parser.parse_args()
    csv_path = args.csv_file
    pkl_path = args.output
    
    # Load existing ligand dictionary if it exists
    ligand_dict = {}
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            ligand_dict = pickle.load(f)
        print(f"✅ Loaded {len(ligand_dict)} existing ligands from pickle.")
    else:
        print("⚠️ Pickle not found, starting with empty dictionary.")
    
    # Load and analyze CSV
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"📁 Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
    
    # Detect CSV format
    try:
        format_info = detect_csv_format(df)
        print(f"🔍 Detected format: {format_info['format']}")
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Process based on detected format
    if format_info['format'] == 'smiles_direct':
        new_ligands = process_smiles_direct_format(df, format_info, ligand_dict)
        # Merge with existing dictionary
        ligand_dict.update(new_ligands)
        print(f"✅ Added {len(new_ligands)} new ligands from direct SMILES.")
        
    elif format_info['format'] == 'ccd_codes':
        ligand_dict = process_ccd_codes_format(df, format_info, ligand_dict)
    
    # Clean up dictionary
    if "No_ligand" in ligand_dict:
        del ligand_dict["No_ligand"]
        print("🧹 Removed 'No_ligand' entry.")
    
    # Save updated pickle
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(ligand_dict, f)
    
    print(f"\n✅ Final dictionary contains {len(ligand_dict)} ligands.")
    print(f"💾 Updated pickle saved to: {pkl_path}")
    
    # Show sample entries
    if ligand_dict:
        print("\n📋 Sample entries:")
        for i, (key, value) in enumerate(list(ligand_dict.items())[:5]):
            print(f"  {key}: {value[:50]}{'...' if len(value) > 50 else ''}")

if __name__ == "__main__":
    main()

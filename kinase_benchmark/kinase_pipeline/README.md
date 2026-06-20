# Kinase Analysis Pipeline

A complete pipeline for processing kinase structures that:
1. Downloads CIF files from PDB (or uses local CIFs)
2. Runs kinase analysis (Kincore/HMM-based)
3. Extracts phosphorylation/mutation info from mmCIF headers
4. Queries PDB API for metadata (experimental structures only)
5. Combines everything into a CSV matching your target format

## Environment Setup

Create and activate the kinconfbench environment (see project root `env.yml`):

```bash
conda env create -f env.yml -n kinconfbench
conda activate kinconfbench
pip install -e .
```

Installing the package with `pip install -e .` avoids path/import errors when running the pipeline.

## Quick Start

### Process PDB IDs (Download and Annotate)

```bash
cd kinconfbench_code
conda activate kinconfbench
python kinase_benchmark/kinase_pipeline/main_pipeline.py -id 4XEY -o results.csv -v
```

This will:
- Download the 4XEY CIF file from PDB
- Run kinase analysis with HMM classification
- Extract phosphorylation and mutations from mmCIF
- Query PDB API for metadata
- Generate `results.csv`

### Process Cofolding Models (Local CIFs)

```bash
python kinase_benchmark/kinase_pipeline/main_pipeline.py -c -ff kinase_benchmark/data/ABL1_HUMAN_1N1_cofolding -o cofolding_analysis.csv -v
```

### Process Multiple PDB IDs

```python
from main_pipeline import KinaseAnalysisPipeline

pipeline = KinaseAnalysisPipeline()
df = pipeline.process_pdb_structures(["9OB4", "1ATP", "2SRC"], "results.csv")
```

## Key Features

### Phosphorylation Detection
- Extracts from `_struct_mod_residue` sections in mmCIF
- Identifies TPO, SEP, PTR, PHO modifications
- Links to specific residue positions

### Mutation Detection  
- Compares structure sequence vs reference sequence
- Uses `_entity_poly.pdbx_seq_one_letter_code` and `_struct_ref_seq.pdbx_db_seq_one_letter_code`
- Reports mutations in standard format (e.g., "A123V")

### Complete Workflow
- **CIF Download**: Automatic download from PDB/mmCIF servers
- **Kinase Analysis**: Wraps existing `kinase_state.py` functionality
- **mmCIF Parsing**: Direct extraction from structural headers
- **PDB API**: Metadata like resolution, R-factors, UniProt IDs
- **Gene Mapping**: Uses your existing UniProt mapping file

## Output Format

Matches your target CSV schema:
- Organism, Group, Gene, UniprotID, PDB, Method, Resolution
- Rfac, FreeRfac, SpatialLabel, DihedralLabel, ActivityLabel
- Chelix-Saltbr, ActLoop, Ligand, LigandType, DFG_Phe
- **Phosphorylation**, **Mutations**, ProteinName, Deposition_Date

## Architecture

```
kinase_pipeline/
├── main_pipeline.py           # Main orchestrator
├── workers/
│   ├── cif_downloader.py      # PDB file download
│   ├── mmcif_extractor.py     # Phosphorylation/mutations from mmCIF
│   ├── kinase_analyzer.py     # Kinase conformational state (Kincore)
│   └── pdb_api_extractor.py   # PDB API metadata
├── utils/
│   ├── cofolding_processor.py # Cofolding model preprocessing
│   └── constants.py           # Ligand/residue constants
└── kincore_funcs/             # Kinase analysis modules (Kincore-standalone2)
    ├── HMMs/                  # HMM profiles for kinase classification
    └── modules/               # run_hmmsearch, identify_group, etc.
```

## Requirements

- Python 3.10+ (or 3.8 per Kincore-standalone2)
- pandas, numpy, biopython, hmmer, pyyaml, requests
- HMM files in `kincore_funcs/HMMs/` (included)

## Command Line Usage

```bash
# Process PDB IDs (download CIFs, annotate)
python main_pipeline.py -id 4XEY 9OB4 -o results.csv -v

# Process cofolding CIFs in a folder
python main_pipeline.py -c -ff path/to/cif_folder -o cofolding_results.csv

# Minimal output (essential columns only)
python main_pipeline.py -id 4XEY -o results.csv --minimal

# Keep work directory (no cleanup)
python main_pipeline.py -id 4XEY -o results.csv --no-cleanup
```

## Testing with Sample Data

```bash
# PDB workflow (download 4XEY)
python kinase_benchmark/kinase_pipeline/main_pipeline.py -id 4XEY -o test_4xey.csv -v

# Cofolding workflow (local data)
python kinase_benchmark/kinase_pipeline/main_pipeline.py -c -ff kinase_benchmark/data/ABL1_HUMAN_1N1_cofolding -o test_cofolding.csv -v
```

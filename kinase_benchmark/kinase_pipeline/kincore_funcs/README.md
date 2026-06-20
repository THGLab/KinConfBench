# Kincore Functions

This folder is taken from [Kincore-standalone2](https://github.com/DunbrackLab/Kincore-standalone2/tree/main/modules).

## Dependencies

The kinase analysis pipeline (main_pipeline, step_3) requires:

- **Python 3.8–3.10** (Kincore recommends 3.8)
- **biopython** – PDB/mmCIF parsing, SearchIO for HMM output
- **hmmer** – provides `hmmsearch` for kinase family classification
- **pandas, numpy**

## Environment Setup

Create the kinconfbench conda environment from the project root:

```bash
conda env create -f env.yml -n kinconfbench
conda activate kinconfbench
pip install -e .
```

Or with a prefix path:

```bash
conda env create -f env.yml --prefix /path/to/envs/kinconfbench
conda activate /path/to/envs/kinconfbench
pip install -e .
```

To use a prefix env in step_3, set `KINCONFBENCH_ENV=""` and `KINCONFBENCH_PREFIX=/path/to/envs/kinconfbench` (see `cofolding_inference/envs.py`).

## HMM Files

Kinase HMM profiles are in `kincore_funcs/HMMs/`. The kinase_analyzer locates them at `kinase_pipeline/kincore_funcs/HMMs`. Ensure this path exists relative to the pipeline when running.

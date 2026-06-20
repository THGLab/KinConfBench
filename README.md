<div align="center">

# KinConfBench: A curated benchmark for cofolding models on kinase conformational states

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.04.07.716788v1-cf2228?style=flat&logo=biorxiv&logoColor=white)](https://www.biorxiv.org/content/10.64898/2026.04.07.716788v1)
[![Figshare](https://img.shields.io/badge/figshare-inference%20data-556472?style=flat&logo=figshare&logoColor=white)](https://doi.org/10.6084/m9.figshare.31986663)
[![License](https://img.shields.io/badge/License-UC%20Regents-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

</div>

---

## Overview

KinConfBench is a **reproducible workflow** for benchmarking protein–ligand **cofolding** on kinases. It wires together:

| | |
| :--- | :--- |
| **Inputs** | MSAs and YAML configs for **Boltz**, **Chai**, and **Protenix**, plus **RoseTTAFold-All-Atom** (RFAA) as a conda-only 4th backend |
| **Compute** | Batched inference, optional Boltz affinity, structured output under `predictions/` |
| **Labels** | Kinase-centric, KinCoRe-style annotation CSVs |
| **Paper metrics** | Optional **analysis** package: success filters, diversity, apo/holo splits, plots |

Everything assumes a **shared directory layout** (`predictions/`, `analysis_csvs/`, etc.) so steps compose cleanly from MSA generation through figures.

**Released inference archive:** Step-2 cofolding outputs (Boltz, Chai, Protenix) for the benchmark are on Figshare as [`10.6084/m9.figshare.31986663`](https://doi.org/10.6084/m9.figshare.31986663) (download and unpack to match the `kinconfbench_inference_data` layout used by the analysis scripts).

---

## Contents

- [Installation](#installation)
- [Pipeline](#pipeline)
- [Inference outputs (Figshare)](#inference-outputs-figshare)
- [Quick start](#quick-start-copy-paste)
- [Repository layout](#repository-layout)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

---

## Installation

### Prerequisites

- [`conda`](https://docs.conda.io/) or **Mamba** for the main **kinconfbench** environment.
- **Separate** conda environments for each cofolding backend (Boltz, Chai, Protenix, RoseTTAFold-All-Atom) used in Steps 1–2. Paths are centralized in [`cofolding_inference/envs.py`](cofolding_inference/envs.py); set `KINCONFBENCH_PREFIX` / `KINCONFBENCH_ENV` if you use a prefix install.
- **RoseTTAFold-All-Atom (RFAA)** is conda-only (compiled SE3Transformer + dgl) and builds its own MSAs/templates inline, so it has a dedicated build + run + database guide: [`cofolding_inference/RFAA_SETUP.md`](cofolding_inference/RFAA_SETUP.md).
- For kinase annotation (Step 3), install dependencies from [`env.yml`](env.yml) (e.g. Biopython, HMMER, pandas, RDKit where needed).

### Create the environment

From the directory that **contains** this `KinConfBench` folder (typical clone layout):

```bash
# Named env (recommended)
conda env create -f KinConfBench/env.yml -n kinconfbench
conda activate kinconfbench

# Or prefix install (set KINCONFBENCH_PREFIX for step_3; use KINCONFBENCH_ENV="" )
# conda env create -f KinConfBench/env.yml --prefix /path/to/envs/kinconfbench
# conda activate /path/to/envs/kinconfbench
```

### Install the package (editable)

```bash
pip install -e KinConfBench
```

For **prefix** installs with `step_3_kinase_cofolding_analysis.py`, set `KINCONFBENCH_PREFIX` and `KINCONFBENCH_ENV=""` so generated scripts use `conda run --prefix <path>`.

---

## Pipeline

```mermaid
flowchart LR
  A[1.0 MSAs] --> B[1.1 Ligands]
  B --> C[1.2 Config]
  C --> D[2 Inference]
  D --> E[3 Kinase labels]
  E --> F[Analysis optional]
```

| Step | Script | Role |
| :--- | :--- | :--- |
| **1.0** | [`step_1_0_generate_all_msas.py`](step_1_0_generate_all_msas.py) | Build method-specific MSA layouts for Boltz / Chai / Protenix. |
| **1.1** | [`step_1_1_generate_ligand_pkl.py`](step_1_1_generate_ligand_pkl.py) | Turn ligand tables (SMILES or CCD-style) into a merged `ligands.pkl`. |
| **1.2** | [`step_1_2_generate_config.py`](step_1_2_generate_config.py) | Protein–ligand combinations and YAML inference config. |
| **2** | [`step_2_run_all_inferences.py`](step_2_run_all_inferences.py) | Run inference (`--methods boltz chai protenix rfaa`), optional affinity; organize CIFs via [`cofolding_inference/gather_structural_files.py`](cofolding_inference/gather_structural_files.py). RFAA setup: [`cofolding_inference/RFAA_SETUP.md`](cofolding_inference/RFAA_SETUP.md). |
| **3** | [`step_3_kinase_cofolding_analysis.py`](step_3_kinase_cofolding_analysis.py) | CIF preprocessing and KinCoRe-style CSVs per prediction set (wraps `kinase_benchmark`). |

**Downstream analysis** (geometric filters, successful subsets, plots, pair metrics) lives under [`analysis/`](analysis/README.md), driven by [`analysis/run_pipeline.py`](analysis/run_pipeline.py).

---

## Inference outputs (Figshare)

Precomputed cofolding inference outputs for KinConfBench are archived on Figshare: **[https://doi.org/10.6084/m9.figshare.31986663](https://doi.org/10.6084/m9.figshare.31986663)**. Use this to reproduce paper analyses without re-running GPU inference; extract the archive so paths align with your local `kinconfbench_inference_data` (or equivalent) root expected by Step 3 and [`analysis/`](analysis/README.md).

---

## Quick start (copy-paste)

```bash
# 1.0 — MSAs
python KinConfBench/step_1_0_generate_all_msas.py proteins.fasta \
  --output-dir tmp/msa_generation --final-dir tmp/msas

# 1.1 — Ligand dictionary (skip if you already have ligands.pkl)
python KinConfBench/step_1_1_generate_ligand_pkl.py data.csv -o ligands.pkl

# 1.2 — Config + combinations pickle
python KinConfBench/step_1_2_generate_config.py \
  --csv_file data.csv --proteins_path proteins.pkl --msa_folder tmp/msas \
  --ligands_path ligands.pkl --output_path combinations.pkl --config_output_path config.yml

# 2 — Inference (default: boltz, chai, protenix; add rfaa for the 4th backend)
python KinConfBench/step_2_run_all_inferences.py config.yml \
  --output-folder inference_results --num-samples 5 --num-gpus 2 --predict-affinity
#   add --methods boltz chai protenix rfaa to include RoseTTAFold-All-Atom
#   (see cofolding_inference/RFAA_SETUP.md for the RFAA env + databases)

# 3 — Kinase analysis (kinconfbench env)
python KinConfBench/step_3_kinase_cofolding_analysis.py -d examples -v
bash examples/scripts/run_all_kinase_analyses.sh
```

If your shell `cwd` is already `KinConfBench/`, drop the `KinConfBench/` prefix on script paths.

---

## Repository layout

High-level map of what lives beside the step scripts:

```text
KinConfBench/
├── step_1_0_generate_all_msas.py
├── step_1_1_generate_ligand_pkl.py
├── step_1_2_generate_config.py
├── step_2_run_all_inferences.py
├── step_3_kinase_cofolding_analysis.py
├── cofolding_inference/          # Backend adapters, env paths, gather CIFs
├── kinase_benchmark/
│   ├── kinase_curation/          # Historical benchmark construction (UniProt → PDB → filters)
│   └── kinase_pipeline/          # Runtime labeling (main_pipeline, KinCoRe-derived code)
├── analysis/                     # Paper-style metrics and plots
├── data/                         # Released tables / small examples
├── env.yml
└── setup.py
```

---

## License

See [LICENSE](LICENSE) (UC Regents; research and not-for-profit use as described in the file).

---

## Acknowledgments

- Mia A. Rosenfeld for contributions to project conception, experimental design, and narrative framing.
- Matthew Welborn, Aleksander Durumeric, and Michael Irvin for helpful suggestions.
- Cofolding backends: [Boltz](https://github.com/jwohlwend/boltz), [Chai-1](https://github.com/chaidiscovery/chai-lab), [Protenix](https://github.com/bytedance/Protenix), [RoseTTAFold-All-Atom](https://github.com/baker-laboratory/RoseTTAFold-All-Atom).
- Kinase conformation labeling builds on ideas and code paths related to [Kincore-standalone2](https://github.com/DunbrackLab/Kincore-standalone2) (see [`kinase_benchmark/kinase_pipeline/kincore_funcs/README.md`](kinase_benchmark/kinase_pipeline/kincore_funcs/README.md)).

---

## Citation

If you use KinConfBench in your research, please cite the **preprint** ([bioRxiv](https://www.biorxiv.org/content/10.64898/2026.04.07.716788v1)) and KinCoRe software suites.

```bibtex
@misc{kinconfbench,
	title = {{KinConfBench}: {A} {Curated} {Benchmark} for {Cofolding} {Models} on {Kinase} {Conformational} {States}},
	url = {https://www.biorxiv.org/content/10.64898/2026.04.07.716788v1},
	doi = {10.64898/2026.04.07.716788},
	language = {en},
	publisher = {bioRxiv},
	author = {Sun, Kunyang and Head-Gordon, Teresa},
	month = apr,
	year = {2026},
}

@misc{kincore2,
	title = {{AlphaFold2} models of the active form of all 437 catalytically competent human protein kinase domains},
	url = {http://biorxiv.org/lookup/doi/10.1101/2023.07.21.550125},
	doi = {10.1101/2023.07.21.550125},
	language = {en},
	author = {Faezov, Bulat and Dunbrack, Roland L.},
	month = jul,
	year = {2023},
}

@article{kincore,
	title = {Kincore: a web resource for structural classification of protein kinases and their inhibitors},
	volume = {50},
	copyright = {https://creativecommons.org/licenses/by/4.0/},
	issn = {0305-1048, 1362-4962},
	shorttitle = {Kincore},
	url = {https://academic.oup.com/nar/article/50/D1/D654/6395339},
	doi = {10.1093/nar/gkab920},
	language = {en},
	number = {D1},
	journal = {Nucleic Acids Research},
	author = {Modi, Vivek and Dunbrack, Roland L},
	month = jan,
	year = {2022},
	pages = {D654--D664},
}
```

# Keep track of all the envs and their setup.
# Refer to the respective GitHub repositories for the latest instructions.

import os as _os
from pathlib import Path

# Each backend can be invoked from either:
#   - a conda env (legacy): set <METHOD>_CONDA_ENV; dispatch uses `conda run -n <env>`
#   - a uv/Python venv: set <METHOD>_VENV to the venv root; dispatch prepends
#     <venv>/bin to PATH. If both are set, venv wins (see base.py).

# Boltz-1/2 Setup
# -------------------------------------------------------------------
# https://github.com/jwohlwend/boltz
# Conda: `conda create -n boltz python=3.10 && conda activate boltz && pip install boltz -U`
# uv:    `uv venv --python 3.10 envs/boltz && uv pip install --python envs/boltz/bin/python boltz -U`
BOLTZ_CONDA_ENV = "boltz"
BOLTZ_VENV = _os.environ.get("BOLTZ_VENV", "")

# Chai-1 Setup
# -------------------------------------------------------------------
# https://github.com/chaidiscovery/chai-lab
# Conda: `conda create -n chai python=3.10 && conda activate chai && pip install chai-lab==0.6.1`
# uv:    `uv venv --python 3.10 envs/chai && uv pip install --python envs/chai/bin/python chai-lab==0.6.1`
CHAI_CONDA_ENV = "chai"
CHAI_VENV = _os.environ.get("CHAI_VENV", "")

# Protenix Setup
# -------------------------------------------------------------------
# https://github.com/bytedance/Protenix
# Conda: `conda create -n protenix python=3.10 && conda activate protenix && pip3 install protenix`
# uv:    `uv venv --python 3.10 envs/protenix && uv pip install --python envs/protenix/bin/python protenix`
PROTENIX_CONDA_ENV = "protenix"
PROTENIX_VENV = _os.environ.get("PROTENIX_VENV", "")

# RoseTTAFold-All-Atom Setup
# -------------------------------------------------------------------
# https://github.com/baker-laboratory/RoseTTAFold-All-Atom
# Conda-only (compiled SE3Transformer + dgl). Build at a prefix, e.g.:
#   mamba env create -f environment.yaml -p /path/to/envs/rfaa
# RFAA needs full env activation (LD_LIBRARY_PATH for dgl/CUDA), so the
# adapter sources+activates RFAA_PREFIX rather than using a PATH prefix.
RFAA_CONDA_ENV = "rfaa"
RFAA_VENV = _os.environ.get("RFAA_VENV", "")
# Conda prefix of the built RFAA env (activated by the adapter):
RFAA_PREFIX = _os.environ.get(
    "RFAA_PREFIX", "/path/to/envs/rfaa"
)
# Path to the cloned RFAA repo (holds rf2aa/, config/, weights, make_msa.sh):
RFAA_REPO = _os.environ.get(
    "RFAA_REPO",
    "/path/to/RoseTTAFold-All-Atom",
)
# Model weights (downloaded into the repo by default):
RFAA_WEIGHTS = _os.environ.get(
    "RFAA_WEIGHTS", _os.path.join(RFAA_REPO, "RFAA_paper_weights.pt")
)
# Database paths for make_msa.sh. Note the upstream wiring quirk:
#   make_msa.sh receives config.sequencedb as $5 and uses it as the *template*
#   (hhsearch) DB; UniRef30/BFD for the MSA come from the DB_UR30/DB_BFD env
#   vars (NOT from config). ModelRunner also reads config.hhdb for template
#   ffindex loading. So sequencedb == hhdb == PDB100 base, and UniRef30/BFD are
#   exported by the adapter from RFAA_UR30 / RFAA_BFD.
# Template DB base, e.g. RFAA/databases/pdb100_2021Mar03/pdb100_2021Mar03
# (REQUIRED — read even at model init). Empty until staged.
RFAA_HHDB = _os.environ.get("RFAA_HHDB", "")
# UniRef30 hhsuite DB base, e.g. RFAA/databases/UniRef30_2020_06/UniRef30_2020_06.
RFAA_UR30 = _os.environ.get("RFAA_UR30", "")
# BFD hhsuite DB base (fallback, rarely needed for well-represented families):
#   RFAA/databases/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt
RFAA_BFD = _os.environ.get("RFAA_BFD", "")
# conda.sh to source for non-interactive activation (NERSC Miniforge default):
RFAA_CONDA_SH = _os.environ.get(
    "RFAA_CONDA_SH",
    "/path/to/conda.sh",
)
# Per-protein MSA cache root. When set, MSAs are pre-built once per unique protein
# at <RFAA_MSA_CACHE>/<protein>/A/t000_.* (via generate_msa_command / make_msa.sh)
# and generate_command symlinks them into each combo's expected dir so make_msa
# skips. Empty -> RFAA builds the MSA inline per combo (the old behavior).
RFAA_MSA_CACHE = _os.environ.get("RFAA_MSA_CACHE", "")
# make_msa.sh resources (hhblits/hhsearch threads + RAM in GB). Read by
# generate_msa_command; the sharded pre-build sets RFAA_MSA_CPUS higher (e.g. 32).
RFAA_MSA_CPUS = _os.environ.get("RFAA_MSA_CPUS", "8")
RFAA_MSA_MEM = _os.environ.get("RFAA_MSA_MEM", "64")

# Kinconfbench (kinase analysis pipeline)
# -------------------------------------------------------------------
# For PDB download/annotation and cofolding analysis (step 3, main_pipeline).
# Set KINCONFBENCH_PREFIX for prefix install, or KINCONFBENCH_ENV for named env.

KINCONFBENCH_PREFIX = "/path/to/envs/kinconfbench"  # change this to your path
KINCONFBENCH_ENV = "kinconfbench"  # change this to your env name

# Kinconfbench data root (for geometric analysis and related scripts)
# Path to kinase_all_data/kinconfbench. Override via KINCONFBENCH_DATA env var.
KINCONFBENCH_DATA = _os.environ.get(
    "KINCONFBENCH_DATA",
    "/path/to/kinconfbench_inference_data/kinconfbench",
)

# Package directory containing this file (cofolding_inference/). step_3 uses FOLDER.parent for kinase_benchmark/...
FOLDER = Path(__file__).resolve().parent
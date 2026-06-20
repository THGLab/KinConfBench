# RoseTTAFold-All-Atom (RFAA) backend setup

RFAA is the 4th cofolding backend. Unlike boltz/chai/protenix it is conda-only
(compiled SE3Transformer + dgl), builds its own MSAs/templates inline, produces
one deterministic PDB per run, and needs large sequence/template databases.

## 1. Clone + conda env

All workspace-level RFAA artifacts live under `RFAA/` (the clone, `databases/`,
`runs/`, `scripts/`). Only the conda env stays outside, at
`/path/to/envs/rfaa` (alongside the other backends' legacy envs).

```bash
cd /path/to/kinconfbench/RFAA
git clone https://github.com/baker-laboratory/RoseTTAFold-All-Atom.git
cd RoseTTAFold-All-Atom
mamba env create -f environment.yaml -p /path/to/envs/rfaa
```

The build-pinned `environment.yaml` solved fine on NERSC Miniforge (2026-05-19).
If it fails on a stale pin (e.g. `dgl ==1.1.2 cuda112py310...`), relax that pin
— see RFAA issues #68/#77/#115.

## 2. SE3Transformer + weights

**Gotcha:** `conda run -p <prefix> <cmd>` does NOT activate PATH on NERSC, so
`python`/`pip` resolve to system Python 2.7. Use the env's binaries explicitly:

```bash
export PATH=/path/to/envs/rfaa/bin:$PATH
cd rf2aa/SE3Transformer
/path/to/envs/rfaa/bin/pip install --no-cache-dir -r requirements.txt
/path/to/envs/rfaa/bin/python setup.py install
cd ../../
wget http://files.ipd.uw.edu/pub/RF-All-Atom/weights/RFAA_paper_weights.pt  # 1.3G
```

### Verify (from the repo root)
```bash
export PATH=/path/to/envs/rfaa/bin:$PATH
python -c "import rf2aa, torch, dgl, se3_transformer; print(torch.__version__, dgl.__version__)"
# -> 2.0.1 1.1.2
python -m rf2aa.run_inference --help
```

## 3. Apply the KinConfBench seed patch

RFAA emits one deterministic structure with no seed knob. To emulate the
diffusion backends' N samples we run N times with different seeds. Apply the
idempotent patch (injects `seed_all(config.seed)` into `run_inference.main`):

```bash
export RFAA_REPO=/path/to/kinconfbench/RFAA/RoseTTAFold-All-Atom
python cofolding_inference/rfaa_pipeline.py patch "$RFAA_REPO"
```

## 3b. cs-blast (PSIPRED dependency)

`make_msa.sh` runs PSIPRED (`input_prep/make_ss.sh`), which needs cs-blast plus
legacy BLAST (`makemat`/`psipass2`, already provided by the `blast-legacy` conda
package). Install cs-blast into the repo:
```bash
bash install_dependencies.sh   # downloads csblast-2.2.3 into the repo
```

## 4. Databases (full-native MSA pipeline)

RFAA builds MSAs+templates inline via `make_msa.sh` (hhblits/hhsearch). None are
pre-staged on NERSC. **Download is ~127G** for the common case:

```bash
mkdir -p RFAA/databases && cd RFAA/databases
# UniRef30 (primary MSA DB) — 46G
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
mkdir -p UniRef30_2020_06 && tar -xzf UniRef30_2020_06_hhsuite.tar.gz -C UniRef30_2020_06
# PDB100 templates (REQUIRED even to init the model) — 81G
wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz && tar -xzf pdb100_2021Mar03.tar.gz
# (the download .tar.gz can be deleted once extraction succeeds)
```

**BFD is fallback-only** — `make_msa.sh` only searches BFD if UniRef30 returns
too few sequences (n75>2000 / n50>4000). Well-represented families (kinases)
essentially always pass on UniRef30, so the 272G BFD is usually unnecessary.
Add it only if a system comes up short:
```bash
cd RFAA/databases
wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz  # 272G
mkdir -p bfd && tar -xzf ...sorted_opt.tar.gz -C bfd
export RFAA_BFD=$PWD/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt
```

### Wiring quirk (important)

`make_msa` calls `./make_msa.sh <fasta> <out> <cpus> <ram> <sequencedb> <hhdb>`,
but `make_msa.sh` reads `$5` (sequencedb) as the **hhsearch template DB** and
ignores `$6`; UniRef30/BFD for the MSA come from the `DB_UR30`/`DB_BFD` **env
vars**. And `ModelRunner.__init__` reads `config.hhdb + "_pdb.ffindex"` at start.
Net effect (handled by the adapter):
- `config.database_params.sequencedb` = `config.database_params.hhdb` = **PDB100 base**
- UniRef30/BFD provided via `DB_UR30`/`DB_BFD`, exported by the adapter from
  `RFAA_UR30`/`RFAA_BFD`.

Set, per shell (the adapter reads these; run from the workspace root):
```bash
export RFAA_HHDB=$PWD/RFAA/databases/pdb100_2021Mar03/pdb100_2021Mar03
export RFAA_UR30=$PWD/RFAA/databases/UniRef30_2020_06/UniRef30_2020_06
# export RFAA_BFD=...   # only if BFD staged
```

## 5. Env vars KinConfBench reads (cofolding_inference/envs.py)

| Var | Default | Purpose |
|---|---|---|
| `RFAA_PREFIX` | `/path/to/envs/rfaa` | conda env to source+activate |
| `RFAA_REPO` | `.../kinconfbench/RFAA/RoseTTAFold-All-Atom` | clone holding rf2aa + configs |
| `RFAA_WEIGHTS` | `<RFAA_REPO>/RFAA_paper_weights.pt` | model checkpoint |
| `RFAA_HHDB` | "" | PDB100 base (sequencedb + hhdb in config) |
| `RFAA_UR30` | "" | UniRef30 base (exported as DB_UR30) |
| `RFAA_BFD` | "" | BFD base (exported as DB_BFD; optional) |
| `RFAA_CONDA_SH` | NERSC Miniforge conda.sh | sourced for activation |

The adapter activates with `source <conda.sh> && conda activate <prefix> &&
export PATH=<prefix>/bin:$PATH` (the explicit PATH prepend is required — see the
gotcha above).

## 6. Run via the pipeline

```bash
python KinConfBench/step_2_run_all_inferences.py config.yml \
    --output-folder out --methods rfaa --num-samples 3 --num-gpus 1
bash out/scripts/run_all_inferences.sh        # runs N seeds per combo
python KinConfBench/cofolding_inference/gather_structural_files.py \
    --source out/outputs --target out/predictions   # PDB -> CIF (rfaa_conf<N>.cif)
python KinConfBench/step_3_kinase_cofolding_analysis.py -d out -v
```

## 6b. Confidence + pLDDT re-ranking

RFAA saves its self-estimated confidence in `<job>_aux.pt` (per-residue pLDDT,
PAE matrix, and the scalars `mean_plddt` / `mean_pae` / `pae_prot` / `pae_inter`;
`pae_inter` = protein-ligand interface error). Reading it needs torch (RFAA env
only), so the inference step (`rfaa.generate_command`) runs
`cofolding_inference/rfaa_pipeline.py confidence` after the seeds to write a plain
`seed<N>/confidence.json` beside each structure.

`gather_structural_files.py` then **ranks conformers by `mean_plddt` (descending)
so `rfaa_conf0` is the best**, and writes `predictions/<combo>/rfaa_ranking.csv`
(conf_idx ↔ seed ↔ plddt/pae). Per-residue pLDDT is also carried into each CIF's
B-factor column. (If `confidence.json` is missing — e.g. an older run — gather
falls back to numeric seed order and warns.)

To (re-)rank an existing run with no GPU / no re-inference, use the driver — it
backfills `confidence.json`, re-gathers ranked CIFs, re-labels, and merges the
confidence into `<combo>_analysis_ranked.csv`:

```bash
bash RFAA/scripts/rerank_pipeline.sh RFAA/runs/<run_dir>
```

> Caveat: across seeds the pLDDT spread is narrow (RFAA is near-deterministic),
> so the ranking is only weakly discriminative — but it reliably floats the
> consensus structure to `conf0` and sinks geometric outliers (e.g. an Inactive
> DFG-HRD blip) toward the bottom.

## Smoke-test findings (ABL1_HUMAN+1N1, N=20, validated 2026-05-20)

- **All 20 succeeded.** Pairwise CA-RMSD 0.18–4.4 Å (mean 0.79 Å): seeds give
  real *structural* spread but converge on one KinCoRe state — DFGin 20/20,
  BLAminus 20/20, Chelix-in 20/20, Active 19/20. So RFAA+seeds adds little
  *conformational-label* diversity; a small N is usually enough.
- **Timing (A100):** MSA+template build ~13–15 min/protein (one-time, hhblits vs
  UniRef30 + hhsearch vs PDB100); inference ~25 s/structure once the MSA is cached.

### Two failures hit + fixed (both in `rfaa_pipeline.py patch` / gather)

1. **PSIPRED buffer-overflow** → empty `ss2` → hhsearch rejected the
   ss2-prepended a3m ("sequence ss_pred contains no residues") → no `t000_.atab`.
   Fixed by the psipred-fallback patch (search the plain a3m when ss2 is empty).
2. **Ligand atom names** — RFAA writes ligand HETATMs with element-only names
   (22 atoms all "C"), which collide as CIF ("Atom C defined twice") and made the
   analysis discard the whole structure. `gather_structural_files._uniquify_atom_names`
   renames colliding atoms (C1,C2,…) only in residues that need it.

## 7. Per-protein MSA pre-build + reuse (the full-run optimization)

`make_msa` caches at `output_path/job_name/chain` and `job_name = <combo>`, so
all seeds of a combo share the MSA but **different ligands of the same protein
would recompute it** — ~1,420 builds instead of 78 for the full run (~18× waste;
the MSA build is the dominant RFAA cost). Since the MSA is a pure function of the
sequence (verified byte-identical across runs), we pre-build it **once per unique
protein** and reuse it.

**Pre-build (CPU-only, sharded across premium jobs):**
```bash
export RFAA_REPO=$PWD/RFAA/RoseTTAFold-All-Atom
export RFAA_HHDB=$PWD/RFAA/databases/pdb100_2021Mar03/pdb100_2021Mar03
export RFAA_UR30=$PWD/RFAA/databases/UniRef30_2020_06/UniRef30_2020_06
export RFAA_MSA_CACHE=$PWD/RFAA/msa_cache
# enumerate 78 unique proteins (from data/kinconfbench_meta_with_sequence_smiles.csv),
# write FASTAs + 5 shard scripts + 5 premium-CPU sbatch + manifest:
python KinConfBench/cofolding_inference/rfaa_pipeline.py build-msas --nshards 5 --cpus 32 --mem 64
sbatch RFAA/scripts/build_msas_shard0.sh   # ... shard4.sh  (CPU premium, ~3-4 h each)
```
Each protein's `make_msa.sh` (the **same workflow** inference runs — hhblits vs
UniRef30 + hhsearch vs PDB100 + psipred) writes
`RFAA/msa_cache/<protein>/A/{t000_.msa0.a3m,t000_.hhr,t000_.atab,t000_.ss2,
t000_.msa0.ss2.a3m}` + `query.fasta`. Shard scripts skip proteins already cached
(re-runnable), and one protein failing doesn't abort the shard.

**Reuse at inference:** with `RFAA_MSA_CACHE` set, `generate_command` prepends a
per-protein-chain snippet that — when the cached `query.fasta` sequence matches the
combo's chain FASTA — symlinks `<cache>/<protein>/A/t000_.*` into
`<combo_dir>/<combo>/<chain>/`, so `make_msa` finds the files and **skips its
search**. On any mismatch / missing cache it falls back to a normal inline build
(`DB_UR30` is still exported), so it is always safe. `RFAA_MSA_CPUS`/`RFAA_MSA_MEM`
(default 8/64) tune the build threads/RAM.

## Notes / caveats

- **Determinism**: RFAA is a regression model; different seeds perturb template
  initialization but structures may stay very similar → low "diversity" and a
  flat "success vs k" curve. The N CIFs still exercise the analysis path.
- **Output**: RFAA writes `<output_path>/<job_name>.pdb`; gather converts to CIF
  with gemmi (installed in the kinconfbench env).
- **SignalP6** (`signalp6=6.0g` in environment.yaml) is license-gated but the
  conda package installs without registered weights; not needed for kinase
  catalytic domains.

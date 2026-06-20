"""
RoseTTAFold-All-Atom (RFAA) co-folding method implementation.

Unlike the diffusion backends (boltz/chai/protenix), RFAA:
  - is conda-only with a compiled SE3Transformer + dgl, so we activate the
    conda prefix (sets LD_LIBRARY_PATH) instead of using a bare PATH prefix;
  - builds its own MSA + templates inline during inference (make_msa.sh against
    UniRef30/BFD + PDB100), so msa_paths is ignored here;
  - produces ONE deterministic PDB per run. To emulate the N-sample diversity of
    the other backends we run N times with different seeds (requires the seed
    patch applied to the RFAA clone via `rfaa_pipeline.py patch`).

Input is a Hydra config (rf2aa/config/inference/) plus a per-chain FASTA; the
ligand is passed as a SMILES string. Output is <output_path>/<job_name>.pdb,
converted to CIF downstream by gather_structural_files.py.
"""

import os
import string
from typing import List, Optional

from .base import CoFoldingMethod
from .envs import (
    RFAA_CONDA_ENV,
    RFAA_VENV,
    RFAA_PREFIX,
    RFAA_REPO,
    RFAA_WEIGHTS,
    RFAA_HHDB,
    RFAA_UR30,
    RFAA_BFD,
    RFAA_CONDA_SH,
    RFAA_MSA_CACHE,
    RFAA_MSA_CPUS,
    RFAA_MSA_MEM,
)


class RoseTTAFoldAA(CoFoldingMethod):
    """RoseTTAFold-All-Atom co-folding method implementation."""

    def __init__(self):
        if RFAA_VENV and os.path.isdir(RFAA_VENV):
            super().__init__(name="rfaa", venv_path=RFAA_VENV)
        else:
            super().__init__(name="rfaa", conda_env=RFAA_CONDA_ENV)
        self.conda_prefix = RFAA_PREFIX
        self.repo = RFAA_REPO
        # Populated by prepare_input; (chain, protein_name, abs_fasta_path) tuples.
        self._protein_chains = []

    def get_conda_command(self) -> str:
        """RFAA needs a full conda activation (LD_LIBRARY_PATH for dgl/CUDA).

        On NERSC `conda activate <prefix>` sets CONDA_PREFIX/LD_LIBRARY_PATH but
        does NOT prepend the env bin to PATH (so `python` stays system py2.7) —
        we prepend it explicitly after activating.
        """
        if self.venv_path:
            return f"PATH={self.venv_path}/bin:$PATH"
        return (
            f"source {RFAA_CONDA_SH} && conda activate {self.conda_prefix} "
            f"&& export PATH={self.conda_prefix}/bin:$PATH &&"
        )

    def _config_dir(self) -> str:
        """RFAA's Hydra primary search path (where base.yaml lives)."""
        return os.path.join(self.repo, "rf2aa", "config", "inference")

    def prepare_input(self, protein_names: List[str], protein_sequences: List[str],
                      ligand_names: List[str], ligand_smiles: List[str],
                      combination_name: str, input_dir: str,
                      msa_paths: Optional[List[str]] = None) -> str:
        """Write per-chain FASTA(s) + a Hydra config for one combination.

        msa_paths is ignored: RFAA builds its own MSA inline.
        Returns the path to the generated Hydra config (named kcb_<combo>.yaml),
        written into the RFAA repo's config/inference dir so `defaults: [base]`
        resolves natively.
        """
        rfaa_input_dir = os.path.join(input_dir, "rfaa", combination_name)
        os.makedirs(rfaa_input_dir, exist_ok=True)

        # Assign chain letters: proteins first (A, B, ...), then ligands.
        chain_letters = list(string.ascii_uppercase)
        protein_chains = {}
        # (chain, protein_name, abs_fasta_path) — used by generate_command to stage
        # a pre-built per-protein MSA (RFAA_MSA_CACHE) into this combo's cache dir.
        self._protein_chains = []
        for i, (pname, seq) in enumerate(zip(protein_names, protein_sequences)):
            chain = chain_letters[i]
            fasta_path = os.path.join(rfaa_input_dir, f"{chain}.fasta")
            with open(fasta_path, "w") as f:
                f.write(f">{chain}\n{seq}\n")
            protein_chains[chain] = os.path.abspath(fasta_path)
            self._protein_chains.append((chain, pname, os.path.abspath(fasta_path)))

        ligand_chains = {}
        offset = len(protein_names)
        for j, smiles in enumerate(ligand_smiles):
            chain = chain_letters[offset + j]
            ligand_chains[chain] = smiles

        # Build the Hydra config text (defaults: base, then our overrides).
        lines = ["defaults:", "  - base", ""]
        lines.append(f'job_name: "{combination_name}"')
        lines.append("seed: 0")
        lines.append(f"checkpoint_path: {os.path.abspath(RFAA_WEIGHTS)}")
        lines.append("database_params:")
        # sequencedb is passed to make_msa.sh as $5 -> used as the hhsearch
        # template DB; hhdb is read at model init for template ffindex. Both
        # must be the PDB100 base. UniRef30/BFD are exported separately.
        lines.append(f'  sequencedb: "{RFAA_HHDB}"')
        lines.append(f'  hhdb: "{RFAA_HHDB}"')
        lines.append("  command: make_msa.sh")
        lines.append("  num_cpus: 8")
        lines.append("  mem: 64")
        lines.append("protein_inputs:")
        for chain, fasta in protein_chains.items():
            lines.append(f"  {chain}:")
            lines.append(f"    fasta_file: {fasta}")
        if ligand_chains:
            lines.append("sm_inputs:")
            for chain, smiles in ligand_chains.items():
                lines.append(f"  {chain}:")
                lines.append(f'    input: "{smiles}"')
                lines.append('    input_type: "smiles"')
        config_text = "\n".join(lines) + "\n"

        # Primary copy: into the RFAA repo's config dir so Hydra finds base.
        config_name = f"kcb_{combination_name}"
        repo_config = os.path.join(self._config_dir(), f"{config_name}.yaml")
        os.makedirs(self._config_dir(), exist_ok=True)
        with open(repo_config, "w") as f:
            f.write(config_text)
        # Provenance copy alongside the FASTAs.
        with open(os.path.join(rfaa_input_dir, f"{config_name}.yaml"), "w") as f:
            f.write(config_text)

        return repo_config

    def generate_command(self, input_path: str, output_dir: str,
                         combination_name: str, num_samples: int = 1) -> str:
        """Run RFAA num_samples times with distinct seeds (one PDB each).

        input_path is the Hydra config produced by prepare_input; we pass its
        basename via --config-name.

        All seeds share one output_path (the combo dir) so the expensive MSA +
        template search runs ONCE: make_msa caches at <output_path>/<job>/<chain>
        and skips if those files exist. RFAA writes <output_path>/<job>.pdb each
        run, so after each seed we move it into seed<N>/ to avoid clobbering.
        """
        config_name = os.path.splitext(os.path.basename(input_path))[0]
        prefix = self.get_conda_command()
        combo_dir = os.path.abspath(
            os.path.join(output_dir, self.name, combination_name)
        )
        pdb = f"{combo_dir}/{combination_name}.pdb"
        aux = f"{combo_dir}/{combination_name}_aux.pt"

        sub_cmds = [f"cd {self.repo}", f"mkdir -p {combo_dir}"]
        # UniRef30/BFD are read by make_msa.sh from these env vars (not config).
        if RFAA_UR30:
            sub_cmds.append(f"export DB_UR30={RFAA_UR30}")
        if RFAA_BFD:
            sub_cmds.append(f"export DB_BFD={RFAA_BFD}")
        # Reuse pre-built per-protein MSAs: symlink RFAA_MSA_CACHE/<protein>/A/t000_.*
        # into this combo's expected dir (<combo_dir>/<job>/<chain>/) so make_msa
        # skips its search. Sequence-guarded against the cache's query.fasta so a
        # stale/wrong cache never feeds a bad MSA — on any mismatch make_msa rebuilds.
        if RFAA_MSA_CACHE and self._protein_chains:
            for chain, pname, fasta_abspath in self._protein_chains:
                cache = f"{RFAA_MSA_CACHE}/{pname}/A"
                dest = f"{combo_dir}/{combination_name}/{chain}"
                sub_cmds.append(
                    f'if [ -f {cache}/t000_.msa0.a3m ] && [ -f {cache}/t000_.hhr ] '
                    f'&& [ -f {cache}/t000_.atab ] '
                    f'&& diff -q <(grep -v "^>" {fasta_abspath}) '
                    f'<(grep -v "^>" {cache}/query.fasta) >/dev/null 2>&1; '
                    f'then mkdir -p {dest} && ln -sf {cache}/t000_.* {dest}/ '
                    f'&& echo "[kcb] reused MSA for {pname} -> {dest}"; '
                    f'else echo "[kcb] no cached MSA for {pname}; make_msa will build it"; fi'
                )
        for seed in range(num_samples):
            seed_out = f"{combo_dir}/seed{seed}"
            seed_pdb = f"{seed_out}/{combination_name}.pdb"
            # Run this seed, then move its PDB (and aux.pt) into seed<N>/. The MSA
            # cache at combo_dir/<job>/ stays put for the next seed.
            run = (
                f"mkdir -p {seed_out} && "
                f"python -m rf2aa.run_inference --config-name {config_name} "
                f"seed={seed} output_path={combo_dir} && "
                f"mv {pdb} {seed_pdb}"
            )
            aux_mv = f"[ -f {aux} ] && mv {aux} {seed_out}/ || true"
            # Seed-level resume + failure tolerance: skip if this seed's PDB already
            # exists (so a resubmit after a walltime kill picks up where it left off),
            # and a single seed failing logs but doesn't abort the combo's other seeds.
            sub_cmds.append(
                f"if [ -f {seed_pdb} ]; then echo '[kcb] seed{seed} cached'; "
                f"else {{ {run}; }} && {{ {aux_mv}; }} "
                f"|| echo '[kcb] seed{seed} FAILED {combination_name}'; fi"
            )
        # Extract per-structure confidence (mean_plddt, pae_*) from each seed's
        # *_aux.pt into confidence.json so the downstream (torch-less) gather step
        # can rank conformers by pLDDT. Runs in the active RFAA env (has torch).
        pipeline = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "rfaa_pipeline.py")
        sub_cmds.append(f"python {pipeline} confidence {combo_dir}")
        # prefix ends with "&&" (conda activate) or is a PATH= assignment.
        body = " && ".join(sub_cmds)
        return f"{prefix} {body}"

    def generate_msa_command(self, input_path, output_dir, protein_name=""):
        """Build one protein's MSA + templates with RFAA's own make_msa.sh.

        This is the SAME workflow inference runs (hhblits vs UniRef30 + hhsearch vs
        PDB100 + psipred), just run as a standalone per-protein pre-pass. Writes the
        cache at <output_dir>/<protein_name>/A/t000_.* (the layout inference's
        make_msa expects) plus a query.fasta used by generate_command's reuse guard.

        input_path  : a single-sequence protein FASTA
        output_dir  : MSA cache root (e.g. RFAA_MSA_CACHE)
        protein_name: cache key (one dir per unique protein)

        make_msa.sh args ($1 fasta, $2 out_dir, $3 CPU, $4 MEM, $5 PDB100 template
        DB); UniRef30/BFD come from DB_UR30/DB_BFD env vars. CPU/MEM from
        RFAA_MSA_CPUS / RFAA_MSA_MEM (the sharded pre-build sets CPUs higher).
        """
        prefix = self.get_conda_command()
        out_dir = os.path.abspath(os.path.join(output_dir, protein_name, "A"))
        fasta = os.path.abspath(input_path)
        sub_cmds = [f"cd {self.repo}"]
        if RFAA_UR30:
            sub_cmds.append(f"export DB_UR30={RFAA_UR30}")
        if RFAA_BFD:
            sub_cmds.append(f"export DB_BFD={RFAA_BFD}")
        sub_cmds.append(f"mkdir -p {out_dir}")
        sub_cmds.append(f"cp {fasta} {out_dir}/query.fasta")
        sub_cmds.append(
            f"./make_msa.sh {fasta} {out_dir} {RFAA_MSA_CPUS} {RFAA_MSA_MEM} {RFAA_HHDB}"
        )
        return f"{prefix} " + " && ".join(sub_cmds)

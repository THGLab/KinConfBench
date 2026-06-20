#!/usr/bin/env python3
"""One driver for the whole RFAA (RoseTTAFold-All-Atom) backend workflow.

RFAA is integrated as a 4th cofolding method via the ``rfaa.py`` adapter (a peer
of boltz/chai/protenix). Everything *around* that adapter — patching the upstream
clone, pre-building MSAs, running the full benchmark, staging + merging outputs,
and extracting confidence — lives here as subcommands of a single CLI instead of
six separate scripts.

Subcommands (run ``rfaa_pipeline.py <cmd> -h`` for each):

  patch        Idempotently patch a RFAA clone (seed RNG from config.seed for
               per-seed diversity; psipred-empty-ss2 fallback in make_msa.sh).
  build-msas   Pre-build MSAs once per unique protein (78), sharded for premium
               CPU jobs — the inference adapter then reuses the cache verbatim.
  run          Resumable, shardable, failure-tolerant full inference driver
               (reuses the adapter's prepare_input + generate_command).
  stage        Reorganise the flat ``<run>/predictions/<combo>/`` tree into the
               gene-nested ``<staged>/<GENE>/predictions/<combo>/`` layout.
  merge        Append-only merge of staged RFAA predictions + per-conformer CSV
               rows into the canonical kinconfbench_inference_data tree.
  confidence   Extract per-structure confidence (mean_plddt/PAE) from RFAA's
               ``*_aux.pt`` into ``confidence.json`` sidecars (torch-only).

Env isolation: ``confidence`` is invoked by ``rfaa.py`` as a subprocess *inside
the RFAA conda env* (torch-only; no pandas / no kinconfbench package). So every
non-stdlib import here is done LAZILY inside its subcommand handler — the module
top level imports stdlib only, and ``rfaa_pipeline.py confidence`` triggers only
``import torch``.
"""
import argparse
import csv
import glob
import io
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# This file lives at <workspace>/KinConfBench/cofolding_inference/rfaa_pipeline.py, and the
# workspace root holds both KinConfBench/ and RFAA/. Derive WS from __file__ so no absolute
# path is baked in; META_CSV is repo-relative. KBIN (the kinconfbench conda-env bin) is
# external to the repo, so it falls back to a placeholder and is overridable via env.
WS = str(Path(__file__).resolve().parents[2])
KBIN = os.environ.get("KINCONFBENCH_BIN", "/path/to/envs/kinconfbench/bin")
META_CSV = str(Path(__file__).resolve().parents[1] / "data" / "kinconfbench_meta_with_sequence_smiles.csv")


# ======================================================================== patch
# Idempotently patch a RoseTTAFold-All-Atom clone so KinConfBench can drive it.
#
# Patch 1 (seed): upstream run_inference.py produces one deterministic structure
# and never seeds the RNGs from config. We inject a seed read (config.seed) + RNG
# seeding at the top of main(), so running N times with different seeds yields N
# conformations (the adapter uses this to emulate the diffusion backends).
#
# Patch 2 (psipred fallback): PSIPRED (psipred 4.01) buffer-overflows on modern
# glibc and emits an empty ss2, which makes hhsearch reject the ss2-prepended a3m
# ("sequence ss_pred contains no residues") -> no t000_.atab -> inference crashes.
# We make make_msa.sh use the ss2 only when it has real residues, else search the
# plain a3m (templates are still found; ss2 is only a minor ranking enhancement).
SEED_MARKER = "# >>> kinconfbench seed patch >>>"
SEED_ORIG = "def main(config):\n    runner = ModelRunner(config)\n    runner.infer()"
SEED_INJECTED = '''def main(config):
    # >>> kinconfbench seed patch >>>
    import random as _random
    import numpy as _np
    _seed = int(config.get("seed", 0))
    _random.seed(_seed)
    _np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)
    # <<< kinconfbench seed patch <<<
    runner = ModelRunner(config)
    runner.infer()'''

MSA_MARKER = "# >>> kinconfbench psipred-fallback patch >>>"
MSA_ORIG = (
    "    cat $out_dir/t000_.ss2 $out_dir/t000_.msa0.a3m > $out_dir/t000_.msa0.ss2.a3m\n"
    "    $HH -i $out_dir/t000_.msa0.ss2.a3m -o $out_dir/t000_.hhr -atab $out_dir/t000_.atab -v 0"
)
MSA_INJECTED = '''    # >>> kinconfbench psipred-fallback patch >>>
    if [ -s $out_dir/t000_.ss2 ] && [ $(grep -v "^>" $out_dir/t000_.ss2 | tr -d "[:space:]" | wc -c) -gt 0 ]; then
        cat $out_dir/t000_.ss2 $out_dir/t000_.msa0.a3m > $out_dir/t000_.msa0.ss2.a3m
        HH_IN=$out_dir/t000_.msa0.ss2.a3m
    else
        HH_IN=$out_dir/t000_.msa0.a3m
    fi
    $HH -i $HH_IN -o $out_dir/t000_.hhr -atab $out_dir/t000_.atab -v 0
    # <<< kinconfbench psipred-fallback patch <<<'''


def _patch(target: Path, marker, orig, injected, name):
    if not target.is_file():
        sys.exit(f"{target} not found")
    src = target.read_text()
    if marker in src:
        print(f"[{name}] already patched: {target}")
        return
    if orig not in src:
        sys.exit(
            f"[{name}] expected block not found in {target}; "
            "upstream may have changed — patch manually."
        )
    target.write_text(src.replace(orig, injected))
    print(f"[{name}] patched: {target}")


def cmd_patch(args):
    repo = Path(args.repo if args.repo else os.environ.get("RFAA_REPO", ""))
    if not str(repo) or not repo.is_dir():
        sys.exit(f"RFAA repo not found: {str(repo)!r} (pass path or set RFAA_REPO)")
    _patch(repo / "rf2aa" / "run_inference.py", SEED_MARKER, SEED_ORIG,
           SEED_INJECTED, "seed")
    _patch(repo / "make_msa.sh", MSA_MARKER, MSA_ORIG, MSA_INJECTED,
           "psipred-fallback")


# =================================================================== confidence
# RFAA's run_inference saves a <job>_aux.pt next to each predicted PDB holding the
# model's self-estimated confidence (per-residue pLDDT, PAE matrix, and the scalar
# summaries mean_plddt / mean_pae / pae_prot / pae_inter). Reading it needs torch,
# which lives only in the RFAA conda env — so this runs there and writes a plain
# confidence.json beside each structure. Downstream gather_structural_files.py
# (kinconfbench env, no torch) then ranks conformers by mean_plddt.
CONF_SCALARS = ("mean_plddt", "mean_pae", "pae_prot", "pae_inter")


def _seed_num(seed_dir: str):
    m = re.search(r"seed(\d+)", os.path.basename(seed_dir))
    return int(m.group(1)) if m else None


def extract_dir(root: str) -> int:
    """Write confidence.json beside every ``*_aux.pt`` found under ``root``."""
    import json
    import torch  # lazy: only the RFAA env has torch; keep module import-light.

    aux_files = sorted(glob.glob(os.path.join(root, "**", "*_aux.pt"), recursive=True))
    n = 0
    for aux in aux_files:
        seed_dir = os.path.dirname(aux)
        a = torch.load(aux, map_location="cpu")
        rec = {"seed": _seed_num(seed_dir)}
        for k in CONF_SCALARS:
            v = a.get(k, None)
            rec[k] = float(v) if v is not None else None
        # Fall back to averaging the per-residue tensor if the scalar is absent.
        if rec.get("mean_plddt") is None and a.get("plddts") is not None:
            rec["mean_plddt"] = float(a["plddts"].mean())
        with open(os.path.join(seed_dir, "confidence.json"), "w") as f:
            json.dump(rec, f, indent=2)
        n += 1
    return n


def cmd_confidence(args):
    n = extract_dir(args.dir)
    print(f"rfaa_confidence: wrote confidence.json for {n} structures under {args.dir}")


# ======================================================================== stage
# Stage the flat RFAA predictions/<combo>/ tree into the gene-nested layout used by
# kinconfbench_inference_data/kinconfbench/<GENE>/predictions/<combo>/. CIFs are
# hard-linked (zero extra disk, survives source purge); the small CSVs are copied.
# The combo's gene comes from the meta CSV (so multi-ligand names like
# ABL1_HUMAN_STI_1N1 map to the right gene without fragile string splitting).
def name_to_gene(meta_csv):
    """combo name (== '{gene}_{ligand}') -> gene_name, from the meta CSV."""
    m = {}
    with open(meta_csv) as f:
        for row in csv.DictReader(f):
            g = row["gene_name"].strip()
            lig = row["ligand"].strip()
            m[f"{g}_{lig}"] = g
    return m


def stage(run, staged, meta_csv, dry_run=False):
    flat = Path(run) / "predictions"
    if not flat.is_dir():
        sys.exit(f"no predictions dir at {flat} (run postprocess_full.sh first)")
    staged = Path(staged)
    n2g = name_to_gene(meta_csv)

    genes, combos_done, combos_empty, cifs, missing_gene, missing_ana = set(), 0, 0, 0, [], []
    for combo_dir in sorted(p for p in flat.iterdir() if p.is_dir()):
        combo = combo_dir.name
        cif_files = sorted(combo_dir.glob("rfaa_conf*.cif"))
        if not cif_files:
            combos_empty += 1                      # the 5 RFAA-incompatible combos
            continue
        gene = n2g.get(combo)
        if gene is None:
            missing_gene.append(combo)
            continue
        dest = staged / gene / "predictions" / combo
        genes.add(gene)
        if dry_run:
            combos_done += 1
            cifs += len(cif_files)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        # CIFs: hard-link (zero extra disk; survives source purge).
        for src in cif_files:
            dst = dest / src.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.link(src, dst)
            cifs += 1
        # Small CSVs: copy (ranking + per-combo analysis, like the abl1 reference).
        for fn in (f"{combo}_analysis.csv", f"{combo}_analysis_ranked.csv", "rfaa_ranking.csv"):
            src = combo_dir / fn
            if src.is_file():
                shutil.copy2(src, dest / fn)
            elif fn != f"{combo}_analysis_ranked.csv":
                missing_ana.append(f"{combo}/{fn}")
        combos_done += 1

    print(f"{'[dry-run] ' if dry_run else ''}staged {combos_done} combos "
          f"across {len(genes)} genes ({cifs} CIFs) -> {staged}")
    print(f"  empty combos skipped (no CIFs): {combos_empty}")
    if missing_gene:
        print(f"  WARNING: {len(missing_gene)} combos had no gene in meta CSV: {missing_gene[:10]}")
    if missing_ana:
        print(f"  NOTE: {len(missing_ana)} combos missing a CSV (label step may have failed): {missing_ana[:10]}")
    return combos_done


def cmd_stage(args):
    staged = args.staged or os.path.join(args.run, "kinconfbench_staged")
    stage(args.run, staged, args.meta_csv, dry_run=args.dry_run)


# =================================================================== build-msas
# Pre-build RFAA MSAs once per unique protein, sharded for N premium-CPU jobs.
# The "method" is RFAA's own make_msa.sh via RoseTTAFoldAA.generate_msa_command —
# the SAME workflow inference runs, so the cached MSAs are reused verbatim.
SBATCH_MSA_TMPL = """#!/bin/bash
#SBATCH -A {account}
#SBATCH -J rfaa_msa_shard{k}
#SBATCH --constraint cpu
#SBATCH --qos {qos}
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task {cpus}
#SBATCH --time {time}
#SBATCH -o {ws}/slurm_logs/rfaa_msa_shard{k}_%j.out
#SBATCH -e {ws}/slurm_logs/rfaa_msa_shard{k}_%j.err
# RFAA MSA build shard {k}/{nshards} (CPU-only: hhblits/hhsearch/psipred).
set -o pipefail
cd {ws}
echo "shard {k} start $(date -Is) on $(hostname)  cpus={cpus}"
bash {shard_sh}
echo "shard {k} done $(date -Is)"
"""


def cmd_build_msas(args):
    # generate_msa_command reads RFAA_MSA_CPUS/MEM at import time -> set BEFORE import.
    os.environ["RFAA_MSA_CPUS"] = str(args.cpus)
    os.environ["RFAA_MSA_MEM"] = str(args.mem)
    from cofolding_inference.rfaa import RoseTTAFoldAA
    from cofolding_inference.envs import RFAA_REPO, RFAA_HHDB, RFAA_UR30

    missing = [n for n, v in (("RFAA_REPO", RFAA_REPO), ("RFAA_HHDB", RFAA_HHDB),
                              ("RFAA_UR30", RFAA_UR30)) if not v]
    if missing:
        sys.exit(f"Export these first (repo + DB paths): {', '.join(missing)}")

    # Unique (protein, sequence), order-preserving.
    seen = {}
    with open(args.meta_csv) as f:
        for row in csv.DictReader(f):
            g, s = row["gene_name"].strip(), row["sequence"].strip()
            seen.setdefault(g, s)
    proteins = list(seen.items())
    print(f"{len(proteins)} unique proteins from {args.meta_csv}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = os.path.abspath(args.cache)
    m = RoseTTAFoldAA()

    # One FASTA + one make_msa command per protein.
    cmds = []
    for g, s in proteins:
        fa = out / f"{g}.fasta"
        fa.write_text(f">{g}\n{s}\n")
        cmds.append((g, m.generate_msa_command(str(fa), cache, g)))

    # Round-robin shard assignment (balances any per-protein time skew).
    shards = [[] for _ in range(args.nshards)]
    for i, item in enumerate(cmds):
        shards[i % args.nshards].append(item)

    scripts_dir = Path(WS) / "RFAA" / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="") as mf:
        w = csv.writer(mf)
        w.writerow(["protein", "shard", "cache_dir"])
        for k, sh in enumerate(shards):
            shard_sh = out / f"shard{k}.sh"
            with open(shard_sh, "w") as f:
                f.write("#!/bin/bash\nset -o pipefail\n")
                f.write(f"# RFAA MSA build shard {k}/{args.nshards} - {len(sh)} proteins\n")
                f.write("# No 'set -e': one protein failing must not abort the shard.\n")
                for g, c in sh:
                    done = f"{cache}/{g}/A/t000_.atab"
                    f.write(f'echo "=== [shard {k}] {g}  $(date -Is) ==="\n')
                    f.write(f'if [ -f {done} ]; then echo "  already cached, skip";\n')
                    f.write(f"else\n  {c} ;\nfi\n")
                    w.writerow([g, k, f"{cache}/{g}/A"])
            shard_sh.chmod(0o755)

            sbatch = scripts_dir / f"build_msas_shard{k}.sh"
            sbatch.write_text(SBATCH_MSA_TMPL.format(
                account=args.account, qos=args.qos, cpus=args.cpus, time=args.time,
                ws=WS, k=k, nshards=args.nshards, shard_sh=shard_sh))
            sbatch.chmod(0o755)

    print(f"wrote {len(proteins)} FASTAs + {args.nshards} shard scripts + sbatch + manifest under {out}")
    print("shard sizes:", [len(s) for s in shards])
    print("submit:", "  ".join(f"sbatch RFAA/scripts/build_msas_shard{k}.sh"
                                for k in range(args.nshards)))


# ========================================================================== run
# Resumable, shardable, failure-tolerant RFAA inference driver for the full run.
# Reuses the RoseTTAFoldAA adapter (prepare_input + generate_command) unchanged,
# but owns the orchestration step_2 lacks at scale: multi-node sharding, per-combo
# + per-seed resume, and per-combo failure tolerance.
def load_combos(meta_csv):
    """One enriched combo per CSV row (matches step_2's schema + naming)."""
    combos = []
    with open(meta_csv) as f:
        for row in csv.DictReader(f):
            g = row["gene_name"].strip()
            lig_field = row["ligand"].strip()
            smiles_field = row["SMILES"].strip()
            if lig_field == "apo" or not smiles_field:
                # Apo: protein-only fold, no sm_inputs. Name stays {gene}_apo.
                ligands, smiles = [], []
            else:
                ligands = lig_field.split("_")
                smiles = smiles_field.split(";")
                if len(ligands) != len(smiles):
                    print(f"WARNING: skipping {g}_{lig_field}: "
                          f"{len(ligands)} ligands vs {len(smiles)} SMILES")
                    continue
            combos.append({
                "proteins": [g],
                "protein_sequences": [row["sequence"].strip()],
                "ligands": ligands,
                "ligand_smiles": smiles,
                "name": f"{g}_{lig_field}",
            })
    return combos


SBATCH_RUN = """#!/bin/bash
#SBATCH -A {account}
#SBATCH -J {job}
#SBATCH --constraint gpu
#SBATCH --gpus {gpus}
#SBATCH --qos {qos}
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --time {time}
#SBATCH -o {ws}/slurm_logs/{job}_%j.out
#SBATCH -e {ws}/slurm_logs/{job}_%j.err
set -o pipefail
cd {ws}
export PATH={kbin}:$PATH
export KINCONFBENCH_DATA={ws}/kinconfbench_inference_data/kinconfbench
export RFAA_REPO={ws}/RFAA/RoseTTAFold-All-Atom
export RFAA_HHDB={ws}/RFAA/databases/pdb100_2021Mar03/pdb100_2021Mar03
export RFAA_UR30={ws}/RFAA/databases/UniRef30_2020_06/UniRef30_2020_06
export RFAA_MSA_CACHE={ws}/RFAA/msa_cache
srun python KinConfBench/cofolding_inference/rfaa_pipeline.py run {driver_args}
"""


def emit_sbatch(args):
    sd = Path(WS) / "RFAA" / "scripts"
    sd.mkdir(parents=True, exist_ok=True)
    out = args.output_folder
    common = f"--gpus {args.gpus} --output-folder {out} --num-samples {args.num_samples}"
    (sd / "run_full_smoke.sbatch").write_text(SBATCH_RUN.format(
        account=args.account, job="rfaa_full_smoke", gpus=args.gpus, qos="premium",
        time="02:00:00", ws=WS, kbin=KBIN,
        driver_args=f"--head {args.smoke} {common}"))
    for k in range(args.nshards):
        (sd / f"run_full_shard{k}.sbatch").write_text(SBATCH_RUN.format(
            account=args.account, job=f"rfaa_full_shard{k}", gpus=args.gpus, qos="regular",
            time="10:00:00", ws=WS, kbin=KBIN,
            driver_args=f"--shard {k} --nshards {args.nshards} {common}"))
    print(f"wrote RFAA/scripts/run_full_smoke.sbatch + run_full_shard0..{args.nshards - 1}.sbatch")
    print("smoke:  sbatch RFAA/scripts/run_full_smoke.sbatch")
    print(f"full :  for k in $(seq 0 {args.nshards - 1}); do sbatch RFAA/scripts/run_full_shard$k.sbatch; done")


def cmd_run(args):
    if args.emit_sbatch:
        emit_sbatch(args)
        return

    from cofolding_inference.rfaa import RoseTTAFoldAA
    from cofolding_inference.envs import RFAA_MSA_CACHE

    combos = load_combos(args.meta_csv)
    if args.combos_file:
        want = {ln.strip() for ln in open(args.combos_file) if ln.strip()}
        work, tag = [c for c in combos if c["name"] in want], "rerun"
        print(f"combos-file: {len(work)}/{len(want)} requested combos matched")
    elif args.head is not None:
        work, tag = combos[:args.head], f"head{args.head}"
    elif args.shard is not None:
        work, tag = combos[args.shard::args.nshards], f"shard{args.shard}of{args.nshards}"
    else:
        work, tag = combos, "all"

    out = os.path.abspath(args.output_folder)
    inp = os.path.join(out, "inputs")
    scripts = os.path.join(out, "scripts")
    logs = os.path.join(out, "logs")
    for d in (inp, scripts, logs):
        os.makedirs(d, exist_ok=True)
    failures = os.path.join(out, f"failures_{tag}.log")

    print(f"[{tag}] {len(work)}/{len(combos)} combos, N={args.num_samples}, {args.gpus} GPUs")
    if not RFAA_MSA_CACHE:
        print("WARNING: RFAA_MSA_CACHE unset -> make_msa will rebuild MSAs inline (slow).")

    # Adapter writes structures under <output_dir>/rfaa/<combo>/...; use the
    # conventional <run>/outputs so gather (--source <run>/outputs) and
    # postprocess_full.sh (which read <run>/outputs/rfaa) line up.
    out_base = os.path.join(out, "outputs")
    m = RoseTTAFoldAA()
    per_gpu = [[] for _ in range(args.gpus)]
    for i, combo in enumerate(work):
        name = combo["name"]
        cfg = m.prepare_input(combo["proteins"], combo["protein_sequences"],
                              combo["ligands"], combo["ligand_smiles"], name, inp)
        cmd = m.generate_command(cfg, out_base, name, num_samples=args.num_samples)
        combo_dir = os.path.join(out_base, "rfaa", name)
        # combo-level resume (skip if already has N PDBs) + failure tolerance.
        guarded = (
            f'echo "=== {name} $(date -Is) ===" ; '
            f'n=$(ls {combo_dir}/seed*/{name}.pdb 2>/dev/null | wc -l) ; '
            f'if [ "$n" -ge {args.num_samples} ]; then echo "  {name}: done ($n), skip" ; '
            f'else {{ {cmd} ; }} || echo "FAIL {name} $(date -Is)" >> {failures} ; fi'
        )
        per_gpu[i % args.gpus].append(guarded)

    # Write one script per GPU and launch them in parallel (pinned via CUDA_VISIBLE_DEVICES).
    procs = []
    for g in range(args.gpus):
        sp = os.path.join(scripts, f"run_{tag}_gpu{g}.sh")
        with open(sp, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"export CUDA_VISIBLE_DEVICES={g}\n")
            f.write(f'echo "[gpu{g}] {len(per_gpu[g])} combos on $(hostname) $(date -Is)"\n')
            for c in per_gpu[g]:
                f.write(c + "\n")
            f.write(f'echo "[gpu{g}] finished $(date -Is)"\n')
        os.chmod(sp, 0o755)
        if args.dry_run:
            print(f"[gpu{g}] {len(per_gpu[g])} combos -> {sp} (dry-run, not launched)")
            continue
        lg = open(os.path.join(logs, f"{tag}_gpu{g}.log"), "w")
        procs.append((g, subprocess.Popen(["bash", sp], stdout=lg, stderr=subprocess.STDOUT), lg))
        print(f"[gpu{g}] launched {len(per_gpu[g])} combos -> {sp}")

    for g, p, lg in procs:
        rc = p.wait()
        lg.close()
        print(f"[gpu{g}] exit {rc}")

    # Summary.
    done = sum(1 for c in work
               if len(list(Path(os.path.join(out_base, "rfaa", c["name"])).glob(f"seed*/{c['name']}.pdb")))
               >= args.num_samples)
    nfail = sum(1 for _ in open(failures)) if os.path.exists(failures) else 0
    print(f"[{tag}] done {done}/{len(work)} combos; failure-log lines: {nfail} ({failures})")


# ======================================================================== merge
# Append-only merge of staged RFAA predictions into the canonical kinconfbench
# tree: existing boltz/chai/protenix/experimental rows + CIFs are never touched —
# only 20 rfaa_conf* rows/files are added per combo. Destination CSVs are CRLF;
# RFAA sources are LF, so appended rows are written CRLF for internal consistency.
# Idempotent: re-running skips any stem whose dest CSV already carries rfaa rows.
CRLF = "\r\n"
RFAA_PREFIX = "rfaa_"


def read_lines_keep_split(path: Path):
    """Return (header_line_no_eol, data_lines_no_eol). Universal-newline split so
    both LF and CRLF sources are handled; empty trailing lines dropped."""
    text = path.read_text(encoding="utf-8")
    raw = text.splitlines()
    lines = [ln for ln in raw if ln != ""]
    if not lines:
        return "", []
    return lines[0], lines[1:]


def ends_with_newline(path: Path) -> bool:
    with path.open("rb") as fh:
        try:
            fh.seek(-1, io.SEEK_END)
        except OSError:
            return True  # empty file
        return fh.read(1) in (b"\n", b"\r")


def conform_row(line: str, src_cols, dst_cols) -> str:
    """CSV-parse a single rfaa data line (in src column order) and re-emit it
    conformed to the dest column layout: dest columns absent from src are
    blank-filled; src columns absent from dest are dropped."""
    fields = next(csv.reader([line]))
    by_name = dict(zip(src_cols, fields))
    out = [by_name.get(c, "") for c in dst_cols]
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(out)
    return buf.getvalue()


def append_rfaa_rows(dest_csv: Path, src_header, src_data, label, stats, dry_run):
    """Append rfaa data rows (CRLF) to dest_csv. Idempotent + header-checked."""
    dst_header, dst_data = read_lines_keep_split(dest_csv)
    dst_cols = dst_header.split(",")
    src_cols = src_header.split(",")

    # idempotency: already merged?
    if any(ln.split(",", 1)[0].startswith(RFAA_PREFIX) for ln in dst_data):
        stats[f"{label}_already_present"] += 1
        return

    rfaa_rows = [ln for ln in src_data if ln.split(",", 1)[0].startswith(RFAA_PREFIX)]
    if not rfaa_rows:
        stats[f"{label}_no_rfaa_rows"] += 1
        return

    if dst_cols != src_cols:
        # canonical schema differs (e.g. dest lacks covalent_info/is_covalent_binder,
        # or src has extra ligand_2_* geometric cols). Conform each rfaa row to the
        # dest layout: drop extra src cols, blank-fill missing dest cols.
        stats[f"{label}_conformed"] += 1
        dropped = [c for c in src_cols if c not in dst_cols]
        filled = [c for c in dst_cols if c not in src_cols]
        print(f"  conform ({label}) {dest_csv.name}: drop={dropped} fill={filled}")
        rfaa_rows = [conform_row(ln, src_cols, dst_cols) for ln in rfaa_rows]

    if dry_run:
        stats[f"{label}_would_append"] += 1
        return

    payload = ("" if ends_with_newline(dest_csv) else CRLF) + "".join(
        r + CRLF for r in rfaa_rows
    )
    with dest_csv.open("ab") as fh:
        fh.write(payload.encode("utf-8"))
    stats[f"{label}_appended"] += 1


def cmd_merge(args):
    dest = args.kinconfbench_root.resolve()
    staged = args.rfaa_staged_root.resolve()
    geo_dir = args.rfaa_geometric_dir.resolve()
    run_preds = (
        args.run_predictions_root.resolve()
        if args.run_predictions_root
        else (staged.parent / "predictions").resolve()
    )
    for label, pth in [("--kinconfbench-root", dest), ("--rfaa-staged-root", staged),
                       ("--rfaa-geometric-dir", geo_dir)]:
        if not pth.is_dir():
            raise SystemExit(f"{label} not found: {pth}")

    stats = {k: 0 for k in (
        "stems", "skip_no_canonical",
        "analysis_appended", "analysis_already_present", "analysis_no_rfaa_rows",
        "analysis_conformed", "analysis_would_append",
        "geom_appended", "geom_already_present", "geom_no_rfaa_rows",
        "geom_conformed", "geom_would_append", "geom_missing",
        "cifs_moved", "cifs_skipped_exist", "run_links_removed",
    )}
    skipped_stems = []

    for gene_dir in sorted(d for d in staged.iterdir() if d.is_dir()):
        preds = gene_dir / "predictions"
        if not preds.is_dir():
            continue
        gene = gene_dir.name
        for stem_dir in sorted(d for d in preds.iterdir() if d.is_dir()):
            stem = stem_dir.name
            src_analysis = stem_dir / f"{stem}_analysis.csv"
            if not any(stem_dir.glob("rfaa_conf*.cif")) or not src_analysis.is_file():
                continue
            stats["stems"] += 1

            dst_analysis = dest / gene / "analysis_csvs" / f"{stem}_analysis.csv"
            dst_geom = dest / gene / "geometric_analysis_csvs" / f"{stem}_geometric.csv"
            dst_preds = dest / gene / "predictions" / stem

            if not dst_analysis.is_file():
                stats["skip_no_canonical"] += 1
                skipped_stems.append(f"{gene}/{stem}")
                continue

            # 1. analysis
            a_hdr, a_data = read_lines_keep_split(src_analysis)
            append_rfaa_rows(dst_analysis, a_hdr, a_data, "analysis", stats, args.dry_run)

            # 2. geometric
            src_geom = geo_dir / f"{stem}_rfaa_geometric.csv"
            if not src_geom.is_file():
                stats["geom_missing"] += 1
            elif not dst_geom.is_file():
                stats["geom_missing"] += 1
                print(f"  no dest geometric CSV for {gene}/{stem}")
            else:
                g_hdr, g_data = read_lines_keep_split(src_geom)
                append_rfaa_rows(dst_geom, g_hdr, g_data, "geom", stats, args.dry_run)

            # 3. CIFs (move staged link into dest; leave run-side link intact)
            dst_preds.mkdir(parents=True, exist_ok=True)
            for cif in sorted(stem_dir.glob("rfaa_conf*.cif")):
                target = dst_preds / cif.name
                if target.exists():
                    stats["cifs_skipped_exist"] += 1
                    continue
                if args.dry_run:
                    stats["cifs_moved"] += 1
                    continue
                cif.rename(target)  # same-fs rename of the hardlink
                stats["cifs_moved"] += 1

            # 3b. optional destructive cleanup of the run-side hardlink
            if args.cleanup_run_links and not args.dry_run:
                run_stem = run_preds / stem
                if run_stem.is_dir():
                    for cif in sorted(run_stem.glob("rfaa_conf*.cif")):
                        cif.unlink()
                        stats["run_links_removed"] += 1

    lines = [f"{k}: {v}" for k, v in stats.items()]
    if skipped_stems:
        lines.append("skip_no_canonical_stems:")
        lines.extend(f"  {s}" for s in skipped_stems)
    report = "\n".join(lines) + "\n"
    print("\n=== merge_rfaa summary" + (" (DRY-RUN)" if args.dry_run else "") + " ===")
    print(report)
    if not args.dry_run:
        manifest = dest / "merge_rfaa_manifest.txt"
        manifest.write_text(report, encoding="utf-8")
        print(f"Manifest: {manifest}")


# ======================================================================= parser
def build_parser():
    p = argparse.ArgumentParser(
        prog="rfaa_pipeline.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("patch", help="patch a RFAA clone (seed + psipred fallback)")
    sp.add_argument("repo", nargs="?", default=None, help="RFAA repo (default: $RFAA_REPO)")
    sp.set_defaults(func=cmd_patch)

    sp = sub.add_parser("confidence", help="extract confidence.json from *_aux.pt (torch)")
    sp.add_argument("dir", help="dir scanned recursively for */seed*/*_aux.pt")
    sp.set_defaults(func=cmd_confidence)

    sp = sub.add_parser("stage", help="flat predictions/ -> gene-nested staged layout")
    sp.add_argument("--run", default=f"{WS}/RFAA/runs/full_1420")
    sp.add_argument("--staged", default=None, help="staged root (default: <run>/kinconfbench_staged)")
    sp.add_argument("--meta-csv", default=META_CSV)
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_stage)

    sp = sub.add_parser("build-msas", help="pre-build per-protein MSAs (sharded sbatch)")
    sp.add_argument("--meta-csv", default=META_CSV, help="CSV with gene_name,ligand,sequence,SMILES")
    sp.add_argument("--cache", default=os.environ.get("RFAA_MSA_CACHE") or f"{WS}/RFAA/msa_cache",
                    help="per-protein MSA cache root (<cache>/<protein>/A/t000_.*)")
    sp.add_argument("--out", default=f"{WS}/RFAA/msa_build",
                    help="where FASTAs + shard scripts + manifest are written")
    sp.add_argument("--nshards", type=int, default=5)
    sp.add_argument("--cpus", type=int, default=32, help="cpus-per-task / make_msa threads")
    sp.add_argument("--mem", type=int, default=64, help="make_msa MEM (GB)")
    sp.add_argument("--account", default="m2834")
    sp.add_argument("--qos", default="premium")
    sp.add_argument("--time", default="06:00:00")
    sp.set_defaults(func=cmd_build_msas)

    sp = sub.add_parser("run", help="resumable, shardable full inference driver")
    sp.add_argument("--meta-csv", default=META_CSV)
    sp.add_argument("--output-folder", default=f"{WS}/RFAA/runs/full_1420")
    sp.add_argument("--num-samples", type=int, default=20)
    sp.add_argument("--gpus", type=int, default=4)
    sp.add_argument("--head", type=int, default=None, help="run first N combos (smoke)")
    sp.add_argument("--shard", type=int, default=None, help="this shard index (combos[shard::nshards])")
    sp.add_argument("--nshards", type=int, default=8)
    sp.add_argument("--combos-file", default=None,
                    help="run only the combo names listed in this file (one per line; targeted rerun)")
    sp.add_argument("--account", default="m2834_g")
    sp.add_argument("--smoke", type=int, default=20, help="combo count for the emitted smoke sbatch")
    sp.add_argument("--emit-sbatch", action="store_true")
    sp.add_argument("--dry-run", action="store_true", help="write per-GPU scripts but do not launch")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("merge", help="append-only merge into canonical kinconfbench tree")
    sp.add_argument("--kinconfbench-root", type=Path, required=True)
    sp.add_argument("--rfaa-staged-root", type=Path, required=True)
    sp.add_argument("--rfaa-geometric-dir", type=Path, required=True)
    sp.add_argument("--run-predictions-root", type=Path, default=None,
                    help="full_1420/predictions (the other hardlink); used only with "
                    "--cleanup-run-links. Defaults to <staged-root>/../predictions.")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--cleanup-run-links", action="store_true",
                    help="after a successful merge, remove the redundant run-side rfaa "
                    "CIF hardlinks so the only link lives in the canonical dest.")
    sp.set_defaults(func=cmd_merge)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

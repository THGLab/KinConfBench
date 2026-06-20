#!/usr/bin/env python3
"""
Idempotently patch a RoseTTAFold-All-Atom clone so KinConfBench can drive it.

Patch 1 (seed): upstream run_inference.py produces one deterministic structure
and never seeds the RNGs from config. We inject a seed read (config.seed) + RNG
seeding at the top of main(), so running N times with different seeds yields N
conformations (the adapter uses this to emulate the diffusion backends).

Patch 2 (psipred fallback): PSIPRED (psipred 4.01) buffer-overflows on modern
glibc and emits an empty ss2, which makes hhsearch reject the ss2-prepended a3m
("sequence ss_pred contains no residues") → no t000_.atab → inference crashes.
We make make_msa.sh use the ss2 only when it has real residues, else search the
plain a3m (templates are still found; ss2 is only a minor ranking enhancement).

Usage:
    python apply_patches.py [RFAA_REPO]      # defaults to $RFAA_REPO
"""
import os
import sys
import pathlib

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


def _patch(target, marker, orig, injected, name):
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


def main():
    repo = pathlib.Path(
        sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RFAA_REPO", "")
    )
    if not repo or not repo.is_dir():
        sys.exit(f"RFAA repo not found: {repo!r} (pass path or set RFAA_REPO)")

    _patch(repo / "rf2aa" / "run_inference.py", SEED_MARKER, SEED_ORIG,
            SEED_INJECTED, "seed")
    _patch(repo / "make_msa.sh", MSA_MARKER, MSA_ORIG, MSA_INJECTED,
            "psipred-fallback")


if __name__ == "__main__":
    main()

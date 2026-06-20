#!/usr/bin/env python3
"""
Build SI table: spatial x dihedral label percentages for KinConfBench benchmark
chains vs Boltz-2 / Chai-1 / Protenix predictions.

Scans every per-system analysis CSV under a kinconfbench tree (default:
<repo>/kinase_all_data/kinconfbench/*/analysis_csvs). Includes all files that
contain a PDB ground-truth row (4-letter structure_id), typically 1420 systems.

Model percentages use the total labeled prediction rows per method as the
denominator (sums to 100% over the 12 categories). CSVs should contain only
_conf0--_conf19 prediction rows (see scripts/trim_kinconfbench_analysis_confs.py);
expect 1420*20 rows per method except rare missing rows (e.g. Chai in VRK1_HUMAN_8E1).

KCB column uses human_kinases_wt_nophos_nocov_filtered.csv with denominator 2225.

Empty spatial/dihedral fields are normalized to the string "None" to match
Table S1 buckets.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TABLE_S1_ROWS: List[Tuple[str, str]] = [
    ("DFGin", "BLAminus"),
    ("DFGin", "BLAplus"),
    ("DFGin", "ABAminus"),
    ("DFGin", "BLBminus"),
    ("DFGin", "BLBplus"),
    ("DFGin", "BLBtrans"),
    ("DFGin", "None"),
    ("DFGinter", "BABtrans"),
    ("DFGinter", "None"),
    ("DFGout", "BBAminus"),
    ("DFGout", "None"),
    ("None", "None"),
]

METHODS = ["boltz", "chai", "protenix"]
CONF_RE = re.compile(r"_conf(\d+)")


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def norm_label(val: Optional[str]) -> str:
    s = (val or "").strip()
    return "None" if s == "" else s


def is_ground_truth_row(structure_id: str) -> bool:
    sid = structure_id.strip()
    if len(sid) != 4:
        return False
    return sid.replace("_", "").replace("-", "").isalnum()


def method_from_structure_id(structure_id: str) -> Optional[str]:
    if not structure_id or "_" not in structure_id:
        return None
    prefix = structure_id.split("_", 1)[0]
    return prefix if prefix in METHODS else None


def discover_kinconfbench_csv_paths(analysis_root: Path) -> List[Path]:
    """All *_analysis.csv under root with at least one PDB ground-truth row."""
    pattern = str(analysis_root / "*/analysis_csvs/*_analysis.csv")
    paths = sorted(Path(p) for p in glob.glob(pattern))
    out: List[Path] = []
    for p in paths:
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        if any(is_ground_truth_row(r.get("structure_id", "") or "") for r in rows):
            out.append(p)
    return out


def aggregate_model_counts(paths: List[Path]) -> Dict[str, Dict[Tuple[str, str], int]]:
    """method -> (spatial, dihedral) -> count."""
    out: Dict[str, Dict[Tuple[str, str], int]] = {m: defaultdict(int) for m in METHODS}
    for p in paths:
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            sid = r.get("structure_id") or ""
            meth = method_from_structure_id(sid)
            if meth is None:
                continue
            if CONF_RE.search(sid) is None:
                continue
            sp = norm_label(r.get("spatial_label"))
            dh = norm_label(r.get("dihedral_label"))
            out[meth][(sp, dh)] += 1
    return {m: dict(out[m]) for m in METHODS}


def benchmark_counts(benchmark_csv: Path) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    with open(benchmark_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sp = norm_label(r.get("spatial_label"))
            dh = norm_label(r.get("dihedral_label"))
            counts[(sp, dh)] += 1
    return dict(counts)


def pct(n: int, denom: float) -> float:
    return 100.0 * float(n) / denom if denom else 0.0


def format_latex_table(
    kcb_counts: Dict[Tuple[str, str], int],
    model_counts: Dict[str, Dict[Tuple[str, str], int]],
    n_benchmark_chains: int,
    n_systems: int,
    pred_totals: Dict[str, int],
) -> str:
    nb = pred_totals["boltz"]
    nc = pred_totals["chai"]
    np_ = pred_totals["protenix"]
    lines = [
        r"% Regenerate: python KinConfBench/analysis/kincore_label_analysis/si_spatial_dihedral_distribution_table.py "
        r"(see --help for paths).",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Distribution of KinCoRe spatial and dihedral labels. "
        r"\textbf{KCB} percentages use chain-level counts from the wild-type, "
        r"non-phosphorylated, no-covalent-ligand benchmark metadata (2225 chains; "
        r"same categories as Table~S\ref{suptab:kincore_summary}). "
        r"\textbf{Boltz-2}, \textbf{Chai-1}, and \textbf{Protenix} percentages are "
        r"over labeled prediction rows (\texttt{\_conf0}--\texttt{\_conf19} only) "
        r"in the \texttt{kinase\_all\_data/kinconfbench} analysis CSVs ($N=1420$ systems; "
        f"row totals Boltz-2 ${nb}$, Chai-1 ${nc}$, Protenix ${np_}$).}}",
        r"\label{suptab:spatial_dihedral_distribution_pct}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Spatial} & \textbf{Dihedral} & \textbf{KCB (\%)} & "
        r"\textbf{Boltz-2 (\%)} & \textbf{Chai-1 (\%)} & \textbf{Protenix (\%)} \\",
        r"\midrule",
    ]
    for sp, dh in TABLE_S1_ROWS:
        k = (sp, dh)
        kcb_n = kcb_counts.get(k, 0)
        row_bits = [
            sp,
            dh,
            f"{pct(kcb_n, n_benchmark_chains):.1f}",
        ]
        for m in METHODS:
            c = model_counts[m].get(k, 0)
            row_bits.append(f"{pct(c, pred_totals[m]):.1f}")
        lines.append(" & ".join(row_bits) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> int:
    repo = repo_root_from_script()
    default_root = repo / "kinase_all_data" / "kinconfbench"
    default_bench = (
        repo
        / "outputs"
        / "run_output_four_methods_original"
        / "fig_dist"
        / "human_kinases_wt_nophos_nocov_filtered.csv"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=default_root,
        help=f"Kinconfbench root (default: {default_root})",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=default_bench,
        help="Benchmark chain metadata CSV",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write LaTeX table fragment here",
    )
    args = parser.parse_args()

    if not args.analysis_root.is_dir():
        print(f"analysis root not found: {args.analysis_root}", file=sys.stderr)
        return 1
    if not args.benchmark_csv.is_file():
        print(f"benchmark csv not found: {args.benchmark_csv}", file=sys.stderr)
        return 1

    paths = discover_kinconfbench_csv_paths(args.analysis_root)
    n_systems = len(paths)

    kcb_raw = benchmark_counts(args.benchmark_csv)
    n_benchmark = sum(kcb_raw.values())

    allowed = set(TABLE_S1_ROWS)
    stray = {k: v for k, v in kcb_raw.items() if k not in allowed}
    if stray:
        print(
            "Warning: benchmark CSV has label pairs outside Table S1 rows:",
            stray,
            file=sys.stderr,
        )

    kcb_counts = {k: kcb_raw.get(k, 0) for k in TABLE_S1_ROWS}
    model_counts = aggregate_model_counts(paths)

    pred_totals: Dict[str, int] = {}
    for m in METHODS:
        pred_totals[m] = sum(model_counts[m].values())
        extra = sum(v for k, v in model_counts[m].items() if k not in allowed)
        if extra:
            print(f"Error: {m} has counts outside Table S1 rows: {extra}", file=sys.stderr)
            return 1
        bucket_sum = sum(model_counts[m].get(k, 0) for k in TABLE_S1_ROWS)
        if bucket_sum != pred_totals[m]:
            print(
                f"Error: {m} bucket sum {bucket_sum} != total {pred_totals[m]}",
                file=sys.stderr,
            )
            return 1

    tex = format_latex_table(
        kcb_counts,
        model_counts,
        n_benchmark_chains=n_benchmark,
        n_systems=n_systems,
        pred_totals=pred_totals,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(tex, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    print(tex, end="")
    print(
        f"# systems={n_systems} boltz_rows={pred_totals['boltz']} "
        f"chai_rows={pred_totals['chai']} protenix_rows={pred_totals['protenix']} "
        f"benchmark_chains={n_benchmark}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

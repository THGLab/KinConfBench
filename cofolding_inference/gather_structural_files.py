"""
Extract structural prediction files from Boltz, Chai, and Protenix and summarize them
into one folder for visualization.
Can be run standalone or called from step_2_run_all_inferences.py.

Author: Oliver Sun
Date: 2025-05-23
"""

import os
import re
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def scan_model_directories(base_path: str) -> Dict[str, List[str]]:
    """
    Scan the base path for all model directories and their protein combinations.

    Args:
        base_path: Path to the base directory containing model folders

    Returns:
        Dictionary mapping model names to lists of protein combinations
    """
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base path {base_path} does not exist")

    model_protein_map: Dict[str, List[str]] = {}
    expected_models = ["boltz", "chai", "protenix", "rfaa"]

    for model_name in expected_models:
        model_dir = base_path / model_name
        if model_dir.exists() and model_dir.is_dir():
            protein_combos = []
            for item in model_dir.iterdir():
                if item.is_dir():
                    protein_combos.append(item.name)
            model_protein_map[model_name] = protein_combos
            print(f"Found {len(protein_combos)} protein combinations for {model_name}")

    return model_protein_map


def collect_boltz_files(base_path: Path, protein_combos: List[str]) -> List[Tuple[str, str, Path]]:
    """
    Collect .cif files from Boltz output directories.

    Args:
        base_path: Path to the base directory containing the boltz folder
        protein_combos: List of protein combination folder names

    Returns:
        List of tuples (protein_combo, conf_idx, cif_file_path)
    """
    cif_files: List[Tuple[str, str, Path]] = []
    boltz_dir = base_path / "boltz"

    if not boltz_dir.exists():
        print(f"Warning: Boltz directory {boltz_dir} does not exist")
        return cif_files

    for protein_combo in protein_combos:
        protein_dir = boltz_dir / protein_combo
        if not protein_dir.exists():
            print(f"Warning: Protein directory {protein_dir} does not exist")
            continue

        for result_dir in protein_dir.glob("boltz_results_*"):
            if result_dir.is_dir():
                predictions_dir = result_dir / "predictions"
                if predictions_dir.exists():
                    for pred_subdir in predictions_dir.iterdir():
                        if pred_subdir.is_dir():
                            for i, cif_file in enumerate(pred_subdir.glob("*.cif")):
                                model_idx = i
                                cif_files.append((protein_combo, model_idx, cif_file))

    return cif_files


def collect_chai_files(base_path: Path, protein_combos: List[str]) -> List[Tuple[str, str, Path]]:
    """
    Collect .cif files from Chai output directories.

    Args:
        base_path: Path to the base directory containing the chai folder
        protein_combos: List of protein combination folder names

    Returns:
        List of tuples (protein_combo, conf_idx, cif_file_path)
    """
    cif_files: List[Tuple[str, str, Path]] = []
    chai_dir = base_path / "chai"

    if not chai_dir.exists():
        print(f"Warning: Chai directory {chai_dir} does not exist")
        return cif_files

    for protein_combo in protein_combos:
        protein_dir = chai_dir / protein_combo
        if not protein_dir.exists():
            print(f"Warning: Protein directory {protein_dir} does not exist")
            continue

        length = 0
        for i, cif_file in enumerate(protein_dir.glob("pred.model_idx_*.cif")):
            model_idx = i
            cif_files.append((protein_combo, model_idx, cif_file))
            length += 1

        for j, cif_file in enumerate(protein_dir.glob("batch*/pred.model_idx_*.cif")):
            model_idx = j + length
            cif_files.append((protein_combo, model_idx, cif_file))

    return cif_files


def collect_protenix_files(base_path: Path, protein_combos: List[str]) -> List[Tuple[str, str, Path]]:
    """
    Collect .cif files from Protenix output directories.

    Args:
        base_path: Path to the base directory containing the protenix folder
        protein_combos: List of protein combination folder names

    Returns:
        List of tuples (protein_combo, conf_idx, cif_file_path)
    """
    cif_files: List[Tuple[str, str, Path]] = []
    protenix_dir = base_path / "protenix"

    if not protenix_dir.exists():
        print(f"Warning: Protenix directory {protenix_dir} does not exist")
        return cif_files

    for protein_combo in protein_combos:
        protein_dir = protenix_dir / protein_combo
        if not protein_dir.exists():
            print(f"Warning: Protein directory {protein_dir} does not exist")
            continue

        for i, cif_file in enumerate(protein_dir.glob("*/*/predictions/*.cif")):
            model_idx = i
            cif_files.append((protein_combo, model_idx, cif_file))

    return cif_files


def _read_rfaa_confidence(seed_dir: Path) -> Optional[dict]:
    """Return a seed's RFAA confidence (mean_plddt, pae_*) or None.

    Prefers confidence.json (written by rfaa_pipeline.py confidence in the RFAA env,
    which has torch); falls back to reading <job>_aux.pt directly only if torch
    happens to be importable in the current env (it usually is not).
    """
    cj = seed_dir / "confidence.json"
    if cj.is_file():
        try:
            with open(cj) as f:
                return json.load(f)
        except Exception:
            pass
    aux = next(iter(sorted(seed_dir.glob("*_aux.pt"))), None)
    if aux is not None:
        try:
            import torch  # only if available in this env
            a = torch.load(str(aux), map_location="cpu")
            return {k: (float(a[k]) if a.get(k) is not None else None)
                    for k in ("mean_plddt", "mean_pae", "pae_prot", "pae_inter")}
        except Exception:
            return None
    return None


def rank_rfaa_records(rfaa_dir: Path, protein_combo: str) -> List[dict]:
    """Rank a combo's RFAA seeds by mean_plddt (descending) -> conf0 is best.

    Returns one dict per structure with keys: conf_idx, seed, mean_plddt,
    mean_pae, pae_prot, pae_inter, pdb_path. Seeds with no confidence score sort
    last; if NONE have a score we keep seed order and warn (so the gather still
    works on pre-confidence data — run rfaa_pipeline.py confidence to enable ranking).
    """
    protein_dir = rfaa_dir / protein_combo
    records: List[dict] = []
    for seed_dir in sorted(protein_dir.glob("seed*"), key=lambda p: p.name):
        pdb = next(iter(sorted(seed_dir.glob("*.pdb"))), None)
        if pdb is None:
            continue
        conf = _read_rfaa_confidence(seed_dir) or {}
        seed_no = conf.get("seed")
        if seed_no is None:
            m = re.search(r"seed(\d+)", seed_dir.name)
            seed_no = int(m.group(1)) if m else len(records)
        records.append({
            "seed": seed_no,
            "mean_plddt": conf.get("mean_plddt"),
            "mean_pae": conf.get("mean_pae"),
            "pae_prot": conf.get("pae_prot"),
            "pae_inter": conf.get("pae_inter"),
            "pdb_path": pdb,
        })

    if any(r["mean_plddt"] is not None for r in records):
        records.sort(key=lambda r: (
            -(r["mean_plddt"] if r["mean_plddt"] is not None else float("-inf")),
            r["seed"],
        ))
    else:
        print(f"Warning: no RFAA confidence found for {protein_combo}; ranking by "
              "seed order (run rfaa_pipeline.py confidence to enable pLDDT ranking)")
        records.sort(key=lambda r: r["seed"])

    for i, r in enumerate(records):
        r["conf_idx"] = i
    return records


def collect_rfaa_files(base_path: Path, protein_combos: List[str]) -> List[Tuple[str, str, Path]]:
    """
    Collect .pdb files from RoseTTAFold-All-Atom output directories.

    RFAA writes one PDB per seed at <base>/rfaa/<combo>/seed<N>/<job>.pdb.
    Conf indices follow the mean_plddt ranking (conf0 = highest pLDDT). Returns
    tuples (protein_combo, conf_idx, pdb_file_path); the PDB is converted to CIF
    in organize_predictions.
    """
    pdb_files: List[Tuple[str, str, Path]] = []
    rfaa_dir = base_path / "rfaa"

    if not rfaa_dir.exists():
        print(f"Warning: RFAA directory {rfaa_dir} does not exist")
        return pdb_files

    for protein_combo in protein_combos:
        protein_dir = rfaa_dir / protein_combo
        if not protein_dir.exists():
            print(f"Warning: Protein directory {protein_dir} does not exist")
            continue
        for rec in rank_rfaa_records(rfaa_dir, protein_combo):
            pdb_files.append((protein_combo, rec["conf_idx"], rec["pdb_path"]))

    return pdb_files


def _uniquify_atom_names(st) -> None:
    """Make atom names unique within each residue.

    RFAA writes ligand atoms with element-only names (e.g. 22 atoms all named
    "C"), which collide when parsed as CIF ("Atom C defined twice"). We rename
    atoms only in residues that actually have duplicates (ligands), so standard
    protein atom names (N, CA, C, O, ...) are preserved.
    """
    for model in st:
        for chain in model:
            for res in chain:
                names = [a.name for a in res]
                if len(names) == len(set(names)):
                    continue  # already unique (e.g. a protein residue)
                counters: dict = {}
                for atom in res:
                    el = atom.element.name or atom.name
                    counters[el] = counters.get(el, 0) + 1
                    atom.name = f"{el}{counters[el]}"


def _fix_polymer_labels(st) -> None:
    """Restore clean polymer mmCIF labels after ``setup_entities``.

    gemmi 0.7.x leaves ``label_seq_id`` unset and auto-names polymer subchains
    (e.g. "Axp"), so the written mmCIF has ``label_asym_id="Axp"`` and a missing
    ``label_seq_id``. Downstream CIF parsers (the kincore classifier) then find
    "no kinase chains". Protein+ligand structures happened to come out clean
    under the older gemmi, but apo (protein-only) structures hit this and got
    skipped. Set ``subchain = chain name`` for every residue and assign a
    sequential ``label_seq`` to polymer residues (ligands keep ``label_seq``
    unset, as in standard mmCIF) so label_asym_id="A" and label_seq_id=1.. — the
    clean form every CIF reader expects."""
    for model in st:
        for chain in model:
            for res in chain:
                res.subchain = chain.name
            for i, res in enumerate(chain.get_polymer(), start=1):
                res.label_seq = i


def _pdb_to_cif(pdb_path: Path, cif_path: Path) -> None:
    """Convert a PDB to mmCIF, uniquifying colliding ligand atom names."""
    import gemmi
    st = gemmi.read_structure(str(pdb_path))
    _uniquify_atom_names(st)
    st.setup_entities()
    _fix_polymer_labels(st)
    st.make_mmcif_document().write_file(str(cif_path))


def organize_predictions(source_dir: str, target_dir: str = "predictions") -> None:
    """
    Organize all prediction files into the target structure.

    Args:
        source_dir: Path to the base directory containing model folders
        target_dir: Target directory name for organized predictions
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    target_path.mkdir(exist_ok=True)

    model_protein_map = scan_model_directories(source_path)

    if "boltz" in model_protein_map:
        print("\nProcessing Boltz files...")
        boltz_files = collect_boltz_files(source_path, model_protein_map["boltz"])
        for protein_combo, conf_idx, cif_file in boltz_files:
            protein_subfolder = target_path / protein_combo
            protein_subfolder.mkdir(exist_ok=True)

            target_filename = f"boltz_conf{conf_idx}.cif"
            target_path_full = protein_subfolder / target_filename
            shutil.copy2(cif_file, target_path_full)

    if "chai" in model_protein_map:
        print("\nProcessing Chai files...")
        chai_files = collect_chai_files(source_path, model_protein_map["chai"])
        for protein_combo, conf_idx, cif_file in chai_files:
            protein_subfolder = target_path / protein_combo
            protein_subfolder.mkdir(exist_ok=True)

            target_filename = f"chai_conf{conf_idx}.cif"
            target_path_full = protein_subfolder / target_filename
            shutil.copy2(cif_file, target_path_full)

    if "protenix" in model_protein_map:
        print("\nProcessing Protenix files...")
        protenix_files = collect_protenix_files(source_path, model_protein_map["protenix"])
        for protein_combo, conf_idx, cif_file in protenix_files:
            protein_subfolder = target_path / protein_combo
            protein_subfolder.mkdir(exist_ok=True)

            target_filename = f"protenix_conf{conf_idx}.cif"
            target_path_full = protein_subfolder / target_filename
            shutil.copy2(cif_file, target_path_full)

    if "rfaa" in model_protein_map:
        print("\nProcessing RFAA files...")
        rfaa_dir = source_path / "rfaa"
        for protein_combo in model_protein_map["rfaa"]:
            if not (rfaa_dir / protein_combo).is_dir():
                continue
            records = rank_rfaa_records(rfaa_dir, protein_combo)
            protein_subfolder = target_path / protein_combo
            protein_subfolder.mkdir(exist_ok=True)

            ranking_rows = []
            for rec in records:
                target_filename = f"rfaa_conf{rec['conf_idx']}.cif"
                _pdb_to_cif(rec["pdb_path"], protein_subfolder / target_filename)
                ranking_rows.append({
                    "structure_id": f"rfaa_conf{rec['conf_idx']}",
                    "conf_idx": rec["conf_idx"],
                    "seed": rec["seed"],
                    "mean_plddt": rec["mean_plddt"],
                    "mean_pae": rec["mean_pae"],
                    "pae_prot": rec["pae_prot"],
                    "pae_inter": rec["pae_inter"],
                    "source_pdb": str(rec["pdb_path"]),
                })
            # Ranking manifest: conf0 = best (highest mean_plddt).
            pd.DataFrame(ranking_rows).to_csv(
                protein_subfolder / "rfaa_ranking.csv", index=False)
            ranked = sum(1 for r in records if r["mean_plddt"] is not None)
            print(f"  {protein_combo}: wrote {len(records)} CIFs"
                  + (f" ranked by mean_plddt (conf0=best, {ranked} scored)"
                     if ranked else " in seed order (no confidence found)")
                  + " + rfaa_ranking.csv")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Organize structural prediction files from different models into a unified directory structure."
    )

    parser.add_argument(
        "--source",
        type=str,
        default=".",
        help="Path to the base directory containing model folders",
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target directory for organized predictions (default: <source>/predictions)",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    source_dir = args.source
    target_dir = args.target if args.target is not None else str(Path(source_dir) / "predictions")

    print("Starting structural prediction file organization...")
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")

    try:
        organize_predictions(source_dir, target_dir)
        print("\nFile organization completed successfully!")

    except Exception as e:
        print(f"Error during file organization: {e}")
        raise


if __name__ == "__main__":
    main()

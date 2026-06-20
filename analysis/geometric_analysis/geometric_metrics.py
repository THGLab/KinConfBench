"""
Core geometric metrics for protein-ligand cofolding benchmarks.

Uses OpenStructure for lDDT-CA, lDDT-PLI, ligand RMSD (pocket-aligned),
and protein CA RMSD (sequence-aligned).
Uses BioPython pairwise2 for sequence alignment.
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import TYPE_CHECKING

import numpy as np
from Bio import pairwise2

if TYPE_CHECKING:
    import pandas as pd

# Ligand residue/chain names to exclude from protein CA analysis
_LIGAND_RESIDUE_NAMES = frozenset({"LIG1", "LIG2", "LG1", "l01", "l02"})
_LIGAND_CHAIN_NAMES = frozenset(
    {"LIG", "LIG1", "LIG2", "LIGA", "LIGB", "LG", "LG1", "LG2", "l01", "l02", "UNL", "MOL"}
)

import ost
from ost import geom as _ost_geom
from ost import io as _ost_io
from ost import mol as _ost_mol
from ost import seq as _ost_seq
from ost.mol.alg.lddt import lDDTScorer
from ost.mol.alg.scoring_base import MMCIFPrep
from ost.mol.alg.ligand_scoring_lddtpli import LDDTPLIScorer
from ost.mol.alg.ligand_scoring_scrmsd import SCRMSDScorer

# Single-bond covalent radii (Å) for common elements in drug-like molecules.
_COVALENT_RADII: dict[str, float] = {
    "B": 0.82, "BR": 1.14, "C": 0.77, "CL": 0.99, "F": 0.64,
    "I": 1.33, "N": 0.75, "O": 0.73, "P": 1.07, "S": 1.02,
    "SE": 1.20, "SI": 1.11,
}
_BOND_TOLERANCE = 0.4


def _strip_hydrogens(ent):
    """Return a copy of *ent* with all hydrogen atoms removed."""
    view = ent.Select("ele!=H")
    return _ost_mol.CreateEntityFromView(view, False)


def _prune_long_bonds(lig) -> int:
    """Delete bonds whose length exceeds the sum of covalent radii + tolerance.

    OST's distance-based connectivity heuristic can add spurious bonds
    when atoms in strained rings have short cross-ring distances (e.g.
    cyclopropane).  *lig* may be an EntityHandle or a ResidueHandle.
    Returns the number of bonds removed.
    """
    if isinstance(lig, _ost_mol.EntityHandle):
        owner = lig
    else:
        owner = lig.GetEntity()
    ed = owner.EditXCS()
    to_delete = []
    seen: set[tuple[str, str]] = set()
    for atom in lig.atoms:
        for bond in atom.GetBondList():
            a1, a2 = bond.GetFirst(), bond.GetSecond()
            pair = tuple(sorted([a1.GetName(), a2.GetName()]))
            if pair in seen:
                continue
            seen.add(pair)
            dist = _ost_geom.Distance(a1.GetPos(), a2.GetPos())
            r1 = _COVALENT_RADII.get(a1.GetElement().upper(), 0.77)
            r2 = _COVALENT_RADII.get(a2.GetElement().upper(), 0.77)
            if dist > r1 + r2 + _BOND_TOLERANCE:
                to_delete.append(bond)
    for bond in to_delete:
        ed.DeleteBond(bond)
    return len(to_delete)


def _extract_ligands_from_entity(ent):
    """Extract non-polymer residues (ligands) from entity."""
    ligands = []
    for chain in ent.chains:
        for res in chain.residues:
            if not res.IsPeptideLinking() and not res.IsNucleotideLinking():
                ligands.append(res)
    return ligands


def _strip_entity_poly_loop(cif_path: str) -> str:
    """Remove the ``_entity_poly`` loop from an mmCIF file.

    Some co-folding tools (e.g. Boltz apo predictions) emit a malformed
    ``_entity_poly`` loop that crashes OST's mmCIF reader.  The loop is
    not required for coordinate loading, so we strip it entirely.
    The related ``_entity_poly_seq`` loop is kept.

    Always writes a temp file (caller must unlink).
    """
    with open(cif_path) as fh:
        lines = fh.readlines()

    out: list[str] = []
    skip = False
    in_semicolon = False
    for line in lines:
        stripped = line.strip()

        if in_semicolon:
            if stripped.startswith(";"):
                in_semicolon = False
            if skip:
                continue
            out.append(line)
            continue

        if stripped.startswith(";"):
            in_semicolon = True
            if skip:
                continue
            out.append(line)
            continue

        if stripped == "loop_":
            skip = False
            out.append(line)
            continue

        if stripped.startswith("_entity_poly.") and not stripped.startswith("_entity_poly_seq."):
            skip = True
            if out and out[-1].strip() == "loop_":
                out.pop()
            continue

        if skip:
            if stripped.startswith("_") or stripped == "#" or stripped == "loop_":
                skip = False
                out.append(line)
            continue

        out.append(line)

    fd, tmp_path = tempfile.mkstemp(suffix=".cif")
    os.close(fd)
    with open(tmp_path, "w") as fh:
        fh.writelines(out)
    return tmp_path


def _sanitize_cif_comp_ids(cif_path: str) -> str | None:
    """Rename HETATM compound IDs that collide with the OST compound library.

    Co-folding methods sometimes assign generic ligand comp_ids (e.g. LG1,
    LG2) that happen to match real PDB compounds. When OST finds them in
    its compound library it expects the library's atom names, leading to
    broken connectivity and failed lDDT-PLI / SCRMSD scoring.  Renaming
    the comp_ids to values absent from the library forces the distance-based
    connectivity heuristic, which works correctly with the actual atom
    coordinates.

    Returns a temp-file path with sanitized content, or None if no changes
    were needed (caller should use the original path).
    """
    from ost import conop

    lib = conop.GetDefaultLib()

    with open(cif_path) as fh:
        lines = fh.readlines()

    comp_id_col: int | None = None
    col_idx = 0
    in_atom_site_header = False
    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            in_atom_site_header = False
            col_idx = 0
        elif stripped.startswith("_atom_site."):
            in_atom_site_header = True
            if stripped == "_atom_site.label_comp_id":
                comp_id_col = col_idx
            col_idx += 1
        elif in_atom_site_header and not stripped.startswith("_"):
            break

    if comp_id_col is None:
        return None

    clashing: set[str] = set()
    for line in lines:
        if line.startswith("HETATM"):
            parts = line.split()
            if len(parts) > comp_id_col:
                cid = parts[comp_id_col]
                if cid not in clashing and lib.FindCompound(cid):
                    clashing.add(cid)

    if not clashing:
        return None

    rename: dict[str, str] = {}
    for name in sorted(clashing):
        candidate = f"X{name}"
        while lib.FindCompound(candidate):
            candidate = f"X{candidate}"
        rename[name] = candidate

    new_lines: list[str] = []
    for line in lines:
        if line.startswith("HETATM"):
            parts = line.split()
            if len(parts) > comp_id_col and parts[comp_id_col] in rename:
                parts[comp_id_col] = rename[parts[comp_id_col]]
                line = " ".join(parts) + "\n"
        new_lines.append(line)

    fd, tmp_path = tempfile.mkstemp(suffix=".cif")
    os.close(fd)
    with open(tmp_path, "w") as fh:
        fh.writelines(new_lines)
    return tmp_path


def load_structure(
    cif_path: str,
    extract_ligands: bool = True,
    sanitize_comp_ids: bool = False,
):
    """
    Load mmCIF with OpenStructure.

    When extract_ligands is True, uses MMCIFPrep with fault_tolerant=True.
    Returns (entity, ligand_list) if extract_ligands else entity.

    Hydrogen atoms are always stripped from the returned entity (and
    ligands) so that PDB references containing explicit hydrogens are
    comparable with co-folding predictions that contain only heavy atoms.

    When sanitize_comp_ids is True, HETATM compound IDs that collide with
    the OST compound library are renamed before loading so that the
    distance-based connectivity heuristic is used instead of the (incorrect)
    library-based one.  Use this for co-folding prediction CIFs where
    generic comp_ids may accidentally match real PDB ligands.
    """
    tmp_path: str | None = None
    if sanitize_comp_ids:
        tmp_path = _sanitize_cif_comp_ids(cif_path)
    load_path = tmp_path if tmp_path else cif_path

    poly_tmp: str | None = None
    try:
        if extract_ligands:
            try:
                ent, ligands = MMCIFPrep(
                    load_path, extract_nonpoly=True, fault_tolerant=True
                )
                ent = _strip_hydrogens(ent)
                ligands = [_strip_hydrogens(lig) for lig in ligands]
            except RuntimeError:
                ent = _ost_io.LoadMMCIF(load_path, fault_tolerant=True)
                ent = _strip_hydrogens(ent)
                ligands = _extract_ligands_from_entity(ent)
            except Exception:
                poly_tmp = _strip_entity_poly_loop(load_path)
                try:
                    ent, ligands = MMCIFPrep(
                        poly_tmp, extract_nonpoly=True, fault_tolerant=True
                    )
                    ent = _strip_hydrogens(ent)
                    ligands = [_strip_hydrogens(lig) for lig in ligands]
                except RuntimeError:
                    ent = _ost_io.LoadMMCIF(poly_tmp, fault_tolerant=True)
                    ent = _strip_hydrogens(ent)
                    ligands = _extract_ligands_from_entity(ent)
            for lig in ligands:
                _prune_long_bonds(lig)
            return ent, ligands
        try:
            ent = MMCIFPrep(load_path, extract_nonpoly=False, fault_tolerant=True)
        except Exception:
            poly_tmp = _strip_entity_poly_loop(load_path)
            try:
                ent = MMCIFPrep(poly_tmp, extract_nonpoly=False, fault_tolerant=True)
            except RuntimeError:
                ent = _ost_io.LoadMMCIF(poly_tmp, fault_tolerant=True)
        return _strip_hydrogens(ent)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if poly_tmp and os.path.exists(poly_tmp):
            os.unlink(poly_tmp)


def _parse_dfg_resnums(row) -> tuple[int, int, int] | None:
    """
    Parse DFG motif residue numbers from analysis CSV row.

    Columns: asp_residue, phe_residue, gly_residue (e.g. D400.0, F401.0, G402.0).
    Returns (asp_num, phe_num, gly_num) or None.
    """
    try:
        asp = str(row.get("asp_residue", "")).strip()
        phe = str(row.get("phe_residue", "")).strip()
        gly = str(row.get("gly_residue", "")).strip()
        if not asp or not phe or not gly or asp == "nan" or phe == "nan" or gly == "nan":
            return None

        def _num(s: str) -> int | None:
            m = re.search(r"(\d+)", s)
            return int(m.group(1)) if m else None

        an, pn, gn = _num(asp), _num(phe), _num(gly)
        if an is None or pn is None or gn is None:
            return None
        return (an, pn, gn)
    except Exception:
        return None


def _is_ligand_chain(cname: str) -> bool:
    """True if chain name indicates a ligand chain."""
    c = cname.strip()
    if c in _LIGAND_CHAIN_NAMES:
        return True
    if c.startswith("LG") or c.startswith("LIG"):
        return True
    return False


def _get_protein_chain_names(ent, min_peptide_residues: int = 10):
    """Return chain names that are protein chains (exclude ligand chains)."""
    protein = []
    for c in ent.chains:
        cname = c.GetName()
        if _is_ligand_chain(cname):
            continue
        n_peptide = 0
        for r in c.residues:
            if r.IsPeptideLinking() and r.FindAtom("CA").IsValid():
                n_peptide += 1
            elif r.GetName().strip() in _LIGAND_RESIDUE_NAMES:
                continue
        if n_peptide >= min_peptide_residues:
            protein.append(cname)
    return protein


def _protein_only_entity(ent):
    """Create entity containing only protein (peptide) chains."""
    protein_names = _get_protein_chain_names(ent)
    if not protein_names:
        return None
    try:
        sel = " or ".join(f"cname='{n}'" for n in protein_names)
        view = ent.Select(sel)
        return _ost_mol.CreateEntityFromView(view, include_exlusive_atoms=False)
    except Exception:
        return None


def _get_chain_mapping(ref_ent, pred_ent):
    """Build chain mapping from pred chains to ref chains (protein only)."""
    ref_chains = _get_protein_chain_names(ref_ent)
    pred_chains = _get_protein_chain_names(pred_ent)
    if not ref_chains or not pred_chains:
        return {}
    if len(ref_chains) == 1 and len(pred_chains) == 1:
        return {pred_chains[0]: ref_chains[0]}
    return {pred_chains[i]: ref_chains[i] for i in range(min(len(pred_chains), len(ref_chains)))}


def _get_chain_residues_with_nums(ent, cname):
    """Return [(resnum, residue, ca_atom), ...] for peptide residues in chain."""
    for c in ent.chains:
        if c.GetName() == cname:
            out = []
            for r in c.residues:
                if not r.IsPeptideLinking():
                    continue
                ca = r.FindAtom("CA")
                if ca.IsValid():
                    out.append((r.GetNumber(), r, ca))
            return out
    return []


def compute_protein_rmsd(
    ref_ent,
    pred_ent,
    ref_dfg: tuple[int, int, int] | None = None,
    pred_dfg: tuple[int, int, int] | None = None,
) -> float | None:
    """
    CA RMSD after Kabsch superposition.

    When ref_dfg and pred_dfg are provided, uses DFG-anchored alignment.
    Otherwise uses sequence alignment to map residues, then Kabsch.
    """
    cmap = _get_chain_mapping(ref_ent, pred_ent)
    ref_coords, pred_coords = [], []

    for pred_cname, ref_cname in cmap.items():
        ref_res = _get_chain_residues_with_nums(ref_ent, ref_cname)
        pred_res = _get_chain_residues_with_nums(pred_ent, pred_cname)
        if not ref_res or not pred_res:
            continue

        if ref_dfg is not None and pred_dfg is not None:
            ref_asp_num, _, _ = ref_dfg
            pred_asp_num, _, _ = pred_dfg
            ref_asp_idx = next((i for i, (n, _, _) in enumerate(ref_res) if n == ref_asp_num), None)
            pred_asp_idx = next((i for i, (n, _, _) in enumerate(pred_res) if n == pred_asp_num), None)
            if ref_asp_idx is not None and pred_asp_idx is not None:
                offset = pred_asp_idx - ref_asp_idx
                for i, (_, _, rca) in enumerate(ref_res):
                    j = i + offset
                    if 0 <= j < len(pred_res):
                        _, _, pca = pred_res[j]
                        p, q = rca.GetPos(), pca.GetPos()
                        ref_coords.append([p[0], p[1], p[2]])
                        pred_coords.append([q[0], q[1], q[2]])
                if len(ref_coords) >= 3 and len(pred_coords) >= 3:
                    break
                ref_coords.clear()
                pred_coords.clear()

        rseq = "".join(r.GetOneLetterCode() or "X" for _, r, _ in ref_res)
        pseq = "".join(r.GetOneLetterCode() or "X" for _, r, _ in pred_res)
        alns = pairwise2.align.globalxx(rseq, pseq)
        if not alns:
            continue
        a, b = alns[0][:2]
        ri, pi = 0, 0
        for ar, pr in zip(a, b):
            if ar != "-" and pr != "-":
                if ri < len(ref_res) and pi < len(pred_res):
                    _, _, rca = ref_res[ri]
                    _, _, pca = pred_res[pi]
                    p, q = rca.GetPos(), pca.GetPos()
                    ref_coords.append([p[0], p[1], p[2]])
                    pred_coords.append([q[0], q[1], q[2]])
                ri += 1
                pi += 1
            elif ar != "-":
                ri += 1
            elif pr != "-":
                pi += 1
        break

    if len(ref_coords) < 3 or len(pred_coords) < 3:
        return None

    ref_arr = np.array(ref_coords, dtype=np.float64)
    pred_arr = np.array(pred_coords, dtype=np.float64)
    ref_cent = ref_arr.mean(axis=0)
    pred_cent = pred_arr.mean(axis=0)
    ref_c = ref_arr - ref_cent
    pred_c = pred_arr - pred_cent
    H = pred_c.T @ ref_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    pred_aligned = (pred_c @ R) + ref_cent
    rmsd = np.sqrt(((ref_arr - pred_aligned) ** 2).sum() / len(ref_arr))
    return float(rmsd)


def _build_residue_mapping(ref_ent, pred_ent, chain_mapping) -> dict | None:
    """Build sequence-alignment-based residue mapping for lDDT."""
    residue_mapping = {}
    for pred_cname, ref_cname in chain_mapping.items():
        ref_res = _get_chain_residues_with_nums(ref_ent, ref_cname)
        pred_res = _get_chain_residues_with_nums(pred_ent, pred_cname)
        if not ref_res or not pred_res:
            return None
        rseq = "".join(r.GetOneLetterCode() or "X" for _, r, _ in ref_res)
        pseq = "".join(r.GetOneLetterCode() or "X" for _, r, _ in pred_res)
        alns = pairwise2.align.globalxx(rseq, pseq)
        if not alns:
            return None
        ref_aligned, pred_aligned = alns[0][:2]
        s_ref = _ost_seq.CreateSequence("target", ref_aligned)
        s_pred = _ost_seq.CreateSequence("model", pred_aligned)
        residue_mapping[pred_cname] = _ost_seq.CreateAlignment(s_ref, s_pred)
    return residue_mapping if residue_mapping else None


def compute_lddt_ca(ref_ent, pred_ent) -> float | None:
    """
    lDDT on C-alpha atoms (protein backbone).
    Uses sequence-based residue mapping.
    """
    try:
        ref_protein = _protein_only_entity(ref_ent)
        pred_protein = _protein_only_entity(pred_ent)
        if ref_protein is None or pred_protein is None:
            return None
        chain_mapping = _get_chain_mapping(ref_protein, pred_protein)
        if not chain_mapping:
            return None
        residue_mapping = _build_residue_mapping(ref_protein, pred_protein, chain_mapping)
        scorer = lDDTScorer(ref_protein, bb_only=True)
        kwargs = {"chain_mapping": chain_mapping, "check_resnames": False}
        if residue_mapping:
            kwargs["residue_mapping"] = residue_mapping
        result = scorer.lDDT(pred_protein, **kwargs)
        global_lddt = result[0]
        return float(global_lddt) if global_lddt is not None else None
    except Exception:
        return None


def compute_lddt_pli(
    ref_ent, ref_ligands, pred_ent, pred_ligands, resnum_alignments: bool = False
) -> float | None:
    """Protein-ligand interface lDDT. Returns None when no ligands or match fails."""
    if not ref_ligands or not pred_ligands:
        return None
    try:
        sc = LDDTPLIScorer(
            pred_ent, ref_ent, pred_ligands, ref_ligands,
            resnum_alignments=resnum_alignments,
            rename_ligand_chain=True,
            substructure_match=True,
        )
        for trg_idx, mdl_idx in sc.assignment:
            if sc.state_matrix[trg_idx, mdl_idx] == 0:
                return float(sc.score_matrix[trg_idx, mdl_idx])
        return None
    except Exception:
        return None


def _get_ligand_display_name(lig) -> str:
    """Ligand display name: comp_id (residue type) or chain name when empty."""
    name = lig.GetName()
    if name and str(name).strip():
        return str(name).strip()
    return lig.chain.GetName() if lig.chain else "LIG"


def _compute_ligand_rmsd_ost(
    ref_ent, ref_ligands, pred_ent, pred_ligands, resnum_alignments: bool = False
) -> dict[tuple[int, int], float]:
    """
    OST SCRMSDScorer ligand RMSD per (trg_idx, mdl_idx) pair.
    Returns {(trg_idx, mdl_idx): rmsd} for pairs with valid scores.
    """
    rmsds: dict[tuple[int, int], float] = {}
    if not ref_ligands or not pred_ligands:
        return rmsds
    try:
        sc = SCRMSDScorer(
            pred_ent, ref_ent, pred_ligands, ref_ligands,
            resnum_alignments=resnum_alignments,
            rename_ligand_chain=True,
            substructure_match=True,
        )
        for trg_idx, mdl_idx in sc.assignment:
            if sc.state_matrix[trg_idx, mdl_idx] == 0:
                rmsds[(trg_idx, mdl_idx)] = float(sc.score_matrix[trg_idx, mdl_idx])
    except Exception:
        pass
    return rmsds


def compute_ligand_metrics_per_ligand(
    ref_ent,
    ref_ligands,
    pred_ent,
    pred_ligands,
    resnum_alignments: bool = False,
) -> list[dict]:
    """
    Per-ligand metrics: ligand_name, lddt_pli, ligand_rmsd.

    Uses LDDTPLIScorer for lddt_pli and SCRMSDScorer for ligand_rmsd.
    When assignment is empty, falls back to index-based pairing ref[i] <-> pred[i].
    Returns list of {ligand_name, lddt_pli, ligand_rmsd}.
    """
    if not ref_ligands or not pred_ligands:
        return []

    pairs: list[tuple[int, int]] = []
    lddt_scores: dict[tuple[int, int], float] = {}
    try:
        sc = LDDTPLIScorer(
            pred_ent, ref_ent, pred_ligands, ref_ligands,
            resnum_alignments=resnum_alignments,
            rename_ligand_chain=True,
            substructure_match=True,
        )
        for trg_idx, mdl_idx in sc.assignment:
            pairs.append((trg_idx, mdl_idx))
            lddt_scores[(trg_idx, mdl_idx)] = float(sc.score_matrix[trg_idx, mdl_idx])
    except Exception:
        pass

    if not pairs:
        n = min(len(ref_ligands), len(pred_ligands))
        pairs = [(i, i) for i in range(n)]

    rmsd_scores = _compute_ligand_rmsd_ost(
        ref_ent, ref_ligands, pred_ent, pred_ligands, resnum_alignments
    )

    out = []
    for trg_idx, mdl_idx in pairs:
        ref_lig = ref_ligands[trg_idx]
        out.append({
            "ligand_name": _get_ligand_display_name(ref_lig),
            "lddt_pli": lddt_scores.get((trg_idx, mdl_idx)),
            "ligand_rmsd": rmsd_scores.get((trg_idx, mdl_idx)),
        })
    return out

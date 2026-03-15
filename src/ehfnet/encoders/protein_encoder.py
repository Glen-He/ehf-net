"""
蛋白质结构编码器

提供蛋白质结构的特征提取和编码功能。
"""

from __future__ import annotations

import logging
import math
import numpy as np
import MDAnalysis as mda

from typing import TypedDict
from MDAnalysis.core.groups import Residue as MDAResidue

from ehfnet.encoders.chemistry import Element, ResidueType, resolve_protein_residue_type
from ehfnet.encoders.feature_specs import (
    PROTEIN_ATOM_CAT_SCHEMA,
    PROTEIN_ATOM_CONT_SCHEMA,
    PROTEIN_ATOM_NAME_TO_CLASS,
    PROTEIN_RESIDUE_BACKBONE_ATOM_NAMES,
    PROTEIN_RESIDUE_CAT_SCHEMA,
    PROTEIN_RESIDUE_CONT_SCHEMA,
    PROTEIN_RESIDUE_TORSION_NAMES,
)
from ehfnet.encoders.protein_segments import segment_residues_by_continuity
from ehfnet.geometry.static import calculate_dihedral

logger = logging.getLogger(__name__)

_BACKBONE_ATOM_SET = {"N", "CA", "C", "O", "OXT"}
_ATOM_NAME_UNK = PROTEIN_ATOM_NAME_TO_CLASS["UNK"]
_SEGMENT_LENGTH_NORM_BASE = math.log1p(512.0)

_SIDECHAIN_DONOR_ATOMS: dict[str, set[str]] = {
    "ARG": {"NE", "NH1", "NH2"},
    "ASN": {"ND2"},
    "CYS": {"SG"},
    "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"},
    "LYS": {"NZ"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TRP": {"NE1"},
    "TYR": {"OH"},
}
_SIDECHAIN_ACCEPTOR_ATOMS: dict[str, set[str]] = {
    "ASN": {"OD1"},
    "ASP": {"OD1", "OD2"},
    "CYS": {"SG"},
    "GLN": {"OE1"},
    "GLU": {"OE1", "OE2"},
    "HIS": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
}
_AROMATIC_ATOMS_BY_RESIDUE: dict[str, set[str]] = {
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
}


class ProteinEncodingResult(TypedDict):
    """
    蛋白质编码结果
    """

    atom_features: dict[str, list[int | float]]
    atom_positions: list[list[float]]
    atom_to_residue_index: list[int]

    residue_features: dict[str, list[int | float]]
    residue_positions: list[list[float]]
    residue_esm_embeddings: list[np.ndarray | None]
    residue_metadata: dict[str, list[int | float]]

    auxiliary: dict[str, list[list[float]]]


class ResidueGeometryResult(TypedDict):
    """
    残基几何特征计算结果。
    """

    torsion_features: list[float]
    observed_torsion_mask: list[float]
    observed_backbone_mask: list[float]


def _get_vecs_by_names(
    source_atoms: dict[str, np.ndarray],
    names: list[str],
) -> list[np.ndarray] | None:
    """
    根据名字取坐标，缺一个就返回 None。
    """

    vecs: list[np.ndarray] = []

    for name in names:
        if name not in source_atoms:
            return None
        vecs.append(source_atoms[name])

    return vecs


def _build_atom_lookup(residue: MDAResidue) -> dict[str, np.ndarray]:
    """
    构建原子名到坐标的查找表；重复原子名时保留首个观测值。
    """

    atoms: dict[str, np.ndarray] = {}

    for atom in residue.atoms:
        atom_name = str(atom.name).strip().upper()
        if atom_name and atom_name not in atoms:
            atoms[atom_name] = atom.position

    return atoms


def _generate_type_atom14_mask(res_type: ResidueType) -> list[float]:
    return [1.0 if atom_name else 0.0 for atom_name in res_type.atom14]


def _generate_observed_atom14_mask(
    atoms: dict[str, np.ndarray],
    res_type: ResidueType,
) -> list[float]:
    return [1.0 if atom_name and atom_name in atoms else 0.0 for atom_name in res_type.atom14]


def _generate_atom14_ambiguity_group(res_type: ResidueType) -> list[float]:
    """
    生成 Atom14 对称/歧义组掩码。
    """

    mask = np.zeros(14, dtype=np.float32)

    if not res_type.atom_swap:
        return mask.tolist()

    atom14_idx_map = {name: i for i, name in enumerate(res_type.atom14) if name}
    current_group = 1.0

    for atom_a, atom_b in res_type.atom_swap.items():
        if atom_a in atom14_idx_map and atom_b in atom14_idx_map:
            idx_a = atom14_idx_map[atom_a]
            idx_b = atom14_idx_map[atom_b]
            mask[idx_a] = current_group
            mask[idx_b] = current_group
            current_group += 1.0

    return mask.tolist()


def _generate_type_torsion_mask(res_type: ResidueType) -> list[float]:
    """
    理论上该残基具备哪些 torsion。
    """

    num_chi = len(res_type.chi_angle)
    return [1.0, 1.0, 1.0] + [1.0] * num_chi + [0.0] * (4 - num_chi)


def _compute_dihedral_with_mask(
    atoms: dict[str, np.ndarray],
    atom_names: list[str],
) -> tuple[float, float, float]:
    """
    计算单个二面角的 sin/cos 以及有效性掩码。
    """

    pts = _get_vecs_by_names(atoms, atom_names)

    if pts is None:
        return 0.0, 0.0, 0.0

    val = calculate_dihedral(*pts)
    if val is None:
        return 0.0, 0.0, 0.0

    return float(np.sin(val)), float(np.cos(val)), 1.0


def _compute_residue_geometry(
    res: MDAResidue,
    prev_res: MDAResidue | None,
    next_res: MDAResidue | None,
    res_type: ResidueType,
) -> ResidueGeometryResult:
    """
    计算残基扭转角特征及观测掩码。
    """

    try:
        atoms = _build_atom_lookup(res)
        prev_atoms = _build_atom_lookup(prev_res) if prev_res is not None else {}
        next_atoms = _build_atom_lookup(next_res) if next_res is not None else {}

        torsion_features: list[float] = []
        torsion_valid: list[float] = []

        # Phi
        if prev_res is not None:
            phi_sin, phi_cos, phi_valid = _compute_dihedral_with_mask(
                {**prev_atoms, **{f"curr_{k}": v for k, v in atoms.items()}},
                ["C", "curr_N", "curr_CA", "curr_C"],
            )
        else:
            phi_sin, phi_cos, phi_valid = 0.0, 0.0, 0.0
        torsion_features.extend([phi_sin, phi_cos])
        torsion_valid.append(phi_valid)

        # Psi
        if next_res is not None:
            psi_sin, psi_cos, psi_valid = _compute_dihedral_with_mask(
                {
                    **atoms,
                    **{f"next_{k}": v for k, v in next_atoms.items()},
                },
                ["N", "CA", "C", "next_N"],
            )
        else:
            psi_sin, psi_cos, psi_valid = 0.0, 0.0, 0.0
        torsion_features.extend([psi_sin, psi_cos])
        torsion_valid.append(psi_valid)

        # Omega
        if next_res is not None:
            omega_sin, omega_cos, omega_valid = _compute_dihedral_with_mask(
                {
                    **atoms,
                    **{f"next_{k}": v for k, v in next_atoms.items()},
                },
                ["CA", "C", "next_N", "next_CA"],
            )
        else:
            omega_sin, omega_cos, omega_valid = 0.0, 0.0, 0.0
        torsion_features.extend([omega_sin, omega_cos])
        torsion_valid.append(omega_valid)

        for chi_names in res_type.chi_angle:
            chi_sin, chi_cos, chi_valid = _compute_dihedral_with_mask(
                atoms,
                list(chi_names),
            )
            torsion_features.extend([chi_sin, chi_cos])
            torsion_valid.append(chi_valid)

        while len(torsion_valid) < len(PROTEIN_RESIDUE_TORSION_NAMES):
            torsion_features.extend([0.0, 0.0])
            torsion_valid.append(0.0)

        observed_backbone_mask = [
            1.0 if atom_name in atoms else 0.0
            for atom_name in PROTEIN_RESIDUE_BACKBONE_ATOM_NAMES
        ]

        return {
            "torsion_features": torsion_features,
            "observed_torsion_mask": torsion_valid,
            "observed_backbone_mask": observed_backbone_mask,
        }

    except Exception as exc:
        logger.warning(
            "Failed to compute residue geometry for %s%s: %s",
            res.resname,
            res.resid,
            exc,
            exc_info=True,
        )
        return {
            "torsion_features": [0.0] * (2 * len(PROTEIN_RESIDUE_TORSION_NAMES)),
            "observed_torsion_mask": [0.0] * len(PROTEIN_RESIDUE_TORSION_NAMES),
            "observed_backbone_mask": [0.0] * len(PROTEIN_RESIDUE_BACKBONE_ATOM_NAMES),
        }


def _safe_chain_label(residue: MDAResidue) -> str:
    chain_id = str(getattr(residue, "chainID", "") or "").strip()
    if chain_id:
        return chain_id

    segid = str(getattr(residue, "segid", "") or "").strip()
    if segid:
        return segid

    return "_"


def _safe_icode(residue: MDAResidue) -> int:
    icode = str(getattr(residue, "icode", "") or "").strip()
    return ord(icode[0]) if icode else 0


def _infer_element_symbol(atom) -> str:
    if atom.element:
        symbol = str(atom.element).strip()
        if symbol:
            return symbol

    atom_name = str(getattr(atom, "name", "") or "").strip().upper()
    for char in atom_name:
        if char.isalpha():
            return char
    return ""


def _segment_length_norm(segment_len: int) -> float:
    return float(min(1.0, math.log1p(max(segment_len, 1)) / _SEGMENT_LENGTH_NORM_BASE))


def _is_backbone_atom(atom_name: str) -> bool:
    return atom_name in _BACKBONE_ATOM_SET


def _is_donor_like(atom_name: str, res_type: ResidueType) -> float:
    if atom_name == "N":
        return 0.0 if res_type.name == "PRO" else 1.0
    if atom_name in {"O", "OXT", "CA", "C"}:
        return 0.0
    return 1.0 if atom_name in _SIDECHAIN_DONOR_ATOMS.get(res_type.name, set()) else 0.0


def _is_acceptor_like(atom_name: str, res_type: ResidueType) -> float:
    if atom_name in {"O", "OXT"}:
        return 1.0
    if atom_name in {"N", "CA", "C"}:
        return 0.0
    return 1.0 if atom_name in _SIDECHAIN_ACCEPTOR_ATOMS.get(res_type.name, set()) else 0.0


def _is_aromatic_like(atom_name: str, res_type: ResidueType) -> float:
    return 1.0 if atom_name in _AROMATIC_ATOMS_BY_RESIDUE.get(res_type.name, set()) else 0.0


class ProteinEncoder:
    """
    蛋白质结构编码器。
    """

    def encode(
        self,
        universe: mda.Universe,
        *,
        esm_embeddings: dict[int, np.ndarray] | None = None,
    ) -> ProteinEncodingResult:
        protein_atoms = universe.select_atoms("protein")

        if len(protein_atoms) == 0:
            raise ValueError("Universe contains no protein atoms")

        all_atoms = sorted(protein_atoms.atoms, key=lambda atom: atom.ix)
        all_residues = sorted(list(protein_atoms.residues), key=lambda residue: residue.ix)

        residue_segments = segment_residues_by_continuity(all_residues)
        prev_res_by_ix: dict[int, MDAResidue | None] = {}
        next_res_by_ix: dict[int, MDAResidue | None] = {}
        segment_meta_by_ix: dict[int, dict[str, int | float]] = {}
        chain_label_to_index: dict[str, int] = {}

        for segment_id, segment in enumerate(residue_segments):
            seg_residues = list(segment.residues)
            seg_len = len(seg_residues)
            rel_len = _segment_length_norm(seg_len)

            for seg_pos, seg_res in enumerate(seg_residues):
                res_ix = int(seg_res.ix)
                prev_res_by_ix[res_ix] = seg_residues[seg_pos - 1] if seg_pos > 0 else None
                next_res_by_ix[res_ix] = (
                    seg_residues[seg_pos + 1] if seg_pos + 1 < seg_len else None
                )

                rel_pos = 0.5 if seg_len <= 1 else float(seg_pos / (seg_len - 1))
                centrality = float(2.0 * rel_pos - 1.0)
                chain_label = _safe_chain_label(seg_res)
                chain_index = chain_label_to_index.setdefault(
                    chain_label,
                    len(chain_label_to_index),
                )
                segment_meta_by_ix[res_ix] = {
                    "source_residue_ix": res_ix,
                    "source_resid": int(seg_res.resid),
                    "source_chain_index": chain_index,
                    "source_icode_code": _safe_icode(seg_res),
                    "source_segment_id": segment_id,
                    "source_segment_offset": seg_pos,
                    "source_segment_length": seg_len,
                    "segment_rel_pos": rel_pos,
                    "segment_centrality": centrality,
                    "segment_length_norm": rel_len,
                    "has_prev_contiguous": 1.0 if seg_pos > 0 else 0.0,
                    "has_next_contiguous": 1.0 if seg_pos + 1 < seg_len else 0.0,
                }

        atom_data = {
            feature.name: []
            for feature in PROTEIN_ATOM_CAT_SCHEMA + PROTEIN_ATOM_CONT_SCHEMA
        }
        atom_positions: list[list[float]] = []
        atom_residue_indices: list[int] = []

        res_ix_to_local_idx = {residue.ix: idx for idx, residue in enumerate(all_residues)}
        residue_type_by_ix = {
            int(residue.ix): resolve_protein_residue_type(residue.resname).residue_type
            for residue in all_residues
        }
        atom_lookup_by_ix = {
            int(residue.ix): _build_atom_lookup(residue)
            for residue in all_residues
        }

        for atom in all_atoms:
            atom_name = str(atom.name).strip().upper()
            symbol = _infer_element_symbol(atom)
            elem = Element.safe_get(symbol)
            res_type = residue_type_by_ix.get(int(atom.residue.ix), ResidueType.UNK)

            atom_data["atomic_idx"].append(elem.idx)
            atom_data["atom_name_class"].append(
                PROTEIN_ATOM_NAME_TO_CLASS.get(atom_name, _ATOM_NAME_UNK)
            )

            atom_data["vdw_radius_mm3"].append(elem.vdw_radius_mm3)
            atom_data["atomic_weight"].append(elem.atomic_weight)
            atom_data["en_pauling"].append(elem.en_pauling)
            atom_data["electron_affinity"].append(elem.electron_affinity)
            atom_data["first_ionization_energy"].append(elem.first_ionization_energy)
            atom_data["is_backbone"].append(1.0 if _is_backbone_atom(atom_name) else 0.0)
            atom_data["is_sidechain"].append(0.0 if _is_backbone_atom(atom_name) else 1.0)
            atom_data["is_alpha_carbon"].append(1.0 if atom_name == "CA" else 0.0)
            atom_data["is_donor_like"].append(_is_donor_like(atom_name, res_type))
            atom_data["is_acceptor_like"].append(_is_acceptor_like(atom_name, res_type))
            atom_data["is_aromatic_like"].append(_is_aromatic_like(atom_name, res_type))

            atom_positions.append(atom.position.tolist())
            atom_residue_indices.append(res_ix_to_local_idx.get(atom.residue.ix, 0))

        residue_data = {
            feature.name: []
            for feature in PROTEIN_RESIDUE_CAT_SCHEMA + PROTEIN_RESIDUE_CONT_SCHEMA
        }
        residue_positions: list[list[float]] = []
        residue_esm_feats: list[np.ndarray | None] = []
        residue_metadata = {
            key: []
            for key in (
                "source_residue_ix",
                "source_resid",
                "source_chain_index",
                "source_icode_code",
                "source_segment_id",
                "source_segment_offset",
                "source_segment_length",
            )
        }
        auxiliary = {
            "type_atom14_mask": [],
            "observed_atom14_mask": [],
            "atom14_ambiguity_group": [],
            "type_torsion_mask": [],
            "observed_torsion_mask": [],
            "observed_backbone_mask": [],
            "chi_pi_periodic_mask": [],
        }

        for residue in all_residues:
            res_ix = int(residue.ix)
            res_type = residue_type_by_ix[res_ix]
            seg_meta = segment_meta_by_ix[res_ix]

            residue_data["residue_type"].append(res_type.index)

            prev_res = prev_res_by_ix.get(res_ix)
            next_res = next_res_by_ix.get(res_ix)
            geometry = _compute_residue_geometry(residue, prev_res, next_res, res_type)
            residue_atoms = atom_lookup_by_ix[res_ix]

            feat_idx = 0
            for name in PROTEIN_RESIDUE_TORSION_NAMES:
                residue_data[f"{name}_sin"].append(geometry["torsion_features"][feat_idx])
                residue_data[f"{name}_cos"].append(geometry["torsion_features"][feat_idx + 1])
                feat_idx += 2

            for name, value in zip(
                PROTEIN_RESIDUE_TORSION_NAMES,
                geometry["observed_torsion_mask"],
                strict=True,
            ):
                residue_data[f"{name}_valid"].append(value)

            for atom_name, value in zip(
                PROTEIN_RESIDUE_BACKBONE_ATOM_NAMES,
                geometry["observed_backbone_mask"],
                strict=True,
            ):
                residue_data[f"backbone_{atom_name.lower()}_observed"].append(value)

            residue_data["has_prev_contiguous"].append(float(seg_meta["has_prev_contiguous"]))
            residue_data["has_next_contiguous"].append(float(seg_meta["has_next_contiguous"]))
            residue_data["segment_rel_pos"].append(float(seg_meta["segment_rel_pos"]))
            residue_data["segment_centrality"].append(float(seg_meta["segment_centrality"]))
            residue_data["segment_length_norm"].append(float(seg_meta["segment_length_norm"]))

            residue_esm_feats.append(
                esm_embeddings[residue.ix]
                if esm_embeddings is not None and residue.ix in esm_embeddings
                else None
            )

            ca_atom = next((atom for atom in residue.atoms if atom.name == "CA"), None)
            if ca_atom is not None:
                residue_positions.append(ca_atom.position.tolist())
            else:
                residue_positions.append(residue.atoms.center_of_geometry().tolist())

            for key in residue_metadata:
                residue_metadata[key].append(int(seg_meta[key]))

            auxiliary["type_atom14_mask"].append(_generate_type_atom14_mask(res_type))
            auxiliary["observed_atom14_mask"].append(
                _generate_observed_atom14_mask(residue_atoms, res_type)
            )
            auxiliary["atom14_ambiguity_group"].append(
                _generate_atom14_ambiguity_group(res_type)
            )
            auxiliary["type_torsion_mask"].append(_generate_type_torsion_mask(res_type))
            auxiliary["observed_torsion_mask"].append(geometry["observed_torsion_mask"])
            auxiliary["observed_backbone_mask"].append(geometry["observed_backbone_mask"])
            auxiliary["chi_pi_periodic_mask"].append(list(res_type.chi_pi_periodic))

        return {
            "atom_features": atom_data,
            "atom_positions": atom_positions,
            "atom_to_residue_index": atom_residue_indices,
            "residue_features": residue_data,
            "residue_positions": residue_positions,
            "residue_esm_embeddings": residue_esm_feats,
            "residue_metadata": residue_metadata,
            "auxiliary": auxiliary,
        }

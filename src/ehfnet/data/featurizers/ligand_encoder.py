"""
配体编码工具。

负责从 RDKit 配体结构提取原子、键和分子级特征，
生成构图流程所需的输入表示。
"""


import logging
from collections.abc import Callable
from typing import TypedDict

import numpy as np

from rdkit import Chem
from rdkit.Chem import (
    Crippen,
    Descriptors,
    QED,
    rdFreeSASA,
    rdMolDescriptors,
    rdPartialCharges,
    rdmolops,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeatureFactory

from ehfnet.data.featurizers.chemistry import Element
from ehfnet.data.featurizers.feature_specs import (
    LIGAND_ATOM_CAT_SCHEMA,
    LIGAND_ATOM_CONT_SCHEMA,
    LIGAND_MOLECULE_CONT_SCHEMA,
)
from ehfnet.geometry import get_moving_atoms

logger = logging.getLogger(__name__)

_LIGAND_CAT_MAX: dict[str, int] = {f.name: f.num_embeddings - 1 for f in LIGAND_ATOM_CAT_SCHEMA}
_calc_mol_wt: Callable[[Chem.Mol], float] = getattr(Descriptors, "MolWt")
_calc_tpsa: Callable[[Chem.Mol], float] = getattr(Descriptors, "TPSA")
_calc_logp: Callable[[Chem.Mol], float] = getattr(Crippen, "MolLogP")
_calc_molar_refractivity: Callable[[Chem.Mol], float] = getattr(Crippen, "MolMR")


class LigandEncodingResult(TypedDict):
    """
    配体编码结果。

    封装配体原子、键和分子级特征的编码输出，
    便于构图流程按统一字段读取配体表示。
    """

    atom_features: dict[str, list[int | float]]
    mol_features: dict[str, float]
    positions: list[list[float]]
    torsion_indices: list[list[int]]
    torsion_masks: list[list[bool]]


ROTATABLE_PATTERN_NON_STRICT = "[!$(*#*)&!D1]-,:;!@[!$(*#*)&!D1]"
ROTATABLE_PATTERN_STRICT = (
    "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])"
    "&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])"
    "-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-,:;!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)"
    "&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]"
)


def _map_formal_charge(charge: int) -> int:
    offset = _LIGAND_CAT_MAX["formal_charge"] // 2
    clamped = max(-offset, min(offset, charge))
    return clamped + offset


def _map_hybridization(hyb_type: Chem.rdchem.HybridizationType) -> int:
    mapping = {
        Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
        Chem.rdchem.HybridizationType.S: 0,
        Chem.rdchem.HybridizationType.SP: 1,
        Chem.rdchem.HybridizationType.SP2: 2,
        Chem.rdchem.HybridizationType.SP3: 3,
        Chem.rdchem.HybridizationType.SP3D: 4,
        Chem.rdchem.HybridizationType.SP3D2: 5,
    }
    return mapping.get(hyb_type, 8)


def _compute_chiral_info(mol: Chem.Mol) -> tuple[set[int], np.ndarray]:
    try:
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        chiral_indices = {
            idx
            for idx, _ in chiral_centers
            if mol.GetAtomWithIdx(idx).GetSymbol() == "C"
        }

        if not chiral_indices:
            return set(), np.full(mol.GetNumAtoms(), -1, dtype=int)

        dist_mat = rdmolops.GetDistanceMatrix(mol)
        chiral_cols = list(chiral_indices)
        min_dist = dist_mat[:, chiral_cols].min(axis=1).astype(int)
        return chiral_indices, min_dist
    except Exception as exc:
        logger.warning(f"Chiral info computation failed: {exc}", exc_info=True)
        return set(), np.full(mol.GetNumAtoms(), -1, dtype=int)


def _compute_gasteiger_charges(mol_hs: Chem.Mol) -> list[float]:
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol_hs)
        charges: list[float] = []
        for atom in mol_hs.GetAtoms():
            val = atom.GetDoubleProp("_GasteigerCharge")
            charges.append(0.0 if np.isnan(val) else val)
        return charges
    except Exception as exc:
        logger.warning(f"Gasteiger charge computation failed: {exc}", exc_info=True)
        return [0.0] * mol_hs.GetNumAtoms()


def _compute_sasa(mol_hs: Chem.Mol) -> list[float]:
    try:
        radii = rdFreeSASA.classifyAtoms(mol_hs)
        rdFreeSASA.CalcSASA(mol_hs, radii)
        return [a.GetDoubleProp("SASA") for a in mol_hs.GetAtoms()]
    except Exception as exc:
        logger.warning(f"SASA computation failed: {exc}", exc_info=True)
        return [0.0] * mol_hs.GetNumAtoms()


def _compute_topological_distances(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray]:
    try:
        dist_mat = rdmolops.GetDistanceMatrix(mol)
        return dist_mat.max(axis=1), dist_mat.mean(axis=1)
    except Exception as exc:
        logger.warning(f"Topological distance computation failed: {exc}", exc_info=True)
        n = mol.GetNumAtoms()
        return np.zeros(n), np.zeros(n)


def _compute_pharmacophore_mask(
    mol: Chem.Mol,
    factory: MolChemicalFeatureFactory,
) -> dict[str, set[int]]:
    feats = factory.GetFeaturesForMol(mol)
    mask = {
        "Acceptor": set(),
        "Donor": set(),
        "Hydrophobe": set(),
        "Positive": set(),
        "Negative": set(),
    }

    for feat in feats:
        fam = feat.GetFamily()
        if fam in mask:
            for idx in feat.GetAtomIds():
                mask[fam].add(idx)

    return mask


def _get_dihedral_indices(
    mol: Chem.Mol,
    u: int,
    v: int,
    *,
    canonical_ranks: list[int],
) -> list[int]:
    def get_neighbor(idx: int, exclude: int) -> int | None:
        neighbors = [
            n.GetIdx()
            for n in mol.GetAtomWithIdx(idx).GetNeighbors()
            if n.GetIdx() != exclude
        ]
        if not neighbors:
            return None

        def score(neighbor_idx: int) -> tuple[int, int, int, int, int]:
            atom = mol.GetAtomWithIdx(neighbor_idx)
            bond = mol.GetBondBetweenAtoms(idx, neighbor_idx)
            bond_order = int(round(10 * bond.GetBondTypeAsDouble())) if bond is not None else 0
            return (
                0 if atom.GetAtomicNum() > 1 else 1,
                -int(atom.GetIsAromatic()),
                -bond_order,
                canonical_ranks[neighbor_idx],
                neighbor_idx,
            )

        return min(neighbors, key=score)

    p0 = get_neighbor(u, v)
    p3 = get_neighbor(v, u)
    if p0 is None or p3 is None:
        return []
    return [p0, u, v, p3]


def _extract_torsion_info(
    mol: Chem.Mol,
    *,
    strict: bool = True,
) -> tuple[list[list[int]], list[list[bool]]]:
    pattern_str = ROTATABLE_PATTERN_STRICT if strict else ROTATABLE_PATTERN_NON_STRICT
    rot_pattern = Chem.MolFromSmarts(pattern_str)
    if rot_pattern is None:
        logger.error("Failed to create query molecule from SMARTS pattern")
        return [], []

    matches = mol.GetSubstructMatches(rot_pattern)
    torsion_indices: list[list[int]] = []
    moving_masks: list[list[bool]] = []
    num_atoms = mol.GetNumAtoms()
    seen_bonds = set()
    canonical_ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))

    for match in matches:
        u, v = match[0], match[1]
        bond = mol.GetBondBetweenAtoms(u, v)
        if not bond:
            continue

        bid = bond.GetIdx()
        if bid in seen_bonds or bond.IsInRing():
            continue
        seen_bonds.add(bid)

        moving_atoms, axis_fix, axis_rot = get_moving_atoms(
            mol,
            bid,
            canonical_ranks=canonical_ranks,
        )
        if not moving_atoms:
            continue

        dihedral = _get_dihedral_indices(
            mol,
            axis_fix,
            axis_rot,
            canonical_ranks=canonical_ranks,
        )
        if not dihedral:
            continue

        mask = np.zeros(num_atoms, dtype=bool)
        mask[moving_atoms] = True
        torsion_indices.append(dihedral)
        moving_masks.append(mask.tolist())

    if torsion_indices:
        sorted_indices = sorted(
            range(len(moving_masks)),
            key=lambda i: (
                -sum(moving_masks[i]),
                torsion_indices[i][1],
                torsion_indices[i][2],
                torsion_indices[i][0],
                torsion_indices[i][3],
            ),
        )
        torsion_indices = [torsion_indices[i] for i in sorted_indices]
        moving_masks = [moving_masks[i] for i in sorted_indices]

    return torsion_indices, moving_masks


class LigandEncoder:
    """
    配体编码器。

    负责从 RDKit 分子对象提取原子与分子级特征，
    并生成后续构图阶段所需的配体输入表示。
    """

    def __init__(self, feature_factory: MolChemicalFeatureFactory):
        """
        初始化配体编码器。

        保存特征工厂和编码所需的静态配置，
        为后续配体特征提取做好准备。

        Args:
            feature_factory: 负责提取化学特征的 RDKit 特征工厂。
        """
        self._feature_factory = feature_factory

    def encode(
        self,
        mol: Chem.Mol,
        *,
        strict_torsion: bool = True,
    ) -> LigandEncodingResult:
        """
        编码单个配体分子。

        从 RDKit 分子对象提取原子、键和分子级特征，
        并整理为构图阶段使用的统一结果结构。

        Args:
            mol: 待读取或处理的 RDKit 分子对象。
            strict_torsion: 是否严格限制可旋转键与扭转规则。

        Returns:
            LigandEncodingResult: 返回配体原子、键、扭转和分子级特征组成的编码结果。

        Raises:
            ValueError: 当输入参数或运行时状态不满足要求时抛出。
        """
        if not mol:
            raise ValueError("Molecule is None")
        if mol.GetNumAtoms() == 0:
            raise ValueError("Molecule has no atoms")
        if mol.GetNumConformers() == 0:
            raise ValueError("Molecule has no conformers")

        mol_hs = Chem.AddHs(mol, addCoords=True)
        conf = mol.GetConformer()

        chiral_indices, dist_to_chiral = _compute_chiral_info(mol)
        gasteiger = _compute_gasteiger_charges(mol_hs)
        sasa = _compute_sasa(mol_hs)
        max_topo, mean_topo = _compute_topological_distances(mol)
        pharma_mask = _compute_pharmacophore_mask(mol, self._feature_factory)
        murcko = MurckoScaffold.GetScaffoldForMol(mol)
        murcko_indices = set(mol.GetSubstructMatch(murcko)) if murcko else set()
        ring_info = mol.GetRingInfo()

        atom_data = {f.name: [] for f in LIGAND_ATOM_CAT_SCHEMA + LIGAND_ATOM_CONT_SCHEMA}
        positions: list[list[float]] = []

        for i, atom in enumerate(mol.GetAtoms()):
            atom_data["num_hs"].append(min(_LIGAND_CAT_MAX["num_hs"], atom.GetTotalNumHs()))
            atom_data["degree"].append(min(_LIGAND_CAT_MAX["degree"], atom.GetDegree()))
            atom_data["implicit_valence"].append(
                min(
                    _LIGAND_CAT_MAX["implicit_valence"],
                    atom.GetValence(Chem.ValenceType.IMPLICIT),
                )
            )

            elem = Element.safe_get(atom.GetAtomicNum())
            atom_data["atomic_idx"].append(elem.idx)
            atom_data["formal_charge"].append(_map_formal_charge(atom.GetFormalCharge()))
            atom_data["hybridization"].append(_map_hybridization(atom.GetHybridization()))
            atom_data["num_radical_electrons"].append(
                min(_LIGAND_CAT_MAX["num_radical_electrons"], atom.GetNumRadicalElectrons())
            )
            atom_data["chirality"].append(int(atom.GetChiralTag()))
            atom_data["is_chiral_center"].append(1 if i in chiral_indices else 0)
            atom_data["distance_to_nearest_chiral"].append(
                0
                if int(dist_to_chiral[i]) < 0
                else min(_LIGAND_CAT_MAX["distance_to_nearest_chiral"], int(dist_to_chiral[i]) + 1)
            )
            atom_data["is_aromatic"].append(1 if atom.GetIsAromatic() else 0)
            atom_data["is_in_ring"].append(1 if atom.IsInRing() else 0)
            atom_data["smallest_ring_size"].append(
                0
                if not atom.IsInRing()
                else min(_LIGAND_CAT_MAX["smallest_ring_size"], max(3, ring_info.MinAtomRingSize(i)))
            )
            atom_data["is_in_murcko_scaffold"].append(1 if i in murcko_indices else 0)

            bonds = atom.GetBonds()
            b_types = [b.GetBondType() for b in bonds]
            atom_data["num_single_bonds"].append(
                min(_LIGAND_CAT_MAX["num_single_bonds"], b_types.count(Chem.BondType.SINGLE))
            )
            atom_data["num_double_bonds"].append(
                min(_LIGAND_CAT_MAX["num_double_bonds"], b_types.count(Chem.BondType.DOUBLE))
            )
            atom_data["num_triple_bonds"].append(
                min(_LIGAND_CAT_MAX["num_triple_bonds"], b_types.count(Chem.BondType.TRIPLE))
            )
            atom_data["num_aromatic_bonds"].append(
                min(_LIGAND_CAT_MAX["num_aromatic_bonds"], b_types.count(Chem.BondType.AROMATIC))
            )
            atom_data["is_acceptor"].append(1 if i in pharma_mask["Acceptor"] else 0)
            atom_data["is_donor"].append(1 if i in pharma_mask["Donor"] else 0)
            atom_data["is_hydrophobe"].append(1 if i in pharma_mask["Hydrophobe"] else 0)
            atom_data["is_positive"].append(1 if i in pharma_mask["Positive"] else 0)
            atom_data["is_negative"].append(1 if i in pharma_mask["Negative"] else 0)

            atom_data["vdw_radius_mm3"].append(elem.vdw_radius_mm3)
            atom_data["atomic_weight"].append(elem.atomic_weight)
            atom_data["en_pauling"].append(elem.en_pauling)
            atom_data["electron_affinity"].append(elem.electron_affinity)
            atom_data["first_ionization_energy"].append(elem.first_ionization_energy)
            atom_data["gasteiger_charge"].append(gasteiger[i])
            atom_data["sasa_contribution"].append(sasa[i])
            atom_data["max_topological_distance"].append(max_topo[i])
            atom_data["mean_topological_distance"].append(mean_topo[i])

            pos = conf.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])

        mol_data = {f.name: 0.0 for f in LIGAND_MOLECULE_CONT_SCHEMA}
        mol_data["wt"] = _calc_mol_wt(mol)
        mol_data["tpsa"] = _calc_tpsa(mol)
        mol_data["logp"] = _calc_logp(mol)
        mol_data["qed"] = QED.qed(mol)
        mol_data["molar_refractivity"] = _calc_molar_refractivity(mol)
        mol_data["hallKier_alpha"] = rdMolDescriptors.CalcHallKierAlpha(mol)
        mol_data["kappa2"] = rdMolDescriptors.CalcKappa2(mol)
        mol_data["chi1v"] = rdMolDescriptors.CalcChi1v(mol)
        mol_data["chi4v"] = rdMolDescriptors.CalcChi4v(mol)

        t_indices, t_masks = _extract_torsion_info(mol, strict=strict_torsion)

        return {
            "atom_features": atom_data,
            "mol_features": mol_data,
            "positions": positions,
            "torsion_indices": t_indices,
            "torsion_masks": t_masks,
        }

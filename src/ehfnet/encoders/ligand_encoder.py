"""
配体分子编码器

提供配体分子的特征提取和编码功能
"""

import logging
import numpy as np

from collections.abc import Callable
from typing import TypedDict
from rdkit import Chem
from rdkit.Chem import (
    rdMolDescriptors,
    rdmolops,
    rdPartialCharges,
    rdFreeSASA,
    QED,
    Crippen,
    Descriptors,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeatureFactory

from ehfnet.encoders.chemistry import Element
from ehfnet.encoders.feature_specs import (
    LIGAND_ATOM_CAT_SCHEMA,
    LIGAND_ATOM_CONT_SCHEMA,
    LIGAND_MOLECULE_CONT_SCHEMA,
)
from ehfnet.geometry.static import get_moving_atoms

logger = logging.getLogger(__name__)

_LIGAND_CAT_MAX: dict[str, int] = {f.name: f.num_embeddings - 1 for f in LIGAND_ATOM_CAT_SCHEMA}

# 为缺失类型提示的 RDKit 函数创建类型安全的包装器
_calc_mol_wt: Callable[[Chem.Mol], float] = getattr(Descriptors, "MolWt")
_calc_tpsa: Callable[[Chem.Mol], float] = getattr(Descriptors, "TPSA")
_calc_logp: Callable[[Chem.Mol], float] = getattr(Crippen, "MolLogP")
_calc_molar_refractivity: Callable[[Chem.Mol], float] = getattr(Crippen, "MolMR")


class LigandEncodingResult(TypedDict):
    """
    配体编码结果
    """

    atom_features: dict[str, list[int | float]]
    mol_features: dict[str, float]
    positions: list[list[float]]
    torsion_indices: list[list[int]]
    torsion_masks: list[list[bool]]


# 可旋转键 SMARTS 模式（宽松模式）
ROTATABLE_PATTERN_NON_STRICT = "[!$(*#*)&!D1]-,:;!@[!$(*#*)&!D1]"

# 可旋转键 SMARTS 模式（严格模式：排除甲基、叔丁基、酰胺键等）
ROTATABLE_PATTERN_STRICT = (
    "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])"
    "&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])"
    "-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-,:;!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)"
    "&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]"
)


def _map_formal_charge(charge: int) -> int:
    """
    映射形式电荷 [-4, 4] -> [0, 8]
    """

    offset = _LIGAND_CAT_MAX["formal_charge"] // 2
    clamped = max(
        -offset,
        min(offset, charge),
    )

    return clamped + offset


def _map_hybridization(hyb_type: Chem.rdchem.HybridizationType) -> int:
    """
    映射杂化类型 -> [0, 8]
    """

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
    """
    计算手性中心信息

    Args:
        mol: RDKit 分子

    Returns:
        (手性中心索引集合, 到最近手性中心的距离数组)
    """

    try:
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        # 只保留碳原子的手性中心
        chiral_indices = {
            idx
            for idx, _ in chiral_centers
            if mol.GetAtomWithIdx(idx).GetSymbol() == "C"
        }

        if not chiral_indices:
            return set(), np.full(
                mol.GetNumAtoms(), -1, dtype=int
            )

        dist_mat = rdmolops.GetDistanceMatrix(mol)
        chiral_cols = list(chiral_indices)
        min_dist = dist_mat[:, chiral_cols].min(axis=1).astype(int)

        return chiral_indices, min_dist

    except Exception as e:
        logger.warning(f"Chiral info computation failed: {e}", exc_info=True)
        return set(), np.full(mol.GetNumAtoms(), -1, dtype=int)


def _compute_gasteiger_charges(mol_hs: Chem.Mol) -> list[float]:
    """
    计算 Gasteiger 偏电荷

    Args:
        mol_hs: 含氢分子

    Returns:
        每个原子的偏电荷列表，失败时返回零向量
    """

    try:
        rdPartialCharges.ComputeGasteigerCharges(mol_hs)
        charges: list[float] = []

        for atom in mol_hs.GetAtoms():
            val = atom.GetDoubleProp("_GasteigerCharge")
            charges.append(0.0 if np.isnan(val) else val)

        return charges

    except Exception as e:
        logger.warning(f"Gasteiger charge computation failed: {e}", exc_info=True)
        return [0.0] * mol_hs.GetNumAtoms()


def _compute_sasa(mol_hs: Chem.Mol) -> list[float]:
    """
    计算溶剂可及表面积

    Args:
        mol_hs: 含氢分子

    Returns:
        每个原子的 SASA 贡献列表，失败时返回零向量
    """

    try:
        radii = rdFreeSASA.classifyAtoms(mol_hs)
        rdFreeSASA.CalcSASA(mol_hs, radii)
        return [a.GetDoubleProp("SASA") for a in mol_hs.GetAtoms()]

    except Exception as e:
        logger.warning(f"SASA computation failed: {e}", exc_info=True)
        return [0.0] * mol_hs.GetNumAtoms()


def _compute_topological_distances(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray]:
    """
    计算拓扑距离统计

    Args:
        mol: RDKit 分子

    Returns:
        (最大拓扑距离数组, 平均拓扑距离数组)
    """

    try:
        dist_mat = rdmolops.GetDistanceMatrix(mol)
        return dist_mat.max(axis=1), dist_mat.mean(axis=1)

    except Exception as e:
        logger.warning(f"Topological distance computation failed: {e}", exc_info=True)
        n = mol.GetNumAtoms()
        return np.zeros(n), np.zeros(n)


def _compute_pharmacophore_mask(
    mol: Chem.Mol, factory: MolChemicalFeatureFactory
) -> dict[str, set[int]]:
    """
    计算药效团特征掩码

    Args:
        mol: RDKit 分子
        factory: 药效团特征工厂

    Returns:
        药效团类型到原子索引集合的映射
    """

    feats = factory.GetFeaturesForMol(mol)
    mask = {
        "Acceptor": set(),
        "Donor": set(),
        "Hydrophobe": set(),
        "Positive": set(),
        "Negative": set(),
    }

    for f in feats:
        fam = f.GetFamily()

        if fam in mask:
            
            for idx in f.GetAtomIds():
                mask[fam].add(idx)

    return mask


def _get_dihedral_indices(
    mol: Chem.Mol,
    u: int,
    v: int,
    *,
    canonical_ranks: list[int],
) -> list[int]:
    """
    寻找定义二面角的四个原子 [p0, u, v, p3]

    Args:
        mol: RDKit 分子
        u: 旋转轴上的第一个原子索引
        v: 旋转轴上的第二个原子索引

    Returns:
        四个原子索引列表，如果无法定义则返回空列表
    """

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
    mol: Chem.Mol, *, strict: bool = True
) -> tuple[list[list[int]], list[list[bool]]]:
    """
    提取扭转信息

    Args:
        mol: RDKit 分子
        strict: 是否使用严格模式过滤可旋转键

    Returns:
        (扭转角索引列表, 移动原子掩码列表)
    """

    # 选择 SMARTS 模式
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
        # SMARTS 匹配的两个原子即为键的两端
        u, v = match[0], match[1]
        bond = mol.GetBondBetweenAtoms(u, v)

        if not bond:
            continue

        bid = bond.GetIdx()

        if bid in seen_bonds:
            continue

        seen_bonds.add(bid)

        # 双重检查：确保不是环上的键
        if bond.IsInRing():
            continue

        # 寻找移动片段
        moving_atoms, axis_fix, axis_rot = get_moving_atoms(
            mol,
            bid,
            canonical_ranks=canonical_ranks,
        )

        if not moving_atoms:
            continue

        # 寻找二面角锚点
        dihedral = _get_dihedral_indices(
            mol,
            axis_fix,
            axis_rot,
            canonical_ranks=canonical_ranks,
        )
        if not dihedral:
            continue

        # 构建掩码
        mask = np.zeros(num_atoms, dtype=bool)
        mask[moving_atoms] = True

        torsion_indices.append(dihedral)
        moving_masks.append(mask.tolist())

    # 拓扑排序：从根到叶（移动原子多 → 移动原子少）
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
    配体分子编码器

    将 RDKit 分子对象编码为神经网络可用的特征表示
    """

    def __init__(self, feature_factory: MolChemicalFeatureFactory):
        """
        初始化编码器

        Args:
            feature_factory: RDKit 药效团特征工厂
        """

        self._feature_factory = feature_factory


    def encode(
        self, mol: Chem.Mol, *, strict_torsion: bool = True
    ) -> LigandEncodingResult:
        """
        编码配体分子

        Args:
            mol: RDKit 分子对象（必须包含构象）
            strict_torsion: 是否使用严格模式过滤可旋转键

        Returns:
            编码结果字典

        Raises:
            ValueError: 分子无效或缺少构象
        """

        # 验证输入
        if not mol:
            raise ValueError("Molecule is None")

        if mol.GetNumAtoms() == 0:
            raise ValueError("Molecule has no atoms")
            
        if mol.GetNumConformers() == 0:
            raise ValueError("Molecule has no conformers")

        # 预计算辅助数据
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

        # 计算原子特征
        atom_data = {
            f.name: [] for f in LIGAND_ATOM_CAT_SCHEMA + LIGAND_ATOM_CONT_SCHEMA
        }
        positions: list[list[float]] = []

        for i, atom in enumerate(mol.GetAtoms()):
            # 分类特征
            atom_data["num_hs"].append(
                min(_LIGAND_CAT_MAX["num_hs"], atom.GetTotalNumHs())
            )
            atom_data["degree"].append(min(_LIGAND_CAT_MAX["degree"], atom.GetDegree()))
            atom_data["implicit_valence"].append(
                min(
                    _LIGAND_CAT_MAX["implicit_valence"],
                    atom.GetValence(Chem.ValenceType.IMPLICIT),
                )
            )

            elem = Element.safe_get(atom.GetAtomicNum())
            atom_data["atomic_idx"].append(elem.idx)

            atom_data["formal_charge"].append(
                _map_formal_charge(atom.GetFormalCharge())
            )
            atom_data["hybridization"].append(_map_hybridization(atom.GetHybridization()))

            atom_data["num_radical_electrons"].append(
                min(_LIGAND_CAT_MAX["num_radical_electrons"], atom.GetNumRadicalElectrons())
            )
            atom_data["chirality"].append(int(atom.GetChiralTag()))
            atom_data["is_chiral_center"].append(1 if i in chiral_indices else 0)
            atom_data["distance_to_nearest_chiral"].append(
                0
                if int(dist_to_chiral[i]) < 0
                else min(
                    _LIGAND_CAT_MAX["distance_to_nearest_chiral"],
                    int(dist_to_chiral[i]) + 1,
                )
            )

            atom_data["is_aromatic"].append(1 if atom.GetIsAromatic() else 0)
            atom_data["is_in_ring"].append(1 if atom.IsInRing() else 0)
            atom_data["smallest_ring_size"].append(
                0
                if not atom.IsInRing()
                else min(
                    _LIGAND_CAT_MAX["smallest_ring_size"],
                    max(3, ring_info.MinAtomRingSize(i)),
                )
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

            # 连续特征
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

        # 计算分子特征
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

        # 计算扭转信息
        t_indices, t_masks = _extract_torsion_info(mol, strict=strict_torsion)

        return {
            "atom_features": atom_data,
            "mol_features": mol_data,
            "positions": positions,
            "torsion_indices": t_indices,
            "torsion_masks": t_masks,
        }

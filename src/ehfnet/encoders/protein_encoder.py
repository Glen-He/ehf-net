"""
蛋白质结构编码器

提供蛋白质结构的特征提取和编码功能
"""

import logging
import numpy as np
import MDAnalysis as mda

from pathlib import Path
from typing import TypedDict
from MDAnalysis.core.groups import Residue as MDAResidue

from ehfnet.encoders.chemistry import Element, ResidueType
from ehfnet.encoders.feature_specs import (
    PROTEIN_ATOM_CAT_SCHEMA,
    PROTEIN_ATOM_CONT_SCHEMA,
    PROTEIN_RESIDUE_CAT_SCHEMA,
    PROTEIN_RESIDUE_CONT_SCHEMA,
)
from ehfnet.geometry.static import calculate_dihedral

logger = logging.getLogger(__name__)

_PROTEIN_RES_CAT_MAX: dict[str, int] = {f.name: f.num_embeddings - 1 for f in PROTEIN_RESIDUE_CAT_SCHEMA}


def _get_vecs_by_names(
    source_atoms: dict[str, np.ndarray], names: list[str]
) -> list[np.ndarray] | None:
    """
    根据名字取坐标，缺一个就返回 None
    """

    vecs: list[np.ndarray] = []

    for name in names:

        if name not in source_atoms:
            return None

        vecs.append(source_atoms[name])

    return vecs


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

    auxiliary: dict[str, list[list[float]]]


def _generate_atom14_mask(res_type: ResidueType) -> list[float]:
    """
    生成 Atom14 存在性掩码
    """

    return [1.0 if atom else 0.0 for atom in res_type.atom14]


def _generate_atom14_symmetry_mask(res_type: ResidueType) -> list[float]:
    """
    生成 Atom14 对称性掩码

    Args:
        res_type: 残基类型

    Returns:
        长度为 14 的掩码列表，对称原子对标记相同的非零值
    """

    mask = np.zeros(14, dtype=np.float32)

    if not res_type.atom_swap:
        return mask.tolist()

    atom14_idx_map = {name: i for i, name in enumerate(res_type.atom14) if name}
    current_sym = 1.0

    for a1, a2 in res_type.atom_swap.items():

        if a1 in atom14_idx_map and a2 in atom14_idx_map:
            idx1, idx2 = atom14_idx_map[a1], atom14_idx_map[a2]
            mask[idx1] = current_sym
            mask[idx2] = current_sym
            current_sym += 1.0

    return mask.tolist()


def _generate_torsion_mask(res_type: ResidueType) -> list[float]:
    """
    生成扭转角掩码

    Args:
        res_type: 残基类型

    Returns:
        长度为 7 的掩码列表 [phi, psi, omega, chi1, chi2, chi3, chi4]
    """

    mask = [1.0, 1.0, 1.0]  # phi, psi, omega
    num_chi = len(res_type.chi_angle)
    mask.extend([1.0] * num_chi + [0.0] * (4 - num_chi))
    return mask


def _compute_residue_torsions(
    res: MDAResidue,
    prev_res: MDAResidue | None,
    next_res: MDAResidue | None,
    res_type: ResidueType,
) -> list[float]:
    """
    计算残基扭转角（Phi, Psi, Omega, Chi1-4）的 sin/cos 值

    Args:
        res: 当前残基
        prev_res: 前一个残基（用于计算 Phi）
        next_res: 后一个残基（用于计算 Psi, Omega）
        res_type: 残基类型

    Returns:
        长度为 14 的列表: [phi_sin, phi_cos, psi_sin, psi_cos, omega_sin, omega_cos,
                          chi1_sin, chi1_cos, chi2_sin, chi2_cos, chi3_sin, chi3_cos, chi4_sin, chi4_cos]
    """
    try:
        # 缓存原子坐标 {name: pos}
        atoms = {a.name: a.position for a in res.atoms}
        prev_atoms = {a.name: a.position for a in prev_res.atoms} if prev_res else {}
        next_atoms = {a.name: a.position for a in next_res.atoms} if next_res else {}

        angles: list[float] = []  # 存储 7 个角度值（弧度）

        # 1. Phi (C_prev - N - CA - C)
        if prev_res:
            pts = _get_vecs_by_names(prev_atoms, ["C"])
            pts_curr = _get_vecs_by_names(atoms, ["N", "CA", "C"])

            if pts and pts_curr:
                full_pts = pts + pts_curr
                val = calculate_dihedral(*full_pts)
                angles.append(val if val is not None else 0.0)

            else:
                angles.append(0.0)

        else:
            angles.append(0.0)

        # 2. Psi (N - CA - C - N_next)
        if next_res:
            pts = _get_vecs_by_names(atoms, ["N", "CA", "C"])
            pts_next = _get_vecs_by_names(next_atoms, ["N"])

            if pts and pts_next:
                full_pts = pts + pts_next
                val = calculate_dihedral(*full_pts)
                angles.append(val if val is not None else 0.0)

            else:
                angles.append(0.0)
        else:
            angles.append(0.0)

        # 3. Omega (CA - C - N_next - CA_next)
        if next_res:
            pts = _get_vecs_by_names(atoms, ["CA", "C"])
            pts_next = _get_vecs_by_names(next_atoms, ["N", "CA"])

            if pts and pts_next:
                full_pts = pts + pts_next
                val = calculate_dihedral(*full_pts)
                angles.append(val if val is not None else 0.0)

            else:
                angles.append(0.0)

        else:
            angles.append(0.0)

        # 4. Chi 1-4 (侧链扭转角)
        for i in range(4):

            if i < len(res_type.chi_angle):
                atom_names = list(res_type.chi_angle[i])
                pts = _get_vecs_by_names(atoms, atom_names)

                if pts:
                    val = calculate_dihedral(*pts)
                    angles.append(val if val is not None else 0.0)

                else:
                    angles.append(0.0)

            else:
                angles.append(0.0)

        # sin/cos 变换
        feats: list[float] = []

        for ang in angles:
            feats.append(np.sin(ang))
            feats.append(np.cos(ang))

        return feats  # [14]

    except Exception as e:
        logger.warning(
            f"Failed to compute torsions for residue {res.resname}{res.resid}: {e}",
            exc_info=True,
        )
        return [0.0] * 14  # 异常时返回零向量


class ProteinEncoder:
    """
    蛋白质结构编码器

    将 MDAnalysis Universe 编码为神经网络可用的特征表示
    """

    def __init__(self):
        """
        初始化编码器
        """

        pass

    def encode(
        self,
        universe: mda.Universe,
        *,
        esm_embeddings: dict[int, np.ndarray] | None = None,
        esm_embedding_file: str | Path | None = None,
        pocket_radius: float | None = None,
        ligand_positions: np.ndarray | None = None,
    ) -> ProteinEncodingResult:
        """
        编码蛋白质结构

        Args:
            universe: MDAnalysis Universe 对象
            esm_embeddings: 预计算的 ESM 嵌入字典 {residue_index: embedding}
            esm_embedding_file: ESM 嵌入文件路径（.npz 格式）
            pocket_radius: 口袋提取半径 (Å)。如果提供，则仅保留该半径内的残基。
            ligand_positions: 配体原子坐标 [L, 3]，用于确定口袋中心。

        Returns:
            编码结果字典
        """

        protein_atoms = universe.select_atoms("protein")

        if len(protein_atoms) == 0:
            raise ValueError("Universe contains no protein atoms")

        # 口袋提取逻辑
        if pocket_radius is not None and ligand_positions is not None:
            # 手动筛选残基，以保证 Residue-level 的完整性
            dist_sq = np.min(
                np.sum((protein_atoms.positions[:, None, :] - np.array(ligand_positions)[None, :, :])**2, axis=-1),
                axis=1
            )
            mask = dist_sq <= (pocket_radius**2)
            
            # 获取满足条件的原子所属的所有残基
            valid_residues = protein_atoms[mask].residues
            
            if len(valid_residues) == 0:
                raise ValueError(
                    f"No valid residues found within {pocket_radius}A of the ligand! "
                    "This complex's structural coordinates are likely severely corrupted (massive drift). "
                    "Refusing to fall back to the full protein."
                )

            else:
                protein_atoms = valid_residues.atoms
                logger.info(f"Extracted pocket with {len(valid_residues)} residues within {pocket_radius}A.")

        all_atoms = sorted(protein_atoms.atoms, key=lambda a: a.ix)
        all_residues = sorted(list(protein_atoms.residues), key=lambda r: r.ix)

        # 建立残基查找表 (segid, resid) -> Residue
        res_lookup = {(r.segid, r.resid): r for r in all_residues}

        # ESM Embedding 加载
        if esm_embeddings is None and esm_embedding_file is not None:

            try:
                esm_path = Path(esm_embedding_file)

                if esm_path.exists():

                    with np.load(esm_path) as data:
                        esm_embeddings = {int(k): data[k].copy() for k in data.files}
                    logger.info(
                        f"Loaded ESM embeddings from {esm_path} ({len(esm_embeddings)} residues)"
                    )

                else:
                    logger.warning(f"ESM embedding file not found: {esm_path}")
                    esm_embeddings = None

            except Exception as e:
                logger.error(f"Failed to load ESM embeddings from {esm_embedding_file}: {e}")
                esm_embeddings = None

        # 原子特征
        atom_data = {
            f.name: [] for f in PROTEIN_ATOM_CAT_SCHEMA + PROTEIN_ATOM_CONT_SCHEMA
        }
        atom_positions: list[list[float]] = []
        atom_residue_indices: list[int] = []

        res_ix_to_local_idx = {r.ix: i for i, r in enumerate(all_residues)}

        for atom in all_atoms:
            if atom.element:
                symbol = atom.element.upper()
            else:
                atom_name = str(getattr(atom, "name", "") or "").strip().upper()
                symbol = "SE" if atom_name == "SE" else (atom_name[0].upper() if atom_name else "")
            elem = Element.safe_get(symbol)

            # 分类特征
            atom_data["atomic_idx"].append(elem.idx)

            # 连续特征
            atom_data["vdw_radius_mm3"].append(elem.vdw_radius_mm3)
            atom_data["atomic_weight"].append(elem.atomic_weight)
            atom_data["en_pauling"].append(elem.en_pauling)
            atom_data["electron_affinity"].append(elem.electron_affinity)
            atom_data["first_ionization_energy"].append(elem.first_ionization_energy)

            # 位置 & 映射
            atom_positions.append(atom.position.tolist())
            atom_residue_indices.append(res_ix_to_local_idx.get(atom.residue.ix, 0))

        # 残基特征
        res_data = {
            f.name: [] for f in PROTEIN_RESIDUE_CAT_SCHEMA + PROTEIN_RESIDUE_CONT_SCHEMA
        }
        res_positions: list[list[float]] = []
        res_esm_feats: list[np.ndarray | None] = []

        aux_data = {
            "atom14_mask": [],
            "atom14_symmetry_mask": [],
            "torsion_angle_mask": [],
            "chi_pi_periodic_mask": [],
        }

        for idx, res in enumerate(all_residues):
            res_type = ResidueType.safe_get(res.resname)

            # 分类特征
            res_data["residue_type"].append(res_type.index)
            res_data["residue_id"].append(min(_PROTEIN_RES_CAT_MAX["residue_id"], idx))

            # 连续特征（扭转角）
            prev_res = res_lookup.get((res.segid, res.resid - 1))
            next_res = res_lookup.get((res.segid, res.resid + 1))

            torsion_feats = _compute_residue_torsions(res, prev_res, next_res, res_type)

            # 按 Schema 填充 sin/cos
            angle_names = ["phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4"]
            feat_idx = 0

            for name in angle_names:
                res_data[f"{name}_sin"].append(torsion_feats[feat_idx])
                res_data[f"{name}_cos"].append(torsion_feats[feat_idx + 1])
                feat_idx += 2

            # ESM Embedding
            if esm_embeddings and res.ix in esm_embeddings:
                res_esm_feats.append(esm_embeddings[res.ix])

            else:
                res_esm_feats.append(None)

            # 位置（CA 优先）
            ca = next((a for a in res.atoms if a.name == "CA"), None)

            if ca:
                res_positions.append(ca.position.tolist())
                
            else:
                res_positions.append(res.atoms.center_of_geometry().tolist())

            # 辅助掩码
            aux_data["atom14_mask"].append(_generate_atom14_mask(res_type))
            aux_data["atom14_symmetry_mask"].append(
                _generate_atom14_symmetry_mask(res_type)
            )
            aux_data["torsion_angle_mask"].append(_generate_torsion_mask(res_type))
            aux_data["chi_pi_periodic_mask"].append(list(res_type.chi_pi_periodic))

        return {
            "atom_features": atom_data,
            "atom_positions": atom_positions,
            "atom_to_residue_index": atom_residue_indices,
            "residue_features": res_data,
            "residue_positions": res_positions,
            "residue_esm_embeddings": res_esm_feats,
            "auxiliary": aux_data,
        }

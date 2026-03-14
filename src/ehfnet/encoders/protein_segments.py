"""
蛋白质连续链段工具

统一定义基于真实肽链连续性的 residue segmentation。
该分段同时服务于：
1. ESM sequence 构建
2. backbone torsion 邻接查找
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from MDAnalysis.core.groups import Residue as MDAResidue


PEPTIDE_CONTINUITY_CN_DISTANCE = 2.2


@dataclass(frozen=True)
class ProteinResidueSegment:
    """
    单条连续肽链段。
    """

    key: str
    residues: tuple[MDAResidue, ...]

    @property
    def residue_ixs(self) -> tuple[int, ...]:
        return tuple(int(res.ix) for res in self.residues)


def _residue_chain_tags(res: MDAResidue) -> tuple[str, str]:
    """
    提取残基所属链标签。

    优先使用明确的链标识；若缺失，则退回空字符串，后续由几何连续性决定分段。
    """

    segid = str(getattr(res, "segid", "") or "").strip()
    chain_id = ""

    if len(res.atoms) > 0:
        first_atom = res.atoms[0]
        chain_id = str(getattr(first_atom, "chainID", "") or "").strip()

        if not chain_id:
            atom_chain_ids = getattr(res.atoms, "chainIDs", None)
            if atom_chain_ids is not None and len(atom_chain_ids) > 0:
                chain_id = str(atom_chain_ids[0] or "").strip()

    return segid, chain_id


def _get_atom_position(res: MDAResidue, atom_name: str) -> np.ndarray | None:
    for atom in res.atoms:
        if atom.name == atom_name:
            return np.asarray(atom.position, dtype=np.float32)
    return None


def continuity_break_reason(
    prev_res: MDAResidue,
    next_res: MDAResidue,
    *,
    max_cn_distance: float = PEPTIDE_CONTINUITY_CN_DISTANCE,
) -> str | None:
    """
    给出两相邻 residue 不能视为同一条连续肽链的原因。
    """

    prev_segid, prev_chain = _residue_chain_tags(prev_res)
    next_segid, next_chain = _residue_chain_tags(next_res)

    if prev_chain and next_chain and prev_chain != next_chain:
        return f"chain mismatch: {prev_chain} != {next_chain}"

    if prev_segid and next_segid and prev_segid != next_segid:
        return f"segid mismatch: {prev_segid} != {next_segid}"

    c_prev = _get_atom_position(prev_res, "C")
    if c_prev is None:
        return "previous residue missing backbone atom C"

    n_next = _get_atom_position(next_res, "N")
    if n_next is None:
        return "next residue missing backbone atom N"

    cn_distance = float(np.linalg.norm(c_prev - n_next))
    if cn_distance > max_cn_distance:
        return f"C-N distance {cn_distance:.3f}A exceeds {max_cn_distance:.3f}A"

    return None


def is_peptide_continuous(
    prev_res: MDAResidue,
    next_res: MDAResidue,
    *,
    max_cn_distance: float = PEPTIDE_CONTINUITY_CN_DISTANCE,
) -> bool:
    """
    判断两个相邻 residue 是否属于同一条连续肽链。

    判据：
    1. 若存在明确链标签，链标签必须一致
    2. prev.C 与 next.N 必须存在
    3. ||C_prev - N_next|| 必须小于阈值
    """

    return (
        continuity_break_reason(
            prev_res,
            next_res,
            max_cn_distance=max_cn_distance,
        )
        is None
    )


def segment_residues_by_continuity(
    residues: list[MDAResidue],
    *,
    max_cn_distance: float = PEPTIDE_CONTINUITY_CN_DISTANCE,
) -> list[ProteinResidueSegment]:
    """
    按真实肽链连续性切分 residue 序列。

    输入 residues 会先按 `residue.ix` 排序，以保证与 MDAnalysis 内部顺序一致。
    """

    if not residues:
        return []

    sorted_residues = sorted(residues, key=lambda r: int(r.ix))
    segments: list[ProteinResidueSegment] = []
    current: list[MDAResidue] = [sorted_residues[0]]

    for res in sorted_residues[1:]:
        prev = current[-1]

        if is_peptide_continuous(prev, res, max_cn_distance=max_cn_distance):
            current.append(res)
            continue

        segid, chain_id = _residue_chain_tags(current[0])
        seg_idx = len(segments)
        key = f"{segid or 'seg'}:{chain_id or 'chain'}:{seg_idx}"
        segments.append(ProteinResidueSegment(key=key, residues=tuple(current)))
        current = [res]

    segid, chain_id = _residue_chain_tags(current[0])
    seg_idx = len(segments)
    key = f"{segid or 'seg'}:{chain_id or 'chain'}:{seg_idx}"
    segments.append(ProteinResidueSegment(key=key, residues=tuple(current)))

    return segments

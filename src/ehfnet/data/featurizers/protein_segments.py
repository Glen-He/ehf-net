"""
蛋白分段工具。

负责按残基连续性切分蛋白链段，
为 ESM 序列提取和上下文处理提供基础片段信息。
"""


from dataclasses import dataclass

import numpy as np

from MDAnalysis.core.groups import Residue as MDAResidue


PEPTIDE_CONTINUITY_CN_DISTANCE = 2.2


@dataclass(frozen=True)
class ProteinResidueSegment:
    """
    连续蛋白链段。

    表示一段按真实连续性切分出的残基序列，
    供 ESM 提取和蛋白上下文构建流程使用。
    """

    key: str
    residues: tuple[MDAResidue, ...]

    @property
    def residue_ixs(self) -> tuple[int, ...]:
        return tuple(int(res.ix) for res in self.residues)


def _residue_chain_tags(res: MDAResidue) -> tuple[str, str]:
    """
    提取残基所属链标签。

    Returns:
        tuple[str, str]: 返回当前残基的 `segid` 与 `chain_id` 组合。
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
    分析连续性中断原因。

    给出两个相邻残基不能视为同一连续肽链的具体原因，
    用于调试链段切分和序列提取逻辑。

    Args:
        prev_res: 前一个残基对象。
        next_res: 后一个残基对象。
        max_cn_distance: 判断肽键连续性时允许的最大 C-N 距离。

    Returns:
        str | None: 返回导致链连续性中断的原因；若两残基连续则返回 `None`。
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
    判断残基是否连续。

    基于主链连接关系和残基顺序判断两个相邻残基是否属于同一肽链，
    是链段切分逻辑的基础判定函数。

    Args:
        prev_res: 前一个残基对象。
        next_res: 后一个残基对象。
        max_cn_distance: 判断肽键连续性时允许的最大 C-N 距离。

    Returns:
        bool: 返回布尔判断结果。
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
    按连续性切分残基序列。

    将整条残基序列拆分为多个真实连续的链段，
    供序列提取、ESM 计算和上下文建模复用。

    Args:
        residues: 待切分或处理的残基序列。
        max_cn_distance: 判断肽键连续性时允许的最大 C-N 距离。

    Returns:
        list[ProteinResidueSegment]: 返回按真实连续性切分后的蛋白链段列表。
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

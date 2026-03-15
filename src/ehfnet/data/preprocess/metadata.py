"""
预处理元数据工具。

负责提取、规范化和校验预处理附加信息，
供缓存兼容性判断和摘要写入流程使用。
"""


from typing import Any

from torch_geometric.data import HeteroData


def normalize_ligand_sanitize_mode(mode: Any) -> str:
    """
    规范化配体清洗模式。

    将不同来源的 sanitize 模式值整理成统一表示，
    便于元数据落盘和缓存兼容性判断。

    Args:
        mode: 待规范化的模式名称。

    Returns:
        str: 返回可稳定写入元数据的规范化清洗模式名称。
    """

    value = str(mode).strip().lower() if mode is not None else "unknown"
    return (
        value
        if value in {"full", "partial", "rejected", "unknown"}
        else "unknown"
    )


def extract_ligand_sanitize_metadata(data: HeteroData) -> dict[str, Any]:
    """
    提取配体清洗元数据。

    从图对象或预处理结果中整理配体清洗相关信息，
    供摘要写入和缓存检查流程使用。

    Args:
        data: 当前处理的图数据对象。

    Returns:
        dict[str, Any]: 返回配体清洗模式及相关标志位组成的元数据字典。
    """

    mode = normalize_ligand_sanitize_mode(
        getattr(data, "ligand_sanitize_mode", "unknown")
    )
    return {
        "ligand_sanitize_mode": mode,
        "ligand_partial_sanitize": bool(
            getattr(data, "ligand_partial_sanitize", mode == "partial")
        ),
        "ligand_full_sanitize_flag": int(
            getattr(data, "ligand_full_sanitize_flag", -1)
        ),
        "ligand_partial_sanitize_flag": int(
            getattr(data, "ligand_partial_sanitize_flag", -1)
        ),
    }

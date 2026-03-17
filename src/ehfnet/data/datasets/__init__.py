"""
数据集子模块入口。

导出数据集构造、配体处理和数据划分工具，
统一上层调用这些数据组件的导入方式。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ehfnet.data.datasets.ligand_sanitize import (
        LigandSanitizationError,
        load_ligand_mol,
        read_ligand_file,
        sanitize_ligand_mol_with_mode,
    )
    from ehfnet.data.datasets.pose_initialization import generate_decoupled_ligand_positions
    from ehfnet.data.datasets.protein_ligand import ProteinLigandDataset
    from ehfnet.data.datasets.splitter import ScaffoldSplitter


__all__ = [
    "LigandSanitizationError",
    "ProteinLigandDataset",
    "ScaffoldSplitter",
    "generate_decoupled_ligand_positions",
    "load_ligand_mol",
    "read_ligand_file",
    "sanitize_ligand_mol_with_mode",
]


def __getattr__(name: str):
    """
    按名称返回公开对象。

    仅在首次访问时执行真实导入，
    用于避免包初始化阶段触发重模块加载或循环依赖。

    Args:
        name: 请求访问或解析的公开对象名称。

    Returns:
        object: 返回与名称对应的惰性导出对象。

    Raises:
        AttributeError: 当访问的属性不存在或对象不满足接口约定时抛出。
    """

    if name in {
        "LigandSanitizationError",
        "load_ligand_mol",
        "read_ligand_file",
        "sanitize_ligand_mol_with_mode",
    }:
        from ehfnet.data.datasets.ligand_sanitize import (
            LigandSanitizationError,
            load_ligand_mol,
            read_ligand_file,
            sanitize_ligand_mol_with_mode,
        )

        exports = {
            "LigandSanitizationError": LigandSanitizationError,
            "load_ligand_mol": load_ligand_mol,
            "read_ligand_file": read_ligand_file,
            "sanitize_ligand_mol_with_mode": sanitize_ligand_mol_with_mode,
        }
        return exports[name]
    if name == "generate_decoupled_ligand_positions":
        from ehfnet.data.datasets.pose_initialization import (
            generate_decoupled_ligand_positions,
        )

        return generate_decoupled_ligand_positions
    if name == "ProteinLigandDataset":
        from ehfnet.data.datasets.protein_ligand import ProteinLigandDataset

        return ProteinLigandDataset
    if name == "ScaffoldSplitter":
        from ehfnet.data.datasets.splitter import ScaffoldSplitter

        return ScaffoldSplitter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""
图神经网络边类型定义

定义了蛋白质-配体复合物在不同层级（原子、残基、分子/口袋）之间的信息传递路径。采用分层交互设计以优化显存占用并提升物理可解释性。
"""

NODE_TYPES: list[str] = [
    "ligand_atom",
    "protein_atom",
    "ligand_molecule",
    "protein_residue",
    "protein_pocket",
]

ATOM_NODE_TYPES: list[str] = ["ligand_atom", "protein_atom"]


# 阶段 1: 层级内信息更新（intra-level）
INTRA_EDGES: list[tuple[str, str, str]] = [
    ("ligand_atom", "intra_proximity", "ligand_atom"),
    ("protein_atom", "intra_proximity", "protein_atom"),
    ("protein_residue", "intra_proximity", "protein_residue"),
]


# 阶段 2: 自底向上聚合（bottom-up aggregation）
AGGREGATE_EDGES: list[tuple[str, str, str]] = [
    ("ligand_atom", "aggregate_to", "ligand_molecule"),
    ("protein_atom", "aggregate_to", "protein_residue"),
    ("protein_residue", "aggregate_to", "protein_pocket"),
]


# 阶段 3: 跨图交互（inter-graph interaction / docking）
#
# 设计约定：
# - DYNAMIC_INTER_EDGES: 随 pose / protein 坐标变化、由 encoder 在每个 block 动态重建
# - STATIC_INTER_EDGES: 几何关系近似全局稳定、在 graph builder / runtime crop 中构建
#
# 这样可以避免 builder 和 encoder 对同一类跨图边重复负责，保持语义清晰。

DYNAMIC_INTER_EDGES: list[tuple[str, str, str]] = [
    # 1. 核心物理交互：配体原子 <-> 蛋白原子
    ("ligand_atom", "inter_proximity", "protein_atom"),
    ("protein_atom", "inter_proximity", "ligand_atom"),

    # 2. 多尺度交互：配体原子 <-> 蛋白残基
    ("ligand_atom", "inter_proximity", "protein_residue"),
    ("protein_residue", "inter_proximity", "ligand_atom"),
]


STATIC_INTER_EDGES: list[tuple[str, str, str]] = [
    # 3. 全局上下文定位：配体原子 <-> 蛋白口袋
    ("ligand_atom", "inter_proximity", "protein_pocket"),
    ("protein_pocket", "inter_proximity", "ligand_atom"),

    # 4. 宏观形状匹配：配体分子 <-> 蛋白口袋
    ("ligand_molecule", "inter_proximity", "protein_pocket"),
    ("protein_pocket", "inter_proximity", "ligand_molecule"),
]


INTER_EDGES: list[tuple[str, str, str]] = DYNAMIC_INTER_EDGES + STATIC_INTER_EDGES


# 阶段 4: 自顶向下广播（top-down broadcast）
BROADCAST_EDGES: list[tuple[str, str, str]] = [
    ("ligand_molecule", "broadcast_to", "ligand_atom"),
    ("protein_pocket", "broadcast_to", "protein_residue"),
    ("protein_residue", "broadcast_to", "protein_atom"),
]

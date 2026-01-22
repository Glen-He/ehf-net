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


# 阶段 1: 层级内信息更新 (Intra-level)
INTRA_EDGES: list[tuple[str, str, str]] = [
    ("ligand_atom", "intra_proximity", "ligand_atom"),
    ("protein_atom", "intra_proximity", "protein_atom"),
    ("protein_residue", "intra_proximity", "protein_residue"),
]


# 阶段 2: 自底向上聚合 (Bottom-up Aggregation)
AGGREGATE_EDGES: list[tuple[str, str, str]] = [
    ("ligand_atom", "aggregate_to", "ligand_molecule"),
    ("protein_atom", "aggregate_to", "protein_residue"),
    ("protein_residue", "aggregate_to", "protein_pocket"),
]


# 阶段 3: 跨图交互 (Inter-graph Interaction / Docking)
INTER_EDGES: list[tuple[str, str, str]] = [

    # 1. 核心物理交互：原子-原子 (基于半径图)
    ("ligand_atom", "inter_proximity", "protein_atom"),
    ("protein_atom", "inter_proximity", "ligand_atom"),

    # 2. 多尺度融合：配体原子-蛋白残基 (直接利用 ESM 进化信息)
    ("ligand_atom", "inter_proximity", "protein_residue"),
    ("protein_residue", "inter_proximity", "ligand_atom"),

    # 3. 全局上下文定位：配体原子-蛋白口袋 (低成本全局坐标感)
    ("ligand_atom", "inter_proximity", "protein_pocket"),
    ("protein_pocket", "inter_proximity", "ligand_atom"),

    # 4. 宏观形状匹配：配体分子-蛋白口袋 (整体极性/形状互补)
    ("ligand_molecule", "inter_proximity", "protein_pocket"),
    ("protein_pocket", "inter_proximity", "ligand_molecule"),
]


# 阶段 4: 自顶向下广播 (Top-down Broadcast)
BROADCAST_EDGES: list[tuple[str, str, str]] = [
    ("ligand_molecule", "broadcast_to", "ligand_atom"),
    ("protein_pocket", "broadcast_to", "protein_residue"),
    ("protein_residue", "broadcast_to", "protein_atom"),
]


# 所有边的集合
ALL_EDGES: list[tuple[str, str, str]] = INTRA_EDGES + AGGREGATE_EDGES + INTER_EDGES + BROADCAST_EDGES

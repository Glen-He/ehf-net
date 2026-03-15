"""
图 schema 定义。

集中声明节点类型、边类型、特征维度和命名约定，
保证构图接口与模型输入的结构一致。
"""


NODE_TYPES: list[str] = [
    "ligand_atom",
    "protein_atom",
    "ligand_molecule",
    "protein_residue",
    "protein_context",
]

ATOM_NODE_TYPES: list[str] = ["ligand_atom", "protein_atom"]

INTRA_EDGES: list[tuple[str, str, str]] = [
    ("ligand_atom", "intra_proximity", "ligand_atom"),
    ("protein_atom", "intra_proximity", "protein_atom"),
    ("protein_residue", "intra_proximity", "protein_residue"),
]

AGGREGATE_EDGES: list[tuple[str, str, str]] = [
    ("ligand_atom", "aggregate_to", "ligand_molecule"),
    ("protein_atom", "aggregate_to", "protein_residue"),
    ("protein_residue", "aggregate_to", "protein_context"),
]

DYNAMIC_INTER_EDGES: list[tuple[str, str, str]] = [
    ("ligand_atom", "inter_proximity", "protein_atom"),
    ("protein_atom", "inter_proximity", "ligand_atom"),
    ("ligand_atom", "inter_proximity", "protein_residue"),
    ("protein_residue", "inter_proximity", "ligand_atom"),
]

STATIC_INTER_EDGES: list[tuple[str, str, str]] = [
    ("ligand_atom", "inter_proximity", "protein_context"),
    ("protein_context", "inter_proximity", "ligand_atom"),
    ("ligand_molecule", "inter_proximity", "protein_context"),
    ("protein_context", "inter_proximity", "ligand_molecule"),
]

INTER_EDGES: list[tuple[str, str, str]] = DYNAMIC_INTER_EDGES + STATIC_INTER_EDGES

BROADCAST_EDGES: list[tuple[str, str, str]] = [
    ("ligand_molecule", "broadcast_to", "ligand_atom"),
    ("protein_context", "broadcast_to", "protein_residue"),
    ("protein_residue", "broadcast_to", "protein_atom"),
]

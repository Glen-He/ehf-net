"""
图成本画像工具。

负责统计异构图的节点、边和扭转等基础规模信息，
为训练阶段的成本估计、批处理调度和显存控制提供统一输入。
"""


from typing import Any, cast

from torch_geometric.data import HeteroData


def build_graph_cost_profile(data: HeteroData) -> dict[str, int]:
    """
    构建单个图样本的成本画像。

    Args:
        data: 当前处理的异构图对象。

    Returns:
        dict[str, int]: 包含节点数、各类边数和扭转约束数量的画像字典。
    """
    node_counts = {
        "ligand_atom_nodes": int(getattr(data["ligand_atom"], "num_nodes", 0)),
        "protein_atom_nodes": int(getattr(data["protein_atom"], "num_nodes", 0)),
        "ligand_molecule_nodes": int(getattr(data["ligand_molecule"], "num_nodes", 0)),
        "protein_residue_nodes": int(getattr(data["protein_residue"], "num_nodes", 0)),
        "protein_context_nodes": int(getattr(data["protein_context"], "num_nodes", 0)),
    }
    edge_buckets = {
        "intra_edges": 0,
        "aggregate_edges": 0,
        "broadcast_edges": 0,
        "static_inter_edges": 0,
        "dynamic_inter_edges": 0,
        "total_edges": 0,
    }

    edge_types = getattr(data, "edge_types", None)
    if edge_types:
        for src, rel, dst in cast(list[tuple[str, str, str]], edge_types):
            edge_index = getattr(data[src, rel, dst], "edge_index", None)
            edge_count = (
                int(edge_index.size(1))
                if edge_index is not None and edge_index.ndim == 2
                else 0
            )
            edge_buckets["total_edges"] += edge_count
            if rel == "intra_proximity":
                edge_buckets["intra_edges"] += edge_count
            elif rel == "aggregate_to":
                edge_buckets["aggregate_edges"] += edge_count
            elif rel == "broadcast_to":
                edge_buckets["broadcast_edges"] += edge_count
            elif rel == "inter_proximity":
                if "context" in src or "context" in dst or "molecule" in src or "molecule" in dst:
                    edge_buckets["static_inter_edges"] += edge_count
                else:
                    edge_buckets["dynamic_inter_edges"] += edge_count

    torsion_indices = getattr(data, "torsion_indices", None)
    torsion_count = (
        int(torsion_indices.size(0))
        if torsion_indices is not None and getattr(torsion_indices, "ndim", 0) == 2
        else 0
    )
    total_nodes = sum(node_counts.values())

    return {
        **node_counts,
        **edge_buckets,
        "torsion_count": torsion_count,
        "total_nodes": total_nodes,
    }


def estimate_dynamic_edge_upper_bounds(
    profile: dict[str, Any],
    *,
    num_gnn_blocks: int,
    dynamic_inter_max_neighbors: int,
    dynamic_residue_max_neighbors: int,
    dynamic_residue_candidate_topk: int,
) -> dict[str, int]:
    """
    根据静态画像估计动态图边的上界规模。

    Args:
        profile: 样本级图成本画像。
        num_gnn_blocks: 动态跨图边在每个 block 中重建的次数。
        dynamic_inter_max_neighbors: 动态原子跨图边的单源邻居上限。
        dynamic_residue_max_neighbors: 动态配体-残基跨图边的单源邻居上限。
        dynamic_residue_candidate_topk: 每个复合物为残基跨图边保留的候选残基上限。

    Returns:
        dict[str, int]: 估计得到的动态原子边、动态残基边及其总量。
    """
    ligand_atoms = int(profile.get("ligand_atom_nodes", 0))
    protein_atoms = int(profile.get("protein_atom_nodes", 0))
    protein_residues = int(profile.get("protein_residue_nodes", 0))
    residue_candidates = (
        protein_residues
        if dynamic_residue_candidate_topk <= 0
        else min(protein_residues, int(dynamic_residue_candidate_topk))
    )
    inter_atom_per_block = min(
        ligand_atoms * max(1, int(dynamic_inter_max_neighbors)),
        ligand_atoms * protein_atoms,
    )
    residue_per_block = min(
        ligand_atoms * max(1, int(dynamic_residue_max_neighbors)),
        ligand_atoms * residue_candidates,
    )
    dynamic_inter_atom_edges = int(num_gnn_blocks) * inter_atom_per_block
    dynamic_residue_edges = int(num_gnn_blocks) * residue_per_block
    return {
        "dynamic_inter_atom_edges": dynamic_inter_atom_edges,
        "dynamic_residue_edges": dynamic_residue_edges,
        "dynamic_total_edges": dynamic_inter_atom_edges + dynamic_residue_edges,
    }


def estimate_graph_cost_units(
    profile: dict[str, Any],
    *,
    num_gnn_blocks: int,
    dynamic_inter_max_neighbors: int,
    dynamic_residue_max_neighbors: int,
    dynamic_residue_candidate_topk: int,
    node_weight: float = 8.0,
    edge_weight: float = 1.0,
    torsion_weight: float = 16.0,
    phase_multiplier: float = 1.0,
) -> int:
    """
    估计图样本在指定阶段的成本单位。

    Args:
        profile: 样本级图成本画像。
        num_gnn_blocks: 动态跨图边重建次数。
        dynamic_inter_max_neighbors: 动态原子跨图边的单源邻居上限。
        dynamic_residue_max_neighbors: 动态配体-残基跨图边的单源邻居上限。
        dynamic_residue_candidate_topk: 每个复合物保留的残基候选上限。
        node_weight: 节点项在成本估计中的权重。
        edge_weight: 边项在成本估计中的权重。
        torsion_weight: 扭转项在成本估计中的权重。
        phase_multiplier: 训练、验证或测试阶段的额外倍率。

    Returns:
        int: 估计后的总成本单位。
    """
    dynamic_cost = estimate_dynamic_edge_upper_bounds(
        profile,
        num_gnn_blocks=num_gnn_blocks,
        dynamic_inter_max_neighbors=dynamic_inter_max_neighbors,
        dynamic_residue_max_neighbors=dynamic_residue_max_neighbors,
        dynamic_residue_candidate_topk=dynamic_residue_candidate_topk,
    )
    static_edge_cost = float(profile.get("total_edges", 0))
    total_nodes = float(profile.get("total_nodes", 0))
    torsion_count = float(profile.get("torsion_count", 0))
    total_cost = (
        edge_weight * (static_edge_cost + float(dynamic_cost["dynamic_total_edges"]))
        + node_weight * total_nodes
        + torsion_weight * torsion_count
    )
    return max(1, int(total_cost * phase_multiplier))

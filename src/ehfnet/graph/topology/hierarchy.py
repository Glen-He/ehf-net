"""
层级拓扑工具。

负责构建不同层级节点之间的聚合边与广播边，
连接原子、残基和上下文层的信息流。
"""


import torch

from torch_geometric.data import HeteroData


def build_aggregate_edges(
    data: HeteroData,
    *,
    aggregate_edges: list[tuple[str, str, str]],
) -> HeteroData:
    """
    构建层级聚合边。

    连接局部节点到高层节点的信息汇聚路径，
    用于原子到残基、局部到上下文等聚合关系建模。

    Args:
        data: 当前处理的图数据对象。
        aggregate_edges: 层级聚合边配置。

    Returns:
        HeteroData: 原地修改并返回的图对象。
    """

    for src, rel, dst in aggregate_edges:
        if not hasattr(data[src], "pos") or data[src].pos.numel() == 0:
            continue

        device = data[src].pos.device
        n_src = int(data[src].pos.size(0))

        if src == "protein_atom" and dst == "protein_residue":
            atom_to_res = data["protein_atom"].residue_idx
            edge_index = torch.stack(
                [torch.arange(len(atom_to_res), device=atom_to_res.device), atom_to_res],
                dim=0,
            )
        else:
            edge_index = torch.stack(
                [
                    torch.arange(n_src, device=device),
                    torch.zeros(n_src, dtype=torch.long, device=device),
                ],
                dim=0,
            )

        data[src, rel, dst].edge_index = edge_index

    return data


def build_broadcast_edges(
    data: HeteroData,
    *,
    broadcast_edges: list[tuple[str, str, str]],
) -> HeteroData:
    """
    构建层级广播边。

    连接高层节点回到局部节点的信息分发路径，
    用于上下文语义回流到低层节点的消息传递。

    Args:
        data: 当前处理的图数据对象。
        broadcast_edges: 层级广播边配置。

    Returns:
        HeteroData: 原地修改并返回的图对象。
    """

    for src, rel, dst in broadcast_edges:
        if not hasattr(data[dst], "pos") or data[dst].pos.numel() == 0:
            continue

        device = data[dst].pos.device
        n_dst = int(data[dst].pos.size(0))

        if src == "protein_residue" and dst == "protein_atom":
            residue_idx = data["protein_atom"].residue_idx.long()
            edge_index = torch.stack(
                [
                    residue_idx,
                    torch.arange(n_dst, dtype=torch.long, device=device),
                ],
                dim=0,
            )
        else:
            edge_index = torch.stack(
                [
                    torch.zeros(n_dst, dtype=torch.long, device=device),
                    torch.arange(n_dst, dtype=torch.long, device=device),
                ],
                dim=0,
            )
        data[src, rel, dst].edge_index = edge_index

    return data

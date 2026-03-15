"""
跨图拓扑工具。

负责构建蛋白与配体之间的静态交互边，
提供跨图消息传递的基础拓扑。
"""


from collections.abc import Callable

import torch
from torch_geometric.data import HeteroData


def build_static_inter_edges(
    data: HeteroData,
    *,
    static_inter_edges: list[tuple[str, str, str]],
    is_edge_enabled: Callable[..., bool],
) -> HeteroData:
    """
    构建静态跨图交互边。

    按预定义规则连接蛋白与配体节点，
    为跨图消息传递提供基础拓扑。

    Args:
        data: 当前处理的图数据对象。
        static_inter_edges: 静态跨图边配置。
        is_edge_enabled: 用于判断某类边是否启用的回调函数。

    Returns:
        dict[tuple[str, str, str], Tensor]: 静态跨图交互边字典。
    """

    inter_edge_types = set(static_inter_edges)
    processed_edges: set[tuple[str, str, str]] = set()

    for src, rel, dst in static_inter_edges:
        if not is_edge_enabled(src, dst):
            continue

        sorted_nodes = sorted([src, dst])
        edge_key = (rel, sorted_nodes[0], sorted_nodes[1])
        if edge_key in processed_edges:
            continue

        if hasattr(data["ligand_atom"], "pos"):
            edge_device = data["ligand_atom"].pos.device
        else:
            edge_device = torch.device("cpu")

        n_src_nodes = int(data[src].num_nodes)
        n_dst_nodes = int(data[dst].num_nodes)
        src_idx = torch.arange(n_src_nodes, device=edge_device).repeat_interleave(
            n_dst_nodes
        )
        dst_idx = torch.arange(n_dst_nodes, device=edge_device).repeat(n_src_nodes)
        edge_index = torch.stack([src_idx, dst_idx], dim=0)

        data[src, rel, dst].edge_index = edge_index
        reverse_edge_type = (dst, rel, src)
        if reverse_edge_type in inter_edge_types:
            data[dst, rel, src].edge_index = (
                edge_index.flip(0) if edge_index.numel() > 0 else edge_index
            )

        processed_edges.add(edge_key)

    return data

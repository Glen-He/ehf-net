"""
图内拓扑工具。

负责构建配体、蛋白原子和残基层级的图内边，
作为局部消息传递的结构骨架。
"""


import torch

from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import knn_graph, radius_graph


def build_same_type_radius_or_knn_edges(
    pos: Tensor,
    *,
    radius_cutoff: float,
    max_num_neighbors: int,
    knn_fallback_k: int,
) -> Tensor:
    """
    构建同类型节点局部边。

    优先保留半径图中的几何邻域结构，
    并对遗漏目标节点补充少量 kNN 边以减少孤点和断链。

    Args:
        pos: 节点坐标张量。
        radius_cutoff: 半径使用的截断阈值。
        max_num_neighbors: 单个节点允许保留的最大邻居数。
        knn_fallback_k: 回退到 kNN 时使用的邻居数。

    Returns:
        Tensor: 同类型节点之间的局部边索引。
    """

    num_nodes = int(pos.size(0))
    device = pos.device

    if num_nodes <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    edge_index = radius_graph(
        pos,
        r=float(radius_cutoff),
        loop=False,
        max_num_neighbors=max(1, int(max_num_neighbors)),
    )

    covered_dst = (
        torch.unique(edge_index[1])
        if edge_index.numel() > 0
        else torch.zeros((0,), dtype=torch.long, device=device)
    )
    all_nodes = torch.arange(num_nodes, device=device)
    uncovered_dst = all_nodes[~torch.isin(all_nodes, covered_dst)]

    if uncovered_dst.numel() > 0:
        knn_edges = knn_graph(
            pos,
            k=min(max(1, int(knn_fallback_k)), num_nodes - 1),
            loop=False,
        )
        if knn_edges.numel() > 0:
            uncovered_mask = torch.isin(knn_edges[1], uncovered_dst)
            extra_edges = knn_edges[:, uncovered_mask]
            if extra_edges.numel() > 0:
                edge_index = (
                    torch.cat([edge_index, extra_edges], dim=1)
                    if edge_index.numel() > 0
                    else extra_edges
                )

    if edge_index.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    return torch.unique(edge_index, dim=1)


def build_intra_edges(
    data: HeteroData,
    *,
    intra_edges: list[tuple[str, str, str]],
    intra_edge_cfg: dict[str, dict[str, float | int]],
) -> HeteroData:
    """
    构建图内边关系。

    按节点类型和局部邻域规则生成图内连接，
    作为模型在各层级内传播信息的基本拓扑。

    Args:
        data: 当前处理的图数据对象。
        intra_edges: 图内边类型集合。
        intra_edge_cfg: 图内边构建所需的邻域配置。

    Returns:
        dict[tuple[str, str, str], Tensor]: 图内边字典。
    """

    for src, rel, dst in intra_edges:
        pos = data[src].pos
        cfg = intra_edge_cfg[src]
        edge_index = build_same_type_radius_or_knn_edges(
            pos,
            radius_cutoff=float(cfg["radius"]),
            max_num_neighbors=int(cfg["max_neighbors"]),
            knn_fallback_k=int(cfg["fallback_k"]),
        )
        data[src, rel, dst].edge_index = edge_index

    return data

"""
跨图边构建工具。

负责按半径建立动态交互边，
并在缺边场景下回退到 kNN 连接策略。
"""


import torch

from torch import Tensor
from torch_geometric.nn import radius


def build_batched_bipartite_knn_edges(
    *,
    src_pos: Tensor,
    src_batch: Tensor,
    dst_pos: Tensor,
    dst_batch: Tensor,
    k: int,
    src_indices: Tensor | None = None,
) -> Tensor:
    """
    构建批内双向 kNN 边。

    在同一 batch 内为源节点和目标节点建立最近邻连接，
    用于半径边缺失时补充跨类型邻接关系。

    Args:
        src_pos: 源节点坐标。
        src_batch: 源节点所属 batch 索引。
        dst_pos: 目标节点坐标。
        dst_batch: 目标节点所属 batch 索引。
        k: kNN 连接时保留的邻居数。
        src_indices: 源节点在原始集合中的索引。

    Returns:
        Tensor: 形状为 `[2, E]` 的双向 kNN 边索引。
    """

    device = src_pos.device

    if src_indices is None:
        src_indices = torch.arange(src_pos.size(0), device=device)

    if src_indices.numel() == 0 or dst_pos.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    edges: list[Tensor] = []
    unique_batches = torch.unique(src_batch[src_indices])

    for batch_id in unique_batches:
        src_mask = src_batch[src_indices] == batch_id
        src_ids = src_indices[src_mask]
        dst_ids = torch.where(dst_batch == batch_id)[0]

        if src_ids.numel() == 0 or dst_ids.numel() == 0:
            continue

        dist = torch.cdist(src_pos[src_ids], dst_pos[dst_ids])
        k_eff = min(max(1, int(k)), int(dst_ids.numel()))
        nn_dst = torch.topk(dist, k=k_eff, largest=False, dim=1).indices

        src_rep = src_ids.repeat_interleave(k_eff)
        dst_sel = dst_ids[nn_dst.reshape(-1)]
        edges.append(torch.stack([src_rep, dst_sel], dim=0))

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    return torch.cat(edges, dim=1)


def build_batched_radius_or_knn_edges(
    *,
    src_pos: Tensor,
    src_batch: Tensor,
    dst_pos: Tensor,
    dst_batch: Tensor,
    radius_cutoff: float,
    knn_k: int,
    ensure_src_coverage: bool = True,
    max_num_neighbors: int = 64,
) -> Tensor:
    """
    构建半径边或回退 kNN 边。

    优先在同一 batch 内建立满足半径阈值的邻接关系，
    当半径边为空或覆盖不完整时再补充 kNN 连接避免节点失联。

    Args:
        src_pos: 源节点坐标。
        src_batch: 源节点所属 batch 索引。
        dst_pos: 目标节点坐标。
        dst_batch: 目标节点所属 batch 索引。
        radius_cutoff: 半径使用的截断阈值。
        knn_k: 回退到 kNN 时使用的邻居数。
        ensure_src_coverage: 是否保证每个源节点至少连接到一个目标节点。
        max_num_neighbors: 单个节点允许保留的最大邻居数。

    Returns:
        Tensor: 优先基于半径、必要时补充 kNN 后的边索引。

    Raises:
        ValueError: 当输入 batch 或坐标张量形状不匹配时抛出。
    """

    device = src_pos.device

    if src_pos.numel() == 0 or dst_pos.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    radius_edges = radius(
        x=dst_pos,
        y=src_pos,
        r=radius_cutoff,
        batch_x=dst_batch,
        batch_y=src_batch,
        max_num_neighbors=max_num_neighbors,
    )

    if radius_edges.numel() > 0:
        edge_fw = radius_edges
    else:
        edge_fw = build_batched_bipartite_knn_edges(
            src_pos=src_pos,
            src_batch=src_batch,
            dst_pos=dst_pos,
            dst_batch=dst_batch,
            k=knn_k,
        )

    if ensure_src_coverage:
        covered_src = torch.unique(edge_fw[0]) if edge_fw.numel() > 0 else torch.zeros((0,), dtype=torch.long, device=device)
        all_src = torch.arange(src_pos.size(0), device=device)
        uncovered_src = all_src[~torch.isin(all_src, covered_src)]

        if uncovered_src.numel() > 0:
            extra_edges = build_batched_bipartite_knn_edges(
                src_pos=src_pos,
                src_batch=src_batch,
                dst_pos=dst_pos,
                dst_batch=dst_batch,
                k=1,
                src_indices=uncovered_src,
            )
            if extra_edges.numel() > 0:
                edge_fw = (
                    torch.cat([edge_fw, extra_edges], dim=1)
                    if edge_fw.numel() > 0
                    else extra_edges
                )

    if edge_fw.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    edge_fw = torch.unique(edge_fw, dim=1)

    if int(edge_fw[0].max().item()) >= src_pos.size(0) or int(edge_fw[1].max().item()) >= dst_pos.size(0):
        raise RuntimeError("Built bipartite edges violate src/dst index contract.")

    return edge_fw

"""
动态跨图边构建工具。

统一管理 ligand->protein_atom / ligand->protein_residue 的边方向契约：
所有正向边统一返回 [lig_idx, protein_idx]。
"""

from __future__ import annotations

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
    构建批内 src->dst 的 kNN 边，返回 [src_idx, dst_idx]。
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
    构建批内 src->dst 的半径边；半径边为空或不完整时回退 / 补充 kNN。

    返回契约始终为 [src_idx, dst_idx]。
    """

    device = src_pos.device

    if src_pos.numel() == 0 or dst_pos.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    # PyG radius(x=dst, y=src) 返回 [src_idx, dst_idx]
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

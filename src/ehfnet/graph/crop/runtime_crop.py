"""
运行时裁剪工具。

负责围绕候选中心裁剪蛋白配体局部子图，
并保持索引、坐标与附加字段的一致性。
"""


import copy

import torch

from torch import Tensor
from torch_geometric.data import HeteroData
from torch_scatter import scatter_min

from ehfnet.graph.builders import GraphBuilder


def _normalize_atom_residue_idx(data: HeteroData) -> Tensor:
    """
    归一化蛋白原子到残基的映射索引。

    运行时局部裁剪既可能收到原始单样本图，也可能收到由 batch
    拆回的样本。后者的 `residue_idx` 可能仍带有全局偏移，因此这里
    在裁剪入口统一归一化为局部残基索引。

    Args:
        data: 当前处理的图数据对象。

    Returns:
        Tensor: 与当前样本局部残基节点对齐的 `residue_idx`。

    Raises:
        RuntimeError: 当 `residue_idx` 无法与局部残基节点数对齐时抛出。
    """
    residue_idx = data["protein_atom"].residue_idx.long()
    num_residues = int(data["protein_residue"].num_nodes)
    if residue_idx.numel() == 0 or num_residues <= 0:
        return residue_idx

    min_idx = int(residue_idx.min().item())
    max_idx = int(residue_idx.max().item())
    if 0 <= min_idx and max_idx < num_residues:
        return residue_idx

    shifted = residue_idx - min_idx
    if min_idx >= 0 and int(shifted.max().item()) < num_residues:
        return shifted

    unique_idx = torch.unique(residue_idx, sorted=True)
    if int(unique_idx.numel()) > num_residues:
        raise RuntimeError(
            "protein_atom.residue_idx references more residues than protein_residue stores "
            f"(unique={int(unique_idx.numel())}, num_residues={num_residues})."
        )

    positions = torch.bucketize(residue_idx, unique_idx)
    normalized = positions.long()
    if int(normalized.max().item()) >= num_residues:
        raise RuntimeError(
            "protein_atom.residue_idx is inconsistent with protein_residue.num_nodes "
            f"(max_idx={max_idx}, num_residues={num_residues})."
        )
    return normalized


def compute_ligand_center(data: HeteroData) -> Tensor:
    """
    计算配体几何中心。

    对当前图中的配体原子坐标取均值，
    作为局部裁剪或候选初始化时的默认中心。

    Args:
        data: 当前处理的图数据对象。

    Returns:
        Tensor: 配体原子的几何中心坐标。
    """
    lig_pos = data["ligand_atom"].pos
    if lig_pos.numel() == 0:
        return torch.zeros(3, dtype=torch.float32)
    return lig_pos.mean(dim=0)


def crop_graph_to_center(
    data: HeteroData,
    *,
    center: Tensor,
    radius: float,
    graph_builder: GraphBuilder,
    min_residues: int,
    atom_margin: float,
) -> HeteroData:
    """
    围绕中心裁剪局部子图。

    根据给定中心和半径筛选蛋白原子与残基，
    并重建局部图拓扑与上下文字段供局部对接流程使用。

    Args:
        data: 当前处理的图数据对象。
        center: 局部裁剪或特征汇总所围绕的中心坐标。
        radius: 局部裁剪半径。
        graph_builder: 用于构图或重建局部图的图构建器。
        min_residues: 局部裁剪后至少保留的残基数。
        atom_margin: 按原子距离补充残基时使用的额外边界。

    Returns:
        HeteroData: 围绕目标中心裁剪并重建后的局部子图。

    Raises:
        ValueError: 当输入中心坐标形状不是 `[3]` 时抛出。
    """
    if center.ndim != 1 or center.numel() != 3:
        raise ValueError("center must have shape [3].")

    center = center.to(
        device=data["protein_residue"].pos.device,
        dtype=data["protein_residue"].pos.dtype,
    )

    residue_pos = data["protein_residue"].pos
    residue_dist = torch.norm(residue_pos - center.unsqueeze(0), dim=-1)

    atom_pos = data["protein_atom"].pos
    atom_residue_idx = _normalize_atom_residue_idx(data)
    atom_dist = torch.norm(atom_pos - center.unsqueeze(0), dim=-1)
    residue_atom_min_dist = torch.full(
        (int(data["protein_residue"].num_nodes),),
        float("inf"),
        dtype=atom_dist.dtype,
        device=atom_dist.device,
    )
    if atom_dist.numel() > 0:
        residue_atom_min_dist = scatter_min(
            atom_dist,
            atom_residue_idx.long(),
            dim=0,
            dim_size=int(data["protein_residue"].num_nodes),
        )[0]

    residue_score = torch.minimum(residue_dist, residue_atom_min_dist)
    residue_mask = (residue_dist <= float(radius)) | (
        residue_atom_min_dist <= float(radius + atom_margin)
    )

    if int(residue_mask.sum().item()) < min_residues:
        k = min(max(min_residues, 1), residue_pos.size(0))
        nearest = torch.topk(residue_score, k=k, largest=False).indices
        residue_mask = torch.zeros_like(residue_mask)
        residue_mask[nearest] = True

    residue_idx = residue_mask.nonzero(as_tuple=False).view(-1)
    residue_old_to_new = torch.full(
        (data["protein_residue"].num_nodes,),
        -1,
        dtype=torch.long,
        device=residue_pos.device,
    )
    residue_old_to_new[residue_idx] = torch.arange(
        residue_idx.numel(),
        device=residue_pos.device,
    )

    atom_mask = residue_mask[atom_residue_idx]
    if int(atom_mask.sum().item()) == 0 and atom_pos.size(0) > 0:
        k = min(max(8, min_residues), atom_pos.size(0))
        nearest_atom = torch.topk(atom_dist, k=k, largest=False).indices
        atom_mask = torch.zeros_like(atom_mask)
        atom_mask[nearest_atom] = True

        residue_mask = torch.zeros_like(residue_mask)
        residue_mask[atom_residue_idx[nearest_atom]] = True
        residue_idx = residue_mask.nonzero(as_tuple=False).view(-1)
        residue_old_to_new = torch.full(
            (data["protein_residue"].num_nodes,),
            -1,
            dtype=torch.long,
            device=residue_pos.device,
        )
        residue_old_to_new[residue_idx] = torch.arange(
            residue_idx.numel(),
            device=residue_pos.device,
        )
        atom_mask = residue_mask[atom_residue_idx]

    atom_idx = atom_mask.nonzero(as_tuple=False).view(-1)
    out = HeteroData()

    def _copy_store(node_type: str, indices: Tensor | None = None) -> None:
        src_store = data[node_type]
        dst_store = out[node_type]
        num_nodes = int(src_store.num_nodes)

        if indices is None:
            indices = torch.arange(
                num_nodes,
                device=src_store.pos.device if hasattr(src_store, "pos") else center.device,
            )

        for key, value in src_store.items():
            if key in {"batch", "ptr", "num_nodes"}:
                continue
            if torch.is_tensor(value):
                if value.ndim > 0 and value.size(0) == num_nodes:
                    dst_store[key] = value[indices].clone()
                else:
                    dst_store[key] = value.clone()
            else:
                dst_store[key] = copy.deepcopy(value)

        dst_store.num_nodes = int(indices.numel())

    _copy_store("ligand_atom")
    _copy_store("ligand_molecule")
    _copy_store("protein_residue", residue_idx)
    _copy_store("protein_atom", atom_idx)

    local_atom_residue_idx = residue_old_to_new[atom_residue_idx[atom_idx]]
    valid_atom_mask = local_atom_residue_idx >= 0
    if not bool(valid_atom_mask.all()):
        atom_idx = atom_idx[valid_atom_mask]
        _copy_store("protein_atom", atom_idx)
        local_atom_residue_idx = residue_old_to_new[atom_residue_idx[atom_idx]]

    out["protein_atom"].residue_idx = local_atom_residue_idx.long()
    out["protein_context"].x_cont = graph_builder.build_context_node_features(
        out,
        center=center,
    )
    out["protein_context"].num_nodes = 1
    out = graph_builder.build_graph_topology(out)

    for attr in [
        "pdb_id",
        "dataset_index",
        "dataset_pdb_id",
        "y_energy",
        "torsion_indices",
        "torsion_moving_mask",
        "loss_progress",
        "loss_warmup_end",
        "loss_is_training",
    ]:
        if hasattr(data, attr):
            value = getattr(data, attr)
            setattr(
                out,
                attr,
                value.clone() if torch.is_tensor(value) else copy.deepcopy(value),
            )

    out.crop_center = center.detach().clone()
    out.crop_radius = float(radius)
    return out

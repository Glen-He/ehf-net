"""
运行时局部裁剪工具。

在 full-protein 缓存上按候选中心动态裁剪出局部蛋白上下文，
以支持 blind / two-stage docking，而不依赖预处理阶段的真实配体口袋裁剪。
"""

from __future__ import annotations

import copy
import torch

from torch import Tensor
from torch_geometric.data import HeteroData

from ehfnet.graph.builder import GraphBuilder


def compute_ligand_center(data: HeteroData) -> Tensor:
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
    min_residues: int = 12,
    atom_margin: float = 2.0,
) -> HeteroData:
    if center.ndim != 1 or center.numel() != 3:
        raise ValueError("center must have shape [3].")

    center = center.to(device=data["protein_residue"].pos.device, dtype=data["protein_residue"].pos.dtype)

    residue_pos = data["protein_residue"].pos
    residue_dist = torch.norm(residue_pos - center.unsqueeze(0), dim=-1)
    residue_mask = residue_dist <= float(radius)

    if int(residue_mask.sum().item()) < min_residues:
        k = min(max(min_residues, 1), residue_pos.size(0))
        nearest = torch.topk(residue_dist, k=k, largest=False).indices
        residue_mask = torch.zeros_like(residue_mask)
        residue_mask[nearest] = True

    residue_idx = residue_mask.nonzero(as_tuple=False).view(-1)
    residue_old_to_new = torch.full(
        (data["protein_residue"].num_nodes,),
        -1,
        dtype=torch.long,
        device=residue_pos.device,
    )
    residue_old_to_new[residue_idx] = torch.arange(residue_idx.numel(), device=residue_pos.device)

    atom_pos = data["protein_atom"].pos
    atom_residue_idx = data["protein_atom"].residue_idx
    atom_dist = torch.norm(atom_pos - center.unsqueeze(0), dim=-1)
    atom_mask = residue_mask[atom_residue_idx] | (atom_dist <= float(radius + atom_margin))

    if int(atom_mask.sum().item()) == 0:
        k = min(max(8, min_residues), atom_pos.size(0))
        nearest_atom = torch.topk(atom_dist, k=k, largest=False).indices
        atom_mask = torch.zeros_like(atom_mask)
        atom_mask[nearest_atom] = True

    atom_idx = atom_mask.nonzero(as_tuple=False).view(-1)

    out = HeteroData()

    def _copy_store(node_type: str, indices: Tensor | None = None) -> None:
        src_store = data[node_type]
        dst_store = out[node_type]
        num_nodes = int(src_store.num_nodes)

        if indices is None:
            indices = torch.arange(num_nodes, device=src_store.pos.device if hasattr(src_store, "pos") else center.device)

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

    local_atom_residue_idx = residue_old_to_new[data["protein_atom"].residue_idx[atom_idx]]
    valid_atom_mask = local_atom_residue_idx >= 0
    if not bool(valid_atom_mask.all()):
        atom_idx = atom_idx[valid_atom_mask]
        _copy_store("protein_atom", atom_idx)
        local_atom_residue_idx = residue_old_to_new[data["protein_atom"].residue_idx[atom_idx]]

    out["protein_atom"].residue_idx = local_atom_residue_idx.long()

    if out["protein_residue"].num_nodes > 0:
        pocket_cont = out["protein_residue"].x_cont.mean(dim=0, keepdim=True)
    else:
        feat_dim = data["protein_residue"].x_cont.size(1)
        pocket_cont = torch.zeros((1, feat_dim), dtype=data["protein_residue"].x_cont.dtype)

    out["protein_pocket"].x_cont = pocket_cont
    out["protein_pocket"].num_nodes = 1

    out = graph_builder._build_graph_topology(out)

    for attr in ["pdb_id", "y_energy", "torsion_indices", "torsion_moving_mask", "loss_progress", "loss_warmup_end", "loss_is_training"]:
        if hasattr(data, attr):
            value = getattr(data, attr)
            setattr(out, attr, value.clone() if torch.is_tensor(value) else copy.deepcopy(value))

    out.crop_center = center.detach().clone()
    out.crop_radius = float(radius)
    return out

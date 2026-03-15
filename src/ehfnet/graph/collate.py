"""
图批处理工具。

负责样本拼接、batch 属性整理和扭转约束合并，
服务 DataLoader 阶段的批处理组装。
"""


from typing import cast

import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.data.data import BaseData


class GraphCollator:
    """
    图批处理器。

    负责将多个 `HeteroData` 样本合并为批对象，
    并同步整理 torsion 约束等跨样本需要重建的附加字段。
    """

    def __init__(self, *, follow_batch: list[str] | None = None) -> None:
        """
        初始化对象。

        Args:
            follow_batch: follow_batch 中列出的节点类型所属的 batch 索引。
        """

        self.follow_batch = follow_batch or []


    def collate(self, samples: list[HeteroData]) -> Batch:
        """
        将样本列表拼接为 Batch。

        调用 PyG 默认拼接逻辑，并补齐 torsion 约束与 residue 索引偏移。

        Args:
            samples: 待拼接的 HeteroData 样本列表。

        Returns:
            Batch: 返回拼接后的批量图对象，含 torsion_indices、torsion_moving_mask 及偏移后的 residue_idx。
        """

        exclude_keys = ["torsion_indices", "torsion_moving_mask"]
        batch = Batch.from_data_list(
            cast(list[BaseData], samples),
            follow_batch=self.follow_batch,
            exclude_keys=exclude_keys,
        )
        batch = self._collate_torsion_constraints(batch, samples)
        return self._collate_residue_indices(batch, samples)


    def _collate_torsion_constraints(self, batch: Batch, samples: list[HeteroData]) -> Batch:
        """
        合并 torsion_indices 与 torsion_moving_mask。

        这两个字段属于跨节点（ligand_atom）的大矩阵约束，不适合让 PyG 默认的拼接逻辑处理，
        需要在 batch 维度上做索引偏移，并构建全局 moving mask。

        Args:
            batch: 已由 PyG 默认逻辑拼接好的 Batch。
            samples: 原始样本列表。

        Returns:
            Batch: 返回补齐扭转约束字段后的批量图对象。

        Raises:
            ValueError: 当输入参数或运行时状态不满足要求时抛出。
        """

        if not samples:
            return batch

        has_any_torsion = any(hasattr(data, "torsion_indices") for data in samples)

        if not has_any_torsion:
            return batch

        all_indices: list[torch.Tensor] = []
        all_masks: list[tuple[torch.Tensor, int, int]] = []
        ligand_atom_offset = 0

        for data in samples:
            if hasattr(data["ligand_atom"], "num_nodes") and data["ligand_atom"].num_nodes is not None:
                n_lig = int(data["ligand_atom"].num_nodes)
            elif hasattr(data["ligand_atom"], "pos"):
                n_lig = data["ligand_atom"].pos.size(0)
            elif hasattr(data["ligand_atom"], "x"):
                n_lig = data["ligand_atom"].x.size(0)
            else:
                raise ValueError("Could not determine number of ligand atoms for collation.")

            if not hasattr(data, "torsion_indices") or not hasattr(data, "torsion_moving_mask"):
                raise ValueError(
                    "torsion_indices and torsion_moving_mask must be provided together."
                )

            torsion_indices = cast(torch.Tensor, data.torsion_indices)
            torsion_moving_mask = cast(torch.Tensor, data.torsion_moving_mask)

            if torsion_indices.ndim != 2 or torsion_indices.size(1) != 4:
                raise ValueError("torsion_indices must have shape [T, 4].")

            if torsion_moving_mask.ndim != 2 or torsion_moving_mask.size(1) != n_lig:
                raise ValueError(
                    "torsion_moving_mask must have shape [T, num_ligand_atoms] for each sample."
                )

            if torsion_indices.dtype != torch.long:
                torsion_indices = torsion_indices.long()

            if torsion_moving_mask.dtype != torch.bool:
                torsion_moving_mask = torsion_moving_mask.bool()

            n_torsions = int(torsion_indices.size(0))

            if n_torsions > 0:
                indices_with_offset = torsion_indices + ligand_atom_offset
                all_indices.append(indices_with_offset)
                all_masks.append((torsion_moving_mask, ligand_atom_offset, n_lig))

            ligand_atom_offset += n_lig

        if all_indices:
            setattr(batch, "torsion_indices", torch.cat(all_indices, dim=0))

        else:
            setattr(batch, "torsion_indices", torch.zeros((0, 4), dtype=torch.long))

        if all_masks:
            total_n_lig = ligand_atom_offset
            total_n_torsions = sum(int(mask.size(0)) for mask, _, _ in all_masks)
            global_mask = torch.zeros((total_n_torsions, total_n_lig), dtype=torch.bool)
            torsion_offset = 0

            for mask, lig_offset, n_lig in all_masks:
                n_torsions = int(mask.size(0))
                global_mask[
                    torsion_offset : torsion_offset + n_torsions,
                    lig_offset : lig_offset + n_lig,
                ] = mask
                torsion_offset += n_torsions

            setattr(batch, "torsion_moving_mask", global_mask)

        else:
            total_n_lig = int(batch["ligand_atom"].num_nodes)
            setattr(
                batch,
                "torsion_moving_mask",
                torch.zeros((0, total_n_lig), dtype=torch.bool),
            )

        return batch


    def _collate_residue_indices(self, batch: Batch, samples: list[HeteroData]) -> Batch:
        """
        对 protein_atom.residue_idx 做显式偏移。

        这是 atom -> residue 的结构映射，不属于 PyG 默认会识别并自动偏移的 edge_index。
        为支持 batched 下的动态 residue 几何重建，需要把它改成全局 residue 索引。

        Returns:
            Batch: 返回完成 residue 索引显式偏移后的批量图对象。
        """

        if not samples or "protein_atom" not in batch.node_types:
            return batch

        if not hasattr(batch["protein_atom"], "residue_idx"):
            return batch

        all_indices: list[torch.Tensor] = []
        residue_offset = 0

        for data in samples:
            if not hasattr(data["protein_atom"], "residue_idx"):
                continue

            residue_idx = cast(torch.Tensor, data["protein_atom"].residue_idx).long()
            all_indices.append(residue_idx + residue_offset)
            residue_offset += int(data["protein_residue"].num_nodes)

        if all_indices:
            batch["protein_atom"].residue_idx = torch.cat(all_indices, dim=0)

        return batch

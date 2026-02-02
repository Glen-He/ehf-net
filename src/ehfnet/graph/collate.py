"""
图数据批处理拼接

提供 HeteroData 的 batch 拼接逻辑，并对扭转约束字段进行定制化合并。
"""

import torch

from typing import cast
from torch_geometric.data import Batch, HeteroData
from torch_geometric.data.data import BaseData


class GraphCollator:
    """
    图数据 Collator。

    负责将多个 HeteroData 样本拼接成 Batch，并补齐跨样本的扭转约束张量。
    """

    def __init__(self, *, follow_batch: list[str] | None = None) -> None:
        """
        Args:
            follow_batch: 需要额外生成 `<node_type>_batch` 索引的节点类型列表
        """

        self.follow_batch = follow_batch or []


    def collate(self, samples: list[HeteroData]) -> Batch:
        """
        将样本列表拼接为 Batch。

        Args:
            samples: HeteroData 样本列表

        Returns:
            Batch 对象
        """

        exclude_keys = ["torsion_indices", "torsion_moving_mask"]
        batch = Batch.from_data_list(
            cast(list[BaseData], samples),
            follow_batch=self.follow_batch,
            exclude_keys=exclude_keys,
        )
        return self._collate_torsion_constraints(batch, samples)


    def _collate_torsion_constraints(self, batch: Batch, samples: list[HeteroData]) -> Batch:
        """
        合并 torsion_indices 与 torsion_moving_mask。

        这两个字段属于跨节点（ligand_atom）的大矩阵约束，不适合让 PyG 默认的拼接逻辑处理，
        需要在 batch 维度上做索引偏移，并构建全局 moving mask。

        Args:
            batch: 已由 PyG 默认逻辑拼接好的 Batch
            samples: 原始样本列表

        Returns:
            更新后的 Batch
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
            # 优先从属性获取，否则尝试推断
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

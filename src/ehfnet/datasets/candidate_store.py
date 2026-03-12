"""
候选构象缓存读取器。

设计原则：
1. 候选池与 HeteroData 主体解耦；
2. 训练阶段只按需采样单个 candidate；
3. 评估阶段可按 pdb_id 读取完整 candidate pool。
"""

from __future__ import annotations

import json
import logging
import os.path as osp
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from torch import Tensor


logger = logging.getLogger(__name__)


@dataclass
class CandidateRecord:
    pos: Tensor
    score: float | None
    rmsd: float | None
    source_id: int
    candidate_index: int


class CandidateStore:
    """
    基于目录缓存的候选构象读取器。

    默认目录结构：
    candidate_root/
      <source>/
        <pdb_id>/
          poses.pt | poses.npy | poses.npz
          meta.json
    """

    SOURCE_ID_MAP = {
        "unknown": 0,
        "vina": 1,
        "smina": 2,
        "gnina": 3,
        "mixed": 4,
    }

    def __init__(self, root: str, source: str = "vina") -> None:
        self.root = root
        self.source = source
        self.source_id = self.SOURCE_ID_MAP.get(source, self.SOURCE_ID_MAP["unknown"])

    def _candidate_dir(self, pdb_id: str) -> str:
        return osp.join(self.root, self.source, str(pdb_id).lower())

    def _meta_path(self, pdb_id: str) -> str:
        return osp.join(self._candidate_dir(pdb_id), "meta.json")

    def _resolve_pose_path(self, pdb_id: str) -> str | None:
        candidate_dir = self._candidate_dir(pdb_id)
        for file_name in ("poses.pt", "poses.npy", "poses.npz"):
            path = osp.join(candidate_dir, file_name)
            if osp.exists(path):
                return path
        return None

    def has_candidates(self, pdb_id: str) -> bool:
        return self._resolve_pose_path(pdb_id) is not None

    def load_candidate_meta(self, pdb_id: str) -> dict[str, Any]:
        meta_path = self._meta_path(pdb_id)
        if not osp.exists(meta_path):
            return {}

        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)

        if not isinstance(meta, dict):
            raise ValueError(f"Candidate meta must be a JSON object, got {type(meta).__name__}")
        return meta

    def _load_pose_tensor(self, pdb_id: str) -> Tensor:
        pose_path = self._resolve_pose_path(pdb_id)
        if pose_path is None:
            raise FileNotFoundError(f"No candidate pose file found for {pdb_id} under {self._candidate_dir(pdb_id)}")

        if pose_path.endswith(".pt"):
            poses = torch.load(pose_path, weights_only=False)
        elif pose_path.endswith(".npy"):
            poses = np.load(pose_path)
        elif pose_path.endswith(".npz"):
            npz = np.load(pose_path)
            if "poses" in npz:
                poses = npz["poses"]
            else:
                first_key = next(iter(npz.keys()), None)
                if first_key is None:
                    raise ValueError(f"Empty npz candidate file: {pose_path}")
                poses = npz[first_key]
        else:
            raise ValueError(f"Unsupported candidate pose format: {pose_path}")

        poses_tensor = poses if isinstance(poses, Tensor) else torch.as_tensor(poses)
        poses_tensor = poses_tensor.to(dtype=torch.float32)

        if poses_tensor.ndim != 3 or poses_tensor.size(-1) != 3:
            raise ValueError(
                f"Candidate poses must have shape [K, N, 3], got {tuple(poses_tensor.shape)} for {pdb_id}"
            )

        return poses_tensor

    def load_all_candidates(self, pdb_id: str) -> tuple[Tensor, dict[str, Any]]:
        poses = self._load_pose_tensor(pdb_id)
        meta = self.load_candidate_meta(pdb_id)
        return poses, meta

    def _candidate_indices_by_strategy(self, meta: dict[str, Any], strategy: str, total: int) -> list[int]:
        rmsd_values = meta.get("candidate_rmsd")
        if rmsd_values is None:
            return list(range(total))

        rmsd = torch.as_tensor(rmsd_values, dtype=torch.float32)
        if rmsd.numel() != total:
            logger.warning("Candidate RMSD metadata length mismatch; fallback to uniform sampling.")
            return list(range(total))

        if strategy == "near_native":
            selected = torch.where(rmsd < 2.0)[0]
        elif strategy == "medium":
            selected = torch.where((rmsd >= 2.0) & (rmsd < 5.0))[0]
        elif strategy == "hard":
            selected = torch.where(rmsd >= 5.0)[0]
        elif strategy == "mixed":
            groups = [
                torch.where(rmsd < 2.0)[0],
                torch.where((rmsd >= 2.0) & (rmsd < 5.0))[0],
                torch.where(rmsd >= 5.0)[0],
            ]
            non_empty = [g for g in groups if g.numel() > 0]
            if not non_empty:
                return list(range(total))
            group_idx = int(torch.randint(len(non_empty), (1,)).item())
            selected = non_empty[group_idx]
        else:
            selected = torch.arange(total)

        if selected.numel() == 0:
            return list(range(total))
        return selected.tolist()

    def sample_candidate(self, pdb_id: str, strategy: str = "uniform") -> CandidateRecord | None:
        if not self.has_candidates(pdb_id):
            return None

        poses, meta = self.load_all_candidates(pdb_id)
        total = int(poses.size(0))
        if total <= 0:
            return None

        candidate_indices = self._candidate_indices_by_strategy(meta, strategy=strategy, total=total)
        sampled_local = int(torch.randint(len(candidate_indices), (1,)).item())
        candidate_index = int(candidate_indices[sampled_local])

        scores = meta.get("candidate_scores") or meta.get("scores")
        rmsd_values = meta.get("candidate_rmsd")

        score = None
        if scores is not None and len(scores) > candidate_index:
            raw_score = scores[candidate_index]
            if raw_score is not None:
                score = float(raw_score)

        rmsd = None
        if rmsd_values is not None and len(rmsd_values) > candidate_index:
            raw_rmsd = rmsd_values[candidate_index]
            if raw_rmsd is not None:
                rmsd = float(raw_rmsd)

        return CandidateRecord(
            pos=poses[candidate_index],
            score=score,
            rmsd=rmsd,
            source_id=self.source_id,
            candidate_index=candidate_index,
        )
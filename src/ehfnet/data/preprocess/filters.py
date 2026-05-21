"""Reusable preprocessing filters with stable PyG cache identities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LigandGeometryPreFilter:
    """Reject samples whose ligand atom geometry is physically implausible."""

    min_atom_distance: float

    def __call__(self, data: Any) -> bool:
        """Return whether a graph sample passes the ligand geometry check."""

        if "ligand_atom" not in data or not hasattr(data["ligand_atom"], "pos"):
            return True
        ligand_pos = data["ligand_atom"].pos
        if ligand_pos.shape[0] <= 1:
            return True
        dist_mat = torch.cdist(ligand_pos, ligand_pos, p=2)
        dist_mat = dist_mat + (
            torch.eye(dist_mat.shape[0], device=dist_mat.device) * 1000.0
        )
        min_dist = float(dist_mat.min().item())
        if min_dist < self.min_atom_distance:
            logger.warning(
                "Filtering sample with unreasonable geometry: "
                "min atom distance = %.3f A",
                min_dist,
            )
            return False
        return True

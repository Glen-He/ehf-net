"""
共享 RBF 模块

高斯径向基函数距离编码，供 FrameAwareConv 与 PredictionHead 统一使用。
"""

import torch
import torch.nn as nn

from torch import Tensor


class GaussianRBF(nn.Module):
    """
    高斯径向基函数距离编码。

    将标量距离 d 扩展为高维特征向量：
        rbf_k(d) = exp(-0.5 · ((d - μ_k) / σ)²)

    μ_k 均匀分布在 [start, stop]，σ = 相邻中心间距。
    输出范围 (0, 1]，在 d = μ_k 处取最大值 1。
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 10.0,
        num_gaussians: int = 50,
    ) -> None:
        super().__init__()
        if stop <= start:
            raise ValueError(f"stop ({stop}) must be > start ({start})")
        if num_gaussians < 4:
            raise ValueError(f"num_gaussians must be >= 4, got {num_gaussians}")

        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer("offset", offset)
        sigma = (stop - start) / (num_gaussians - 1)
        self.coeff = -0.5 / (sigma**2)
        self.offset: Tensor

    def forward(self, dist: Tensor) -> Tensor:
        """
        Args:
            dist: 距离标量 [N] 或 [..., 1]（单位：Å），应为非负值

        Returns:
            RBF 特征 [..., num_gaussians]，范围 [0, 1]
        """
        dist = torch.clamp(dist, min=0.0)
        if dist.ndim == 1:
            dist_exp = dist.unsqueeze(-1)
        else:
            dist_exp = dist
        diff = dist_exp - self.offset
        return torch.exp(self.coeff * diff.pow(2))

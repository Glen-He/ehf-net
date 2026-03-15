"""
RBF 编码层。

提供距离的高斯径向基展开，
用于几何距离特征的连续表示。
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
        *,
        num_gaussians: int = 50,
    ) -> None:
        """
        初始化高斯 RBF 层。

        配置距离展开所需的中心分布和范围参数，
        为连续距离特征编码提供基础。

        Args:
            start: 高斯基中心覆盖的起始距离。
            stop: 高斯基中心覆盖的终止距离。
            num_gaussians: 用于展开距离特征的高斯基数量。

        Raises:
            ValueError: 当输入参数或运行时状态不满足要求时抛出。
        """
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
        执行距离的 RBF 展开。

        将输入距离映射到高维高斯基表示，
        供几何相关模块使用连续距离编码。

        Args:
            dist: 待展开的距离张量。

        Returns:
            Tensor: 返回与输入距离对齐的 RBF 特征张量，最后一维为 `num_gaussians`。
        """
        dist = torch.clamp(dist, min=0.0)
        if dist.ndim == 1:
            dist_exp = dist.unsqueeze(-1)
        else:
            dist_exp = dist
        diff = dist_exp - self.offset
        return torch.exp(self.coeff * diff.pow(2))

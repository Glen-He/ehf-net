"""
节点嵌入层

提供各类节点（原子、分子、残基、口袋）的嵌入层实现。
"""

import math
import torch

from torch import nn, Tensor

from ehfnet.encoders.feature_specs import (
    CatFeature,
    LIGAND_ATOM_CAT_SCHEMA,
    PROTEIN_ATOM_CAT_SCHEMA,
    PROTEIN_ATOM_SCALAR_DIM,
    PROTEIN_RESIDUE_CAT_SCHEMA,
    PROTEIN_RESIDUE_CONT_SCHEMA,
    PROTEIN_RESIDUE_CONTEXT_DIM,
    PROTEIN_RESIDUE_TORSION_DIM,
    PROTEIN_RESIDUE_TORSION_VALID_DIM,
    PROTEIN_RESIDUE_TORSION_VALID_START,
)


class TimeEmbedding(nn.Module):
    """
    时间嵌入模块

    将标量时间 t 映射到高维向量，采用固定正弦/余弦频率嵌入 + MLP。
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """
        Args:
            dim: 正弦嵌入的维度（必须是偶数）
            hidden_dim: MLP 的输出维度
        """
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}.")

        half_dim = dim // 2

        # 计算频率：10000^(-2k/d)
        # 为了数值稳定性，使用对数空间：exp(-log(10000) * (k / half_dim))
        exponent = -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32)
        exponent = exponent / half_dim
        freqs = torch.exp(exponent)

        # register_buffer 保存常量，不参与训练
        self.register_buffer("freqs", freqs.unsqueeze(0))
        self.freqs: Tensor

        # 将固定嵌入投影到隐藏维度的 MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        前向传播

        Args:
            t: 时间标量 [B]，通常在 [0, 1] 范围内

        Returns:
            时间嵌入向量 [B, hidden_dim]
        """
        # t: [B] -> [B, 1]
        t = t.unsqueeze(-1)

        # args: [B, half_dim]
        args = t * self.freqs

        # sinusoid: [B, dim]
        sinusoid = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # 投影到 hidden_dim
        return self.mlp(sinusoid)


class AtomEmbedding(nn.Module):
    """
    通用原子嵌入基类

    组合分类特征（Embedding）与连续特征（Linear），并拼接坐标。
    """

    def __init__(
        self,
        cat_schema: list[CatFeature],
        cont_feature_count: int,
        hidden_dim: int,
        stats: dict | None = None,
        *,
        scalar_feature_count: int | None = None,
    ) -> None:
        """
        Args:
            cat_schema: 分类特征配置列表
            cont_feature_count: 连续特征数量
            hidden_dim: 隐藏层维度
            stats: 统计数据字典，包含 mean 和 std (用于标准化)
        """

        super().__init__()

        # 分类特征嵌入
        self.embedding_layers = nn.ModuleList()
        total_categorical_dim = 0

        for feat in cat_schema:
            self.embedding_layers.append(
                nn.Embedding(
                    num_embeddings=feat.num_embeddings, embedding_dim=feat.embed_dim
                )
            )
            total_categorical_dim += feat.embed_dim
        
        self.scalar_feature_count = (
            cont_feature_count if scalar_feature_count is None else int(scalar_feature_count)
        )
        if not (0 <= self.scalar_feature_count <= cont_feature_count):
            raise ValueError(
                "scalar_feature_count must be in [0, cont_feature_count], got "
                f"{self.scalar_feature_count} for cont_feature_count={cont_feature_count}."
            )
        self.flag_feature_count = cont_feature_count - self.scalar_feature_count

        if stats is not None:
            mean = stats["mean"]
            std = stats["std"] + 1e-6
            if mean.numel() != cont_feature_count or std.numel() != cont_feature_count:
                raise ValueError(
                    f"AtomEmbedding stats shape mismatch: expected {cont_feature_count}, "
                    f"got mean={mean.numel()} std={std.numel()}."
                )
            if self.scalar_feature_count > 0:
                self.register_buffer("mean", mean[: self.scalar_feature_count].clone())
                self.register_buffer("std", std[: self.scalar_feature_count].clone())

        scalar_branch_dim = max(hidden_dim // 2, 16) if self.scalar_feature_count > 0 else 0
        flag_branch_dim = max(hidden_dim // 4, 8) if self.flag_feature_count > 0 else 0

        if self.scalar_feature_count > 0:
            self.scalar_branch = nn.Sequential(
                nn.Linear(self.scalar_feature_count, scalar_branch_dim),
                nn.SiLU(),
                nn.LayerNorm(scalar_branch_dim),
                nn.Linear(scalar_branch_dim, scalar_branch_dim),
            )

        if self.flag_feature_count > 0:
            self.flag_branch = nn.Sequential(
                nn.Linear(self.flag_feature_count, flag_branch_dim),
                nn.SiLU(),
                nn.LayerNorm(flag_branch_dim),
                nn.Linear(flag_branch_dim, flag_branch_dim),
            )

        mlp_in_dim = total_categorical_dim + scalar_branch_dim + flag_branch_dim

        self.projection_mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 这是一个可学习的向量，给每种类型的原子打上一个“身份标签”
        self.source_type_embedding = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)

        self.output_dim = 3 + hidden_dim

    def forward(
        self, x_cat: Tensor, x_cont: Tensor, pos: Tensor
    ) -> Tensor:
        """
        前向传播

        Args:
            x_cat: 离散特征 [N, num_cat_features]
            x_cont: 连续特征 [N, num_cont_features]
            pos: 原子坐标 [N, 3]

        Returns:
            拼接坐标后的原子特征 [N, 3 + hidden_dim]
        """

        feature_list: list[Tensor] = []

        x_cat_long = x_cat.long()

        for layer, raw_ids in zip(
            self.embedding_layers, x_cat_long.transpose(0, 1), strict=True
        ):
            feature_list.append(layer(raw_ids))

        if self.scalar_feature_count > 0:
            scalar_cont = x_cont[:, : self.scalar_feature_count]
            if hasattr(self, "mean"):
                scalar_cont = (scalar_cont - self.mean) / self.std
            feature_list.append(self.scalar_branch(scalar_cont))

        if self.flag_feature_count > 0:
            flag_cont = x_cont[:, self.scalar_feature_count :]
            feature_list.append(self.flag_branch(flag_cont))

        full_features = torch.cat(feature_list, dim=-1)
        projected_features = self.projection_mlp(full_features)
        
        # 添加来源类型标签（每个子类实例有独立的参数）
        projected_features = projected_features + self.source_type_embedding

        # 输出形状：[N, 3 + hidden_dim]
        return torch.cat([pos, projected_features], dim=-1)


class LigandAtomEmbedding(AtomEmbedding):
    """
    配体原子嵌入

    基于 AtomEmbedding，对配体原子特征进行编码。
    """

    def __init__(self, cont_feature_count: int, hidden_dim: int, stats: dict | None = None) -> None:
        super().__init__(
            LIGAND_ATOM_CAT_SCHEMA,
            cont_feature_count,
            hidden_dim,
            stats,
            scalar_feature_count=cont_feature_count,
        )


class ProteinAtomEmbedding(AtomEmbedding):
    """
    蛋白质原子嵌入

    基于 AtomEmbedding，对蛋白质原子特征进行编码。
    """

    def __init__(self, cont_feature_count: int, hidden_dim: int, stats: dict | None = None) -> None:
        super().__init__(
            PROTEIN_ATOM_CAT_SCHEMA,
            cont_feature_count,
            hidden_dim,
            stats,
            scalar_feature_count=PROTEIN_ATOM_SCALAR_DIM,
        )


class LigandMoleculeEmbedding(nn.Module):
    """
    配体分子嵌入

    仅对连续特征进行线性投影。
    """

    def __init__(self, cont_feature_count: int, hidden_dim: int, stats: dict | None = None) -> None:
        """
        Args:
            cont_feature_count: 连续特征数量
            hidden_dim: 隐藏层维度
            stats: 统计数据字典
        """
        super().__init__()
        self.output_dim = hidden_dim
        
        if stats is not None:
            mean = stats["mean"]
            std = stats["std"] + 1e-6
            if mean.numel() != cont_feature_count or std.numel() != cont_feature_count:
                raise ValueError(
                    f"LigandMoleculeEmbedding stats shape mismatch: expected {cont_feature_count}, "
                    f"got mean={mean.numel()} std={std.numel()}."
                )
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        self.projection_mlp = nn.Sequential(
            nn.Linear(cont_feature_count, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x_cont: Tensor) -> Tensor:
        """
        前向传播

        Args:
            x_cont: 连续特征 [M, num_cont_features]

        Returns:
            投影后的分子特征 [M, hidden_dim]
        """

        # 标准化连续特征 (Z-Score)
        if hasattr(self, "mean"):
            x_cont = (x_cont - self.mean) / self.std

        return self.projection_mlp(x_cont)


class ProteinResidueEmbedding(nn.Module):
    """
    蛋白质残基嵌入

    将 residue 的几何/有效性/segment 特征与 ESM 特征分支编码后再融合。
    """

    def __init__(self, cont_feature_count: int, hidden_dim: int, stats: dict | None = None) -> None:
        """
        Args:
            cont_feature_count: 连续特征数量（包含扭转角 sin/cos + ESM embeddings）
            hidden_dim: 隐藏层维度
            stats: 统计数据字典
        """

        super().__init__()

        residue_cont_dim = len(PROTEIN_RESIDUE_CONT_SCHEMA)
        torsion_dim = PROTEIN_RESIDUE_TORSION_DIM
        context_dim = PROTEIN_RESIDUE_CONTEXT_DIM
        esm_dim = cont_feature_count - residue_cont_dim
        if esm_dim <= 0:
            raise ValueError(
                f"cont_feature_count must be greater than residue_cont_dim={residue_cont_dim}, got {cont_feature_count}."
            )

        self._torsion_dim = torsion_dim
        self._context_dim = context_dim
        self._esm_dim = esm_dim
        self._residue_cont_dim = residue_cont_dim
        self.unk_esm_embedding = nn.Parameter(
            torch.randn((esm_dim,), dtype=torch.float32) * 0.02
        )

        # 分类特征嵌入
        config_list = PROTEIN_RESIDUE_CAT_SCHEMA

        self.embedding_layers = nn.ModuleList()
        total_categorical_dim = 0

        for feat in config_list:
            self.embedding_layers.append(
                nn.Embedding(
                    num_embeddings=feat.num_embeddings, embedding_dim=feat.embed_dim
                )
            )
            total_categorical_dim += feat.embed_dim
        
        # residue branch 不再做 dataset z-score：
        # - torsion 已在 [-1, 1]
        # - observed/segment flags 是结构语义，不应按数据集分布缩放
        # - ESM 保持独立分支，避免破坏预训练空间
        _ = stats

        self.residue_cont_norm = nn.LayerNorm(residue_cont_dim)
        self.esm_norm = nn.LayerNorm(esm_dim)

        self.residue_branch = nn.Sequential(
            nn.Linear(residue_cont_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.esm_branch = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        mlp_in_dim = total_categorical_dim + hidden_dim * 2

        self.projection_mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.output_dim = hidden_dim

    def forward(
        self,
        x_cat: Tensor,
        x_cont: Tensor,
        *,
        esm_missing_mask: Tensor | None = None,
    ) -> Tensor:
        """
        前向传播

        Args:
            x_cat: 离散特征 [R, num_cat_features]
            x_cont: 连续特征 [R, num_cont_features]

        Returns:
            投影后的残基特征 [R, hidden_dim]
        """

        if x_cont.size(1) != self._residue_cont_dim + self._esm_dim:
            raise ValueError(
                f"Expected x_cont last dim {self._residue_cont_dim + self._esm_dim}, got {x_cont.size(1)}."
            )

        residue_cont = x_cont[:, : self._residue_cont_dim]
        esm_cont = x_cont[:, self._residue_cont_dim :]

        torsion_cont = residue_cont[:, : self._torsion_dim]
        torsion_valid = residue_cont[
            :,
            PROTEIN_RESIDUE_TORSION_VALID_START : PROTEIN_RESIDUE_TORSION_VALID_START + PROTEIN_RESIDUE_TORSION_VALID_DIM,
        ]
        torsion_cont = torsion_cont * torsion_valid.repeat_interleave(2, dim=1)
        residue_cont = torch.cat(
            [torsion_cont, residue_cont[:, self._torsion_dim :]],
            dim=1,
        )

        if esm_missing_mask is not None:
            if esm_missing_mask.ndim != 1 or esm_missing_mask.size(0) != residue_cont.size(0):
                raise ValueError(
                    "esm_missing_mask must have shape [R] matching x_cont.shape[0]."
                )

            mask = esm_missing_mask.to(device=esm_cont.device, dtype=torch.bool)
            unk = self.unk_esm_embedding.to(device=esm_cont.device, dtype=esm_cont.dtype)
            esm_cont = torch.where(mask.unsqueeze(-1), unk.unsqueeze(0), esm_cont)

        embedded_list: list[Tensor] = []
        x_cat_long = x_cat.long()

        for layer, raw_ids in zip(
            self.embedding_layers, x_cat_long.transpose(0, 1), strict=True
        ):
            embedded_list.append(layer(raw_ids))

        residue_branch = self.residue_branch(self.residue_cont_norm(residue_cont))
        esm_branch = self.esm_branch(self.esm_norm(esm_cont))
        embedded_list.extend([residue_branch, esm_branch])

        full_features = torch.cat(embedded_list, dim=-1)

        return self.projection_mlp(full_features)


class ProteinPocketEmbedding(nn.Module):
    """
    蛋白质口袋（整体）嵌入

    将显式 pocket summary 特征投影到隐藏空间，并叠加可学习 token。
    """

    def __init__(self, cont_feature_count: int, hidden_dim: int) -> None:
        """
        Args:
            cont_feature_count: pocket 连续特征维度
            hidden_dim: 隐藏层维度
        """

        super().__init__()
        self.cont_norm = nn.LayerNorm(cont_feature_count)
        self.projection_mlp = nn.Sequential(
            nn.Linear(cont_feature_count, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.initial_pocket_emb = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.output_dim = hidden_dim


    def forward(self, x_cont: Tensor) -> Tensor:
        """
        前向传播

        Args:
            x_cont: 口袋连续特征 [N_pocket, D]

        Returns:
            口袋节点嵌入 [num_nodes, hidden_dim]
        """

        projected = self.projection_mlp(self.cont_norm(x_cont))
        return projected + self.initial_pocket_emb.expand(x_cont.size(0), -1)

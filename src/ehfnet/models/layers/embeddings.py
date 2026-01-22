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
    PROTEIN_RESIDUE_CAT_SCHEMA,
    PROTEIN_RESIDUE_CONT_SCHEMA,
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
            raise ValueError(f"dim 必须是偶数，当前为 {dim}")

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
    ) -> None:
        """
        Args:
            cat_schema: 分类特征配置列表
            cont_feature_count: 连续特征数量
            hidden_dim: 隐藏层维度
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

        # 连续特征归一化
        self.cont_norm = nn.LayerNorm(cont_feature_count)

        mlp_in_dim = total_categorical_dim + cont_feature_count

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

        # 归一化连续特征
        x_cont_normed = self.cont_norm(x_cont)
        feature_list.append(x_cont_normed)

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

    def __init__(self, cont_feature_count: int, hidden_dim: int) -> None:
        super().__init__(LIGAND_ATOM_CAT_SCHEMA, cont_feature_count, hidden_dim)


class ProteinAtomEmbedding(AtomEmbedding):
    """
    蛋白质原子嵌入

    基于 AtomEmbedding，对蛋白质原子特征进行编码。
    """

    def __init__(self, cont_feature_count: int, hidden_dim: int) -> None:
        super().__init__(PROTEIN_ATOM_CAT_SCHEMA, cont_feature_count, hidden_dim)


class LigandMoleculeEmbedding(nn.Module):
    """
    配体分子嵌入

    仅对连续特征进行线性投影。
    """

    def __init__(self, cont_feature_count: int, hidden_dim: int) -> None:
        """
        Args:
            cont_feature_count: 连续特征数量
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.output_dim = hidden_dim

        # 连续特征归一化
        self.cont_norm = nn.LayerNorm(cont_feature_count)

        self.projection_mlp = nn.Sequential(
            nn.Linear(cont_feature_count, hidden_dim),
            nn.SiLU(),
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

        x_cont_normed = self.cont_norm(x_cont)
        return self.projection_mlp(x_cont_normed)


class ProteinResidueEmbedding(nn.Module):
    """
    蛋白质残基嵌入

    组合分类特征（Embedding）和连续特征（Linear）。
    注意：连续特征包含预训练的 ESM embeddings，使用 LayerNorm 保持其语义空间。
    """

    def __init__(self, cont_feature_count: int, hidden_dim: int) -> None:
        """
        Args:
            cont_feature_count: 连续特征数量（包含扭转角 sin/cos + ESM embeddings）
            hidden_dim: 隐藏层维度
        """

        super().__init__()

        torsion_dim = len(PROTEIN_RESIDUE_CONT_SCHEMA)
        esm_dim = cont_feature_count - torsion_dim
        if esm_dim <= 0:
            raise ValueError(
                f"cont_feature_count must be greater than torsion_dim={torsion_dim}, got {cont_feature_count}."
            )

        self._torsion_dim = torsion_dim
        self._esm_dim = esm_dim
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

        # 连续特征归一化（包含扭转角 sin/cos + ESM embeddings）
        self.cont_norm = nn.LayerNorm(cont_feature_count)

        mlp_in_dim = total_categorical_dim + cont_feature_count

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

        if esm_missing_mask is not None:
            if esm_missing_mask.ndim != 1 or esm_missing_mask.size(0) != x_cont.size(0):
                raise ValueError(
                    "esm_missing_mask must have shape [R] matching x_cont.shape[0]."
                )

            mask = esm_missing_mask.to(device=x_cont.device, dtype=torch.bool)
            torsion_dim = self._torsion_dim
            unk = self.unk_esm_embedding.to(device=x_cont.device, dtype=x_cont.dtype)
            esm_part = x_cont[:, torsion_dim:]
            esm_part = torch.where(mask.unsqueeze(-1), unk.unsqueeze(0), esm_part)
            x_cont = torch.cat([x_cont[:, :torsion_dim], esm_part], dim=1)

        embedded_list: list[Tensor] = []
        x_cat_long = x_cat.long()

        for layer, raw_ids in zip(
            self.embedding_layers, x_cat_long.transpose(0, 1), strict=True
        ):
            embedded_list.append(layer(raw_ids))

        # 归一化连续特征（包含 ESM embeddings）
        x_cont_normed = self.cont_norm(x_cont)
        embedded_list.append(x_cont_normed)

        full_features = torch.cat(embedded_list, dim=-1)

        return self.projection_mlp(full_features)


class ProteinPocketEmbedding(nn.Module):
    """
    蛋白质口袋（整体）嵌入

    为每个口袋节点学习一个可训练的初始嵌入向量。
    """

    def __init__(self, hidden_dim: int) -> None:
        """
        Args:
            hidden_dim: 隐藏层维度
        """

        super().__init__()
        self.initial_pocket_emb = nn.Parameter(torch.randn(1, hidden_dim))
        self.output_dim = hidden_dim


    def forward(self, num_nodes: int) -> Tensor:
        """
        前向传播

        Args:
            num_nodes: 口袋节点数量

        Returns:
            口袋节点嵌入 [num_nodes, hidden_dim]
        """

        return self.initial_pocket_emb.repeat(num_nodes, 1)

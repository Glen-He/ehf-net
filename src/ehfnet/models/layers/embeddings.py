"""
嵌入层工具。

负责时间、原子、残基和分子级特征嵌入，
为编码器提供统一的隐藏表示输入。
"""


import math
import torch

from torch import nn, Tensor

from ehfnet.data.featurizers import (
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
    时间嵌入模块。

    将连续时间步映射为高维隐藏表示，
    为流匹配训练和推理中的时间条件建模提供统一输入。
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """
        初始化对象。

        Args:
            dim: 维度。
            hidden_dim: 隐藏层维度。

        Raises:
            ValueError: 当输入参数或运行时状态不满足要求时抛出。
        """
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}.")

        half_dim = dim // 2

        exponent = -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32)
        exponent = exponent / half_dim
        freqs = torch.exp(exponent)

        self.register_buffer("freqs", freqs.unsqueeze(0))
        self.freqs: Tensor

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        前向传播

        Args:
            t: 时间步标量或向量，形状 [B] 或 [N]。

        Returns:
            Tensor: 时间嵌入向量，形状 [B, hidden_dim]。
        """
        t = t.unsqueeze(-1)
        args = t * self.freqs
        sinusoid = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(sinusoid)


class AtomEmbedding(nn.Module):
    """
    通用原子嵌入基类。

    负责融合分类特征、连续特征与坐标相关输入，
    为配体原子和蛋白原子嵌入提供共享实现。
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
        初始化通用原子嵌入层。

        配置分类特征嵌入、连续特征投影和归一化逻辑，
        作为配体与蛋白原子嵌入的共享基础实现。

        Args:
            cat_schema: catschema。
            cont_feature_count: cont特征的数量。
            hidden_dim: 隐藏层维度。
            stats: 统计量。
            scalar_feature_count: scalar特征的数量。

        Raises:
            ValueError: 当输入参数或运行时状态不满足要求时抛出。
        """

        super().__init__()

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

        self.source_type_embedding = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)

        self.output_dim = 3 + hidden_dim

    def forward(
        self, x_cat: Tensor, x_cont: Tensor, pos: Tensor
    ) -> Tensor:
        """
        前向传播

        Args:
            x_cat: xcat。
            x_cont: xcont。
            pos: 节点坐标张量。

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

        projected_features = projected_features + self.source_type_embedding
        return torch.cat([pos, projected_features], dim=-1)


class LigandAtomEmbedding(AtomEmbedding):
    """
    配体原子嵌入层。

    基于通用原子嵌入逻辑编码配体原子特征，
    为主干编码器提供配体侧节点初始表示。
    """

    def __init__(
        self,
        cont_feature_count: int,
        hidden_dim: int,
        *,
        stats: dict | None = None,
    ) -> None:
        """
        初始化配体原子嵌入层。

        基于通用原子嵌入层配置配体特征输入，
        为配体原子节点生成初始隐藏表示。

        Args:
            cont_feature_count: 连续特征数量。
            hidden_dim: 隐藏层维度。
            stats: 归一化统计量，含 mean/std。
        """
        super().__init__(
            LIGAND_ATOM_CAT_SCHEMA,
            cont_feature_count,
            hidden_dim,
            stats,
            scalar_feature_count=cont_feature_count,
        )


class ProteinAtomEmbedding(AtomEmbedding):
    """
    蛋白原子嵌入层。

    基于通用原子嵌入逻辑编码蛋白原子特征，
    为蛋白原子层消息传递提供初始表示。
    """

    def __init__(
        self,
        cont_feature_count: int,
        hidden_dim: int,
        *,
        stats: dict | None = None,
    ) -> None:
        """
        初始化蛋白原子嵌入层。

        基于通用原子嵌入层配置蛋白特征输入，
        为蛋白原子节点生成初始隐藏表示。

        Args:
            cont_feature_count: 连续特征数量。
            hidden_dim: 隐藏层维度。
            stats: 归一化统计量，含 mean/std。
        """
        super().__init__(
            PROTEIN_ATOM_CAT_SCHEMA,
            cont_feature_count,
            hidden_dim,
            stats,
            scalar_feature_count=PROTEIN_ATOM_SCALAR_DIM,
        )


class LigandMoleculeEmbedding(nn.Module):
    """
    配体分子嵌入层。

    对配体分子级连续特征进行投影编码，
    为分子级全局节点提供可与其他层级交互的隐藏表示。
    """

    def __init__(
        self,
        cont_feature_count: int,
        hidden_dim: int,
        *,
        stats: dict | None = None,
    ) -> None:
        """
        初始化配体分子嵌入层。

        Args:
            cont_feature_count: 连续特征数量。
            hidden_dim: 隐藏层维度。
            stats: 归一化统计量，含 mean/std。

        Raises:
            ValueError: 当 stats 维度与 cont_feature_count 不匹配时抛出。
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
            x_cont: 配体分子连续特征，形状 [M, cont_feature_count]。

        Returns:
            Tensor: 投影后的分子特征，形状 [M, hidden_dim]。
        """

        if hasattr(self, "mean"):
            x_cont = (x_cont - self.mean) / self.std

        return self.projection_mlp(x_cont)


class ProteinResidueEmbedding(nn.Module):
    """
    蛋白残基嵌入层。

    融合残基几何特征、有效性特征与 ESM 特征分支，
    生成残基层消息传递和中心提议使用的输入表示。
    """

    def __init__(
        self,
        cont_feature_count: int,
        hidden_dim: int,
        *,
        stats: dict | None = None,
    ) -> None:
        """
        初始化蛋白残基嵌入层。

        Args:
            cont_feature_count: 连续特征数量（含 ESM 维度）。
            hidden_dim: 隐藏层维度。
            stats: 归一化统计量，含 mean/std。

        Raises:
            ValueError: 当输入参数或运行时状态不满足要求时抛出。
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
            x_cat: xcat。
            x_cont: xcont。
            esm_missing_mask: esmmissingmask。

        Returns:
            Tensor: 返回融合类别特征、连续特征和 ESM 分支后的残基嵌入表示。

        Raises:
            ValueError: 当输入参数或运行时状态不满足要求时抛出。
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


class ProteinContextEmbedding(nn.Module):
    """
    蛋白上下文嵌入层。

    将局部上下文 summary 特征映射到隐藏空间，
    为局部对接阶段的 context 节点提供统一表示。
    """

    def __init__(self, cont_feature_count: int, hidden_dim: int) -> None:
        """
        初始化对象。

        Args:
            cont_feature_count: cont特征的数量。
            hidden_dim: 隐藏层维度。
        """

        super().__init__()
        self.cont_norm = nn.LayerNorm(cont_feature_count)
        self.projection_mlp = nn.Sequential(
            nn.Linear(cont_feature_count, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.initial_context_emb = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.output_dim = hidden_dim


    def forward(self, x_cont: Tensor) -> Tensor:
        """
        前向传播

        Args:
            x_cont: xcont。

        Returns:
            局部上下文节点嵌入 [num_nodes, hidden_dim]
        """

        projected = self.projection_mlp(self.cont_norm(x_cont))
        return projected + self.initial_context_emb.expand(x_cont.size(0), -1)

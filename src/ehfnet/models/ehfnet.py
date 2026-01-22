"""
EHFNet 主模型

结合分层 EGNN 编码器与统一预测头。
"""

import torch

from typing import TypedDict
from torch import nn, Tensor
from torch_geometric.data import HeteroData

from ehfnet.models.layers.encoder import EHFEncoder
from ehfnet.models.heads.prediction import PredictionHead
from ehfnet.geometry.dynamics import VelocityDecomposer


class EHFNetOutput(TypedDict):
    """
    EHFNet 模型输出类型定义
    """

    x_dict: dict[str, Tensor]   # 节点特征字典
    v_atomic: Tensor            # 混合后的总原子速度 [N, 3]
    v_coarse: Tensor            # EGNN 原始速度 [N, 3]
    v_correction: Tensor        # 物理头预测速度 [N, 3]
    alpha: Tensor               # 平均混合权重标量
    v_translation: Tensor       # 平移速度 [B, 3]
    v_rotation: Tensor          # 旋转速度 [B, 3]
    v_torsion: Tensor | None    # 扭转角速度 [T, 1] 或 None
    binding_affinity: Tensor    # 结合能 [B, 1]
    force_atomic: Tensor        # 原子级力场 [N, 3]
    pos_updated: Tensor         # 更新后的坐标 [N, 3]


class EHFNet(nn.Module):
    """
    EHFNet 顶层模型

    功能：
    1. 调用 EHFEncoder 获取更新后的原子坐标和特征
    2. 调用 PredictionHead 同时预测：
       - 原子级速度场 (v_atomic)
       - 配体-蛋白结合能 (binding_affinity)
    3. 将 v_atomic 分解为符合物理约束的：
       - 刚体平移速度 (v_translation)
       - 刚体旋转速度 (v_rotation)
       - 扭转角速度 (v_torsion)
    """

    def __init__(
        self,
        hidden_dim: int,
        time_dim: int,
        num_gnn_blocks: int,
        lig_atom_cont_count: int,
        lig_mol_cont_count: int,
        pro_atom_cont_count: int,
        pro_res_cont_count: int,
        *,
        m_dim_scalar: int = 16,
        dropout_rate: float = 0.0,
        num_rbf: int = 50,
        r_cutoff: float = 10.0,
        fix_protein: bool = True,
    ) -> None:
        """
        Args:
            hidden_dim: 隐藏层维度
            time_dim: 时间嵌入维度（必须是偶数）
            num_gnn_blocks: GNN 块数量
            lig_atom_cont_count: 配体原子连续特征数量
            lig_mol_cont_count: 配体分子连续特征数量
            pro_atom_cont_count: 蛋白原子连续特征数量
            pro_res_cont_count: 蛋白残基连续特征数量
            m_dim_scalar: EGNN 消息维度
            dropout_rate: Dropout 比例
            num_rbf: RBF 基函数数量
            r_cutoff: 截断距离（单位：Å）
            fix_protein: 是否冻结蛋白坐标（刚性对接）
        """

        super().__init__()

        self.encoder = EHFEncoder(
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_gnn_blocks=num_gnn_blocks,
            lig_atom_cont_count=lig_atom_cont_count,
            lig_mol_cont_count=lig_mol_cont_count,
            pro_atom_cont_count=pro_atom_cont_count,
            pro_res_cont_count=pro_res_cont_count,
            m_dim_scalar=m_dim_scalar,
            dropout_rate=dropout_rate,
            fix_protein=fix_protein,
        )

        self.prediction_head = PredictionHead(
            hidden_dim=hidden_dim,
            num_rbf=num_rbf,
            r_cutoff=r_cutoff,
            dropout_rate=dropout_rate,
        )

        # 上下文感知门控网络
        # 为每个原子动态决定 EGNN 和 PredictionHead 的混合权重
        # 输入：原子特征（已融合时间嵌入）
        # 输出：每个原子的 alpha ∈ [0, 1]
        #   - alpha ≈ 1: 信任 EGNN（保守的几何预测）
        #   - alpha ≈ 0: 信任 PredictionHead（物理力场）
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # 速度分解器，用于将原子速度分解为平移、旋转、扭转分量
        self.decomposer = VelocityDecomposer()


    def forward(self, data: HeteroData, t: Tensor) -> EHFNetOutput:
        """
        前向传播

        Args:
            data: 异构图数据
            t: 时间步 [B]

        Returns:
            模型输出字典
        """

        # 1. 编码器
        ctx = self.encoder(data, t)
        x_dict = ctx["x_dict"]
        pos_dict = ctx["pos_dict"]
        vel_dict = ctx["vel_dict"]
        pos_initial = ctx["initial_ligand_pos"]

        # 2. 从 encoder 获取粗略速度 v_coarse（EGNN 隐式速度）
        v_coarse = vel_dict["ligand_atom"]

        # 3. 预测头获取物理修正 v_correction（显式力场）
        predictions = self.prediction_head(
            lig_atom_feat=x_dict["ligand_atom"],
            lig_atom_pos=pos_dict["ligand_atom"],
            lig_batch=data["ligand_atom"].batch,
            pro_atom_feat=x_dict["protein_atom"],
            pro_atom_pos=pos_dict["protein_atom"],
            pro_atom_batch=data["protein_atom"].batch,
            lig_mol_feat=x_dict["ligand_molecule"],
        )

        v_correction = predictions["v_atomic"]
        binding_affinity = predictions["binding_affinity"]
        force_atomic = predictions["force_atomic"]

        # 4. 上下文感知门控融合
        h_ligand = x_dict["ligand_atom"]  # [N_lig, hidden_dim]
        alpha = self.fusion_gate(h_ligand)  # [N_lig, 1]
        
        # 按原子加权融合
        # alpha ≈ 1: 主要使用 EGNN 的保守预测（适合刚性骨架）
        # alpha ≈ 0: 主要使用 PredictionHead 的物理力场（适合极性基团）
        v_atomic = alpha * v_coarse + (1 - alpha) * v_correction

        # 5. 使用联合优化分解速度
        masses = data["ligand_atom"].masses
        batch = data["ligand_atom"].batch
        device = v_atomic.device

        torsion_indices = getattr(
            data,
            "torsion_indices",
            torch.empty((0, 4), dtype=torch.long, device=device),
        )
        torsion_moving_mask = getattr(
            data,
            "torsion_moving_mask",
            torch.empty(
                (0, pos_dict["ligand_atom"].size(0)), dtype=torch.bool, device=device
            ),
        )

        v_translation, v_rotation, v_torsion = self.decomposer.decompose(
            pos=pos_dict["ligand_atom"],
            vel=v_atomic,
            masses=masses,
            batch=batch,
            torsion_indices=torsion_indices,
            torsion_moving_mask=torsion_moving_mask,
        )

        if v_torsion is not None and v_torsion.numel() > 0:
            v_torsion = v_torsion.unsqueeze(-1)
            
        else:
            v_torsion = None

        return {
            "x_dict": x_dict,
            "v_atomic": v_atomic,
            "v_coarse": v_coarse,
            "v_correction": v_correction,
            "alpha": alpha.mean(),
            "v_translation": v_translation,
            "v_rotation": v_rotation,
            "v_torsion": v_torsion,
            "binding_affinity": binding_affinity,
            "force_atomic": force_atomic,
            "pos_updated": pos_dict["ligand_atom"],
        }

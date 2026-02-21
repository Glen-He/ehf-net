"""
EHFNet 主模型

结合分层 EGNN 编码器与统一预测头。
"""

import torch

from typing import TypedDict
from torch import nn, Tensor
from torch_geometric.data import HeteroData
from torch_scatter import scatter_mean

from ehfnet.models.layers.encoder import EHFEncoder
from ehfnet.models.heads.prediction import PredictionHead


class EHFNetOutput(TypedDict):
    """
    EHFNet 模型输出类型定义
    """

    x_dict: dict[str, Tensor]       # 节点特征字典
    v_translation: Tensor           # 刚体平移速度 [B, 3]
    v_rotation: Tensor              # 刚体旋转速度 [B, 3]
    v_torsion: Tensor               # 扭转角速度 [T] （numel 可为 0）
    binding_affinity: Tensor        # 结合能 [B, 1]
    force_atomic: Tensor            # 原子级力场 [N, 3]
    steric_clash_batch: Tensor | None  # 每分子位阻惩罚量 [B]，无边时为 None
    pos_updated: Tensor             # 更新后的坐标 [N, 3]


class EHFNet(nn.Module):
    """
    EHFNet 顶层模型

    架构：
    1. 调用 EHFEncoder 获取原子层级特征
    2. 直接生成 SE(3) 切空间向量：
       - v_translation [B, 3]：平移速度（MLP 开环分子级特征）
       - v_rotation [B, 3]：旋转速度（局角速度5轴）
       - v_torsion [T]：每根扮转键的标量角速度
    3. 调用 PredictionHead 预测结合亲和力 (辅助信号)
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
        normalization_stats: dict | None = None,
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
            normalization_stats: 归一化统计数据
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
            stats=normalization_stats,
        )

        self.prediction_head = PredictionHead(
            hidden_dim=hidden_dim,
            num_rbf=num_rbf,
            r_cutoff=r_cutoff,
            dropout_rate=dropout_rate,
            affinity_stats=normalization_stats.get("affinity") if normalization_stats else None,
        )



        # 扭转 readout：每个扭转键的标量角速度
        # 输入: [T, H*2] (键轴两端原子特征拼接)
        self.torsion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )


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
        x_dict  = ctx["x_dict"]
        pos_dict = ctx["pos_dict"]
        vel_dict = ctx["vel_dict"]           # pos_final - pos_init，等变向量
        initial_lig_pos = ctx["initial_ligand_pos"]

        lig_atom_feat = x_dict["ligand_atom"]     # [N_lig, H]
        lig_mol_feat  = x_dict["ligand_molecule"] # [B, H]
        lig_batch     = data["ligand_atom"].batch  # [N_lig]
        lig_vel       = vel_dict["ligand_atom"]    # [N_lig, 3] — 等变位移向量

        # 2. 等变宏观运动读出
        B = lig_mol_feat.shape[0]

        # 平移：各原子位移方向的质心均值
        v_translation = scatter_mean(lig_vel, lig_batch, dim=0, dim_size=B)  # [B, 3]

        # 旋转：原子位移方向产生的角动量均值 L = r × v_trans
        com_init  = scatter_mean(initial_lig_pos, lig_batch, dim=0, dim_size=B)  # [B, 3]
        r         = initial_lig_pos - com_init[lig_batch]                        # [N, 3]
        L         = torch.linalg.cross(r, lig_vel)                               # [N, 3]
        v_rotation = scatter_mean(L, lig_batch, dim=0, dim_size=B)               # [B, 3]

        # 3. 扭转角速度（角度是旋转不变量，MLP 读出合法）
        device = lig_atom_feat.device
        torsion_indices = getattr(
            data,
            "torsion_indices",
            torch.empty((0, 4), dtype=torch.long, device=device),
        )

        if torsion_indices.numel() > 0 and torsion_indices.size(0) > 0:
            a1_feat = lig_atom_feat[torsion_indices[:, 1]]   # [T, H]
            a2_feat = lig_atom_feat[torsion_indices[:, 2]]   # [T, H]
            bond_feat = torch.cat([a1_feat, a2_feat], dim=-1)    # [T, 2H]
            v_torsion = self.torsion_head(bond_feat).squeeze(-1) # [T]
        else:
            v_torsion = torch.zeros(0, device=device, dtype=lig_mol_feat.dtype)

        # 4. PredictionHead：使用初始坐标规避 EGNN 坐标漂移
        predictions = self.prediction_head(
            lig_atom_feat=lig_atom_feat,
            lig_atom_pos=initial_lig_pos,              # 使用编码前的初始坐标
            lig_batch=lig_batch,
            pro_atom_feat=x_dict["protein_atom"],
            pro_atom_pos=data["protein_atom"].pos,     # 始终使用原始蛋白坐标
            pro_atom_batch=data["protein_atom"].batch,
            lig_mol_feat=lig_mol_feat,
        )

        binding_affinity = predictions["binding_affinity"]
        force_atomic     = predictions["force_atomic"]
        steric_clash_batch = predictions.get("steric_clash_batch")

        return {
            "x_dict": x_dict,
            "v_translation": v_translation,
            "v_rotation": v_rotation,
            "v_torsion": v_torsion,
            "binding_affinity": binding_affinity,
            "force_atomic": force_atomic,
            "steric_clash_batch": steric_clash_batch,
            "pos_updated": pos_dict["ligand_atom"],
        }

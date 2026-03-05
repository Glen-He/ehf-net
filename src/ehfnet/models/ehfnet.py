"""
EHFNet 主模型

结合分层 EGNN 编码器与统一预测头。
"""

import torch
import torch.nn.functional as F

from typing import TypedDict
from torch import nn, Tensor
from torch_geometric.data import HeteroData
from torch_scatter import scatter_mean

from ehfnet.geometry.dynamics import compute_principal_frame
from ehfnet.models.layers.encoder import EHFEncoder
from ehfnet.models.heads.prediction import PredictionHead


class EHFNetOutput(TypedDict):
    """
    EHFNet 模型输出类型定义
    """

    x_dict: dict[str, Tensor]               # 节点特征字典
    v_translation: Tensor                   # 刚体平移速度 [B, 3]
    v_rotation: Tensor                      # 刚体旋转速度 [B, 3]
    v_torsion: Tensor                       # 扭转角速度 [T] （numel 可为 0）
    binding_affinity: Tensor                # 结合能 [B, 1]
    steric_clash_batch: Tensor | None       # 每分子位阻惩罚量 [B]，无边时为 None


class EHFNet(nn.Module):
    """
    EHFNet 顶层模型

    编码器获取原子特征，readout 生成 SE(3) 切空间速度与亲和力预测。
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

        # 扭转角速度 readout [T, H*2+4] → [T]
        # 输入：键两端原子特征(H*2) + 几何特征(4: sin/cos当前二面角 + 键长 + 移动侧原子数)
        # 增强几何感知，容量加深至 3 层
        self.torsion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 体帧旋转速度 readout [B, H] → [B, 3]
        # 直接输出轴角向量（方向 × 角速度），不做 direction/magnitude 分离
        # 避免 softplus 在小角度时的梯度消失问题
        self.rot_body_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        # 平移幅度 readout [B, H] → [B, 1]
        self.trans_scale_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # [新增] 体帧平移方向 readout [B, H] → [B, 3]
        # MLP 在主惯量帧（body frame）中预测方向，经 R_frame 投射回世界帧保证等变性
        self.trans_body_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        # [新增] EGNN-MLP 融合门控 [B, H] → [B, 1]
        # 自适应选择：t→1 时 EGNN 位移信号强偏 EGNN，t≈0 时偏 MLP
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
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
        vel_dict = ctx["vel_dict"]                          # pos_final - pos_init，等变向量
        initial_lig_pos = ctx["initial_ligand_pos"]

        lig_atom_feat = x_dict["ligand_atom"]               # [N_lig, H]
        lig_mol_feat  = x_dict["ligand_molecule"]           # [B, H]
        lig_batch     = data["ligand_atom"].batch           # [N_lig]
        lig_vel       = vel_dict["ligand_atom"]             # [N_lig, 3] — 等变位移向量

        # 2. 等变宏观运动读出
        B = lig_mol_feat.shape[0]

        # 共享：计算主惯量帧（平移和旋转共用）
        masses = getattr(data["ligand_atom"], "masses", None)

        if masses is None:
            masses = torch.ones(initial_lig_pos.shape[0],
                                device=initial_lig_pos.device,
                                dtype=initial_lig_pos.dtype)

        R_frame = compute_principal_frame(
            initial_lig_pos, lig_batch, masses, dim_size=B
        )                                                                               # [B, 3, 3]

        # 平移：Hybrid Fusion（EGNN 物理先验 + MLP 体帧方向，门控融合）
        # 分支 1：EGNN 等变位移信号（物理先验）
        v_com_raw = scatter_mean(lig_vel, lig_batch, dim=0, dim_size=B)                 # [B, 3] Equivariant

        # 分支 2：MLP 体帧方向 → 世界帧等变向量（严格保持 SE(3) 等变性）
        v_body_trans = self.trans_body_head(lig_mol_feat)                               # [B, 3] Invariant
        v_mlp_trans = (R_frame @ v_body_trans.unsqueeze(-1)).squeeze(-1)                # [B, 3] Equivariant

        # 门控融合：网络自适应选择信任 EGNN 还是 MLP
        gate = torch.sigmoid(self.fusion_gate(lig_mol_feat))                            # [B, 1] in (0,1)
        trans_scale = F.softplus(self.trans_scale_head(lig_mol_feat))                   # [B, 1]
        v_translation = (gate * v_com_raw + (1.0 - gate) * v_mlp_trans) * trans_scale   # [B, 3]

        # 旋转：体帧 MLP → 世界帧等变角速度
        # 直接输出轴角向量，无 softplus / normalize 约束
        omega_body = self.rot_body_head(lig_mol_feat)                                   # [B, 3]
        v_rotation = (R_frame @ omega_body.unsqueeze(-1)).squeeze(-1)                   # [B, 3]

        # 3. 扭转角速度
        device = lig_atom_feat.device
        torsion_indices = getattr(
            data,
            "torsion_indices",
            torch.empty((0, 4), dtype=torch.long, device=device),
        )

        if torsion_indices.numel() > 0 and torsion_indices.size(0) > 0:
            a1_feat = lig_atom_feat[torsion_indices[:, 1]]              # [T, H]
            a2_feat = lig_atom_feat[torsion_indices[:, 2]]              # [T, H]

            # 几何特征：当前二面角的 sin/cos + 键长 + 移动侧原子数
            lig_pos = data["ligand_atom"].pos
            idx0, idx1 = torsion_indices[:, 0], torsion_indices[:, 1]
            idx2, idx3 = torsion_indices[:, 2], torsion_indices[:, 3]

            # 二面角
            b0 = -(lig_pos[idx1] - lig_pos[idx0])
            b1_vec = F.normalize(lig_pos[idx2] - lig_pos[idx1], dim=-1, eps=1e-8)
            b2 = lig_pos[idx3] - lig_pos[idx2]
            v_perp = b0 - (b0 * b1_vec).sum(-1, keepdim=True) * b1_vec
            w_perp = b2 - (b2 * b1_vec).sum(-1, keepdim=True) * b1_vec
            cos_dih = (v_perp * w_perp).sum(-1)                        # [T]
            sin_dih = (torch.cross(b1_vec, v_perp, dim=-1) * w_perp).sum(-1)  # [T]

            # 键长（归一化到 ~1）
            bond_len = torch.norm(lig_pos[idx2] - lig_pos[idx1], dim=-1) / 1.5  # [T]

            # 移动侧原子数（归一化）
            tor_mask = getattr(data, "torsion_moving_mask", None)
            if tor_mask is not None and tor_mask.numel() > 0:
                n_moving = tor_mask.float().sum(-1) / max(lig_pos.shape[0], 1)  # [T]
            else:
                n_moving = torch.zeros(torsion_indices.size(0), device=device, dtype=lig_mol_feat.dtype)

            geo_feat = torch.stack([cos_dih, sin_dih, bond_len, n_moving], dim=-1)  # [T, 4]
            bond_feat = torch.cat([a1_feat, a2_feat, geo_feat], dim=-1)   # [T, 2H+4]
            v_torsion = self.torsion_head(bond_feat).squeeze(-1)        # [T]
        else:
            v_torsion = torch.zeros(0, device=device, dtype=lig_mol_feat.dtype)

        # 4. 亲和力预测（使用初始坐标规避 EGNN 坐标漂移）
        predictions = self.prediction_head(
            lig_atom_feat=lig_atom_feat,
            lig_atom_pos=initial_lig_pos,
            lig_batch=lig_batch,
            pro_atom_feat=x_dict["protein_atom"],
            pro_atom_pos=data["protein_atom"].pos,
            pro_atom_batch=data["protein_atom"].batch,
            lig_mol_feat=lig_mol_feat,
        )

        binding_affinity = predictions["binding_affinity"]
        steric_clash_batch = predictions.get("steric_clash_batch")

        return {
            "x_dict": x_dict,
            "v_translation": v_translation,
            "v_rotation": v_rotation,
            "v_torsion": v_torsion,
            "binding_affinity": binding_affinity,
            "steric_clash_batch": steric_clash_batch,
        }

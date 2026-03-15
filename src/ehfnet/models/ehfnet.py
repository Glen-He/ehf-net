"""
EHFNet 主模型。

整合编码器、预测头与辅助分支，
输出对接过程所需的位姿和打分信号。
"""


from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.data import HeteroData
from torch_scatter import scatter_mean

from ehfnet.geometry import compute_center_of_mass, compute_principal_frame
from ehfnet.models.layers import EHFEncoder
from ehfnet.models.heads import PredictionHead


class EHFNetOutput(TypedDict):
    """
    EHFNet 输出结构。

    封装模型前向传播产生的主要预测张量，
    统一训练、验证和推理阶段对输出字段的访问方式。
    """

    x_dict: dict[str, Tensor]
    v_translation: Tensor
    v_rotation: Tensor
    v_torsion: Tensor
    binding_affinity: Tensor
    pose_quality: Tensor
    pose_rank_score: Tensor
    steric_clash_batch: Tensor | None


class EHFNet(nn.Module):
    """
    EHFNet 顶层模型。

    整合多层级编码器、几何更新分支和统一预测头，
    输出对接过程需要的位姿更新信号、排序信号和相关打分。
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
        m_dim_scalar: int,
        dropout_rate: float,
        num_rbf: int,
        r_cutoff: float,
        force_cutoff: float,
        interaction_profile: str,
        normalization_stats: dict | None = None,
        frame_refine_threshold: float,
        frame_refine_temperature: float,
        energy_guide_threshold: float,
        energy_guide_temperature: float,
        clash_threshold: float,
        clash_push_threshold: float,
        clash_push_force: float,
        score_clamp_min: float,
        score_clamp_max: float,
        force_limit: float,
        max_neighbors: int,
        min_max_neighbors: int,
        knn_fallback_k: int,
        dynamic_inter_cutoff: float,
        dynamic_inter_knn_k: int,
        dynamic_residue_cutoff: float,
        dynamic_residue_knn_k: int,
    ) -> None:
        """
        初始化 EHFNet 模型。

        配置编码器、预测头和相关几何参数，
        建立完整的两阶段盲对接主模型。

        Args:
            hidden_dim: 隐藏层维度。
            time_dim: 时间嵌入维度。
            num_gnn_blocks: 主干 GNN 块数量。
            lig_atom_cont_count: 配体原子连续特征维度。
            lig_mol_cont_count: 配体分子连续特征维度。
            pro_atom_cont_count: 蛋白原子连续特征维度。
            pro_res_cont_count: 蛋白残基连续特征维度。
            m_dim_scalar: 消息传递分支的标量维度。
            dropout_rate: Dropout 比例。
            num_rbf: RBF 基函数数量。
            r_cutoff: 几何邻域构建的距离截断半径。
            force_cutoff: 力相关分支使用的局部截断半径。
            interaction_profile: 跨图交互拓扑配置。
            normalization_stats: 输入特征归一化统计量。
            frame_refine_threshold: 主惯量帧细化门控阈值。
            frame_refine_temperature: 主惯量帧细化门控温度。
            energy_guide_threshold: 能量引导门控阈值。
            energy_guide_temperature: 能量引导门控温度。
            clash_threshold: 位阻判定阈值。
            clash_push_threshold: 位阻推开分支使用的距离阈值。
            clash_push_force: 位阻推开分支的力缩放系数。
            score_clamp_min: 分数裁剪下界。
            score_clamp_max: 分数裁剪上界。
            force_limit: 力大小的软限制。
            max_neighbors: 预测头阶段每个节点保留的最大邻居数。
            min_max_neighbors: 预测头动态邻居数的下限。
            knn_fallback_k: 回退到 kNN 时使用的邻居数。
            dynamic_inter_cutoff: 动态跨图原子边的半径阈值。
            dynamic_inter_knn_k: 动态跨图原子边回退到 kNN 时的邻居数。
            dynamic_residue_cutoff: 动态配体-残基边的半径阈值。
            dynamic_residue_knn_k: 动态配体-残基边回退到 kNN 时的邻居数。

        Raises:
            ValueError: 当输入参数或运行时状态不满足要求时抛出。
        """

        super().__init__()
        required_args = {
            "m_dim_scalar": m_dim_scalar,
            "dropout_rate": dropout_rate,
            "num_rbf": num_rbf,
            "r_cutoff": r_cutoff,
            "force_cutoff": force_cutoff,
            "interaction_profile": interaction_profile,
            "frame_refine_threshold": frame_refine_threshold,
            "frame_refine_temperature": frame_refine_temperature,
            "energy_guide_threshold": energy_guide_threshold,
            "energy_guide_temperature": energy_guide_temperature,
            "clash_threshold": clash_threshold,
            "clash_push_threshold": clash_push_threshold,
            "clash_push_force": clash_push_force,
            "score_clamp_min": score_clamp_min,
            "score_clamp_max": score_clamp_max,
            "force_limit": force_limit,
            "max_neighbors": max_neighbors,
            "min_max_neighbors": min_max_neighbors,
            "knn_fallback_k": knn_fallback_k,
            "dynamic_inter_cutoff": dynamic_inter_cutoff,
            "dynamic_inter_knn_k": dynamic_inter_knn_k,
            "dynamic_residue_cutoff": dynamic_residue_cutoff,
            "dynamic_residue_knn_k": dynamic_residue_knn_k,
        }
        missing_args = [name for name, value in required_args.items() if value is None]
        if missing_args:
            raise ValueError(
                "EHFNet is missing required explicit configuration values: "
                f"{missing_args}."
            )
        self.frame_refine_threshold = float(frame_refine_threshold)
        self.frame_refine_temperature = float(frame_refine_temperature)
        self.energy_guide_threshold = float(energy_guide_threshold)
        self.energy_guide_temperature = float(energy_guide_temperature)

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
            interaction_profile=interaction_profile,
            stats=normalization_stats,
            num_rbf=num_rbf,
            dynamic_inter_cutoff=dynamic_inter_cutoff,
            dynamic_inter_knn_k=dynamic_inter_knn_k,
            dynamic_residue_cutoff=dynamic_residue_cutoff,
            dynamic_residue_knn_k=dynamic_residue_knn_k,
        )

        self.prediction_head = PredictionHead(
            hidden_dim=hidden_dim,
            num_rbf=num_rbf,
            r_cutoff=r_cutoff,
            force_cutoff=force_cutoff,
            dropout_rate=dropout_rate,
            clash_threshold=clash_threshold,
            clash_push_threshold=clash_push_threshold,
            clash_push_force=clash_push_force,
            score_clamp_min=score_clamp_min,
            score_clamp_max=score_clamp_max,
            force_limit=force_limit,
            max_neighbors=max_neighbors,
            min_max_neighbors=min_max_neighbors,
            knn_fallback_k=knn_fallback_k,
        )

        self.torsion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.torsion_refine_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.rot_body_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        self.rot_scale_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.trans_scale_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.trans_body_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.energy_guidance_gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.energy_trans_refine_scale = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.energy_rot_refine_scale = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.center_proposal_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )


    def predict_center_logits(
        self,
        *,
        residue_x_cat: Tensor,
        residue_x_cont: Tensor,
        residue_pos: Tensor,
        residue_batch: Tensor,
        lig_mol_x_cont: Tensor,
        residue_esm_missing_mask: Tensor | None = None,
        residue_prior_feat: Tensor | None = None,
    ) -> Tensor:
        """
        预测残基层中心提议分数。

        在不执行完整对接前向的情况下仅运行中心提议相关分支，
        用于候选中心选择阶段的快速打分。

        Args:
            residue_x_cat: 残基分类特征张量。
            residue_x_cont: 残基连续特征张量。
            residue_pos: 残基坐标张量。
            residue_batch: 残基所属 batch 索引。
            lig_mol_x_cont: 配体分子级连续特征张量。
            residue_esm_missing_mask: 标记残基 ESM 是否缺失的布尔掩码。
            residue_prior_feat: 残基级几何先验特征。

        Returns:
            Tensor: 返回计算得到的张量结果。
        """
        residue_feat = self.encoder.protein_residue_embedder(
            residue_x_cat,
            residue_x_cont,
            esm_missing_mask=residue_esm_missing_mask,
        )
        lig_feat = self.encoder.ligand_molecule_embedder(lig_mol_x_cont)

        if lig_feat.size(0) == 0 or residue_feat.size(0) == 0:
            return residue_feat.new_zeros((residue_feat.size(0), 1))

        protein_context = scatter_mean(
            residue_feat,
            residue_batch,
            dim=0,
            dim_size=lig_feat.size(0),
        )
        protein_center = scatter_mean(
            residue_pos,
            residue_batch,
            dim=0,
            dim_size=lig_feat.size(0),
        )
        rel_pos = residue_pos - protein_center[residue_batch]
        rel_norm = torch.norm(rel_pos, dim=-1, keepdim=True)
        if residue_prior_feat is None:
            residue_prior_feat = residue_pos.new_zeros((residue_pos.size(0), 4))

        proposal_input = torch.cat(
            [
                residue_feat,
                protein_context[residue_batch],
                lig_feat[residue_batch],
                rel_pos,
                rel_norm,
                residue_prior_feat,
            ],
            dim=-1,
        )
        return self.center_proposal_head(proposal_input)


    def forward(self, data: HeteroData, t: Tensor) -> EHFNetOutput:
        """
        前向传播

        Args:
            data: 当前处理的图数据对象。
            t: t。

        Returns:
            模型输出字典
        """

        ctx = self.encoder(data, t)
        x_dict = ctx["x_dict"]
        pos_dict = ctx["pos_dict"]
        displacement_dict = ctx["displacement_dict"]
        initial_lig_pos = ctx["initial_ligand_pos"]
        current_lig_pos = pos_dict["ligand_atom"]

        lig_atom_feat = x_dict["ligand_atom"]
        lig_mol_feat = x_dict["ligand_molecule"]
        lig_batch = data["ligand_atom"].batch
        lig_displacement = displacement_dict["ligand_atom"]
        B = lig_mol_feat.shape[0]

        masses = getattr(data["ligand_atom"], "masses", None)

        if masses is None:
            masses = torch.ones(initial_lig_pos.shape[0],
                                device=initial_lig_pos.device,
                                dtype=initial_lig_pos.dtype)

        R_frame_initial = compute_principal_frame(
            initial_lig_pos, lig_batch, masses, dim_size=B
        )
        R_frame_current = compute_principal_frame(
            current_lig_pos.detach(), lig_batch, masses, dim_size=B
        )
        frame_refine_gate = torch.sigmoid(
            (t.view(B, 1, 1) - self.frame_refine_threshold)
            / max(self.frame_refine_temperature, 1e-6)
        )

        v_com_raw = scatter_mean(lig_displacement, lig_batch, dim=0, dim_size=B)
        v_body_trans = self.trans_body_head(lig_mol_feat)
        v_mlp_trans_initial = (R_frame_initial @ v_body_trans.unsqueeze(-1)).squeeze(-1)
        v_mlp_trans_current = (R_frame_current @ v_body_trans.unsqueeze(-1)).squeeze(-1)
        v_mlp_trans = (
            (1.0 - frame_refine_gate.squeeze(-1)) * v_mlp_trans_initial
            + frame_refine_gate.squeeze(-1) * v_mlp_trans_current
        )

        gate = torch.sigmoid(self.fusion_gate(lig_mol_feat))
        trans_scale = F.softplus(self.trans_scale_head(lig_mol_feat))
        v_translation = (gate * v_com_raw + (1.0 - gate) * v_mlp_trans) * trans_scale

        omega_dir = F.normalize(self.rot_body_head(lig_mol_feat), dim=-1, eps=1e-8)
        rot_scale = F.softplus(self.rot_scale_head(lig_mol_feat))
        omega_body = omega_dir * rot_scale
        v_rotation_initial = (R_frame_initial @ omega_body.unsqueeze(-1)).squeeze(-1)
        v_rotation_current = (R_frame_current @ omega_body.unsqueeze(-1)).squeeze(-1)
        v_rotation = (
            (1.0 - frame_refine_gate.squeeze(-1)) * v_rotation_initial
            + frame_refine_gate.squeeze(-1) * v_rotation_current
        )

        device = lig_atom_feat.device
        torsion_indices = getattr(
            data,
            "torsion_indices",
            torch.empty((0, 4), dtype=torch.long, device=device),
        )

        guidance_input = torch.cat([lig_mol_feat, t.view(B, 1)], dim=-1)
        learned_refine_gate = torch.sigmoid(self.energy_guidance_gate(guidance_input))
        time_refine_gate = torch.sigmoid(
            (t.view(B, 1) - self.energy_guide_threshold)
            / max(self.energy_guide_temperature, 1e-6)
        )
        refine_gate = learned_refine_gate * time_refine_gate

        predictions = self.prediction_head(
            lig_atom_feat=lig_atom_feat,
            lig_atom_pos=current_lig_pos,
            lig_batch=lig_batch,
            pro_atom_feat=x_dict["protein_atom"],
            pro_atom_pos=data["protein_atom"].pos,
            pro_atom_batch=data["protein_atom"].batch,
            lig_mol_feat=lig_mol_feat,
        )

        interaction_context = predictions.get("ligand_interaction_context")
        if interaction_context is None:
            interaction_context = torch.zeros_like(lig_atom_feat)

        force_atom = predictions.get("ligand_force")
        if force_atom is None:
            force_atom = torch.zeros_like(current_lig_pos)

        if force_atom.numel() > 0:
            current_com = compute_center_of_mass(current_lig_pos, lig_batch, masses, dim_size=B)
            rel_to_com = current_lig_pos - current_com[lig_batch]

            force_trans = scatter_mean(force_atom, lig_batch, dim=0, dim_size=B)
            force_torque = scatter_mean(torch.cross(rel_to_com, force_atom, dim=-1), lig_batch, dim=0, dim_size=B)

            force_trans_norm = torch.norm(force_trans, dim=-1, keepdim=True)
            force_torque_norm = torch.norm(force_torque, dim=-1, keepdim=True)

            trans_refine = F.normalize(force_trans, dim=-1, eps=1e-8) * torch.tanh(force_trans_norm / 5.0)
            rot_refine = F.normalize(force_torque, dim=-1, eps=1e-8) * torch.tanh(force_torque_norm / 5.0)

            trans_refine = trans_refine * F.softplus(self.energy_trans_refine_scale(guidance_input))
            rot_refine = rot_refine * F.softplus(self.energy_rot_refine_scale(guidance_input))

            v_translation = v_translation + refine_gate * trans_refine
            v_rotation = v_rotation + refine_gate * rot_refine

        if torsion_indices.numel() > 0 and torsion_indices.size(0) > 0:
            a1_feat = lig_atom_feat[torsion_indices[:, 1]]
            a2_feat = lig_atom_feat[torsion_indices[:, 2]]
            bond_feat = torch.cat([a1_feat, a2_feat], dim=-1)
            ctx1 = interaction_context[torsion_indices[:, 1]]
            ctx2 = interaction_context[torsion_indices[:, 2]]
            bond_refine_feat = torch.cat([a1_feat, a2_feat, ctx1, ctx2], dim=-1)
            torsion_batch = lig_batch[torsion_indices[:, 1]]
            torsion_refine_gate = refine_gate[torsion_batch].view(-1)

            base_torsion = self.torsion_head(bond_feat).squeeze(-1)
            refine_torsion = self.torsion_refine_head(bond_refine_feat).squeeze(-1)
            v_torsion = base_torsion + torsion_refine_gate * refine_torsion
        else:
            v_torsion = torch.zeros(0, device=device, dtype=lig_mol_feat.dtype)

        binding_affinity = predictions["binding_affinity"]
        pose_quality = predictions["pose_quality"]
        pose_rank_score = predictions["pose_rank_score"]
        steric_clash_batch = predictions.get("steric_clash_batch")

        return {
            "x_dict": x_dict,
            "v_translation": v_translation,
            "v_rotation": v_rotation,
            "v_torsion": v_torsion,
            "binding_affinity": binding_affinity,
            "pose_quality": pose_quality,
            "pose_rank_score": pose_rank_score,
            "steric_clash_batch": steric_clash_batch,
        }

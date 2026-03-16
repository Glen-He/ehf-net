"""
中心采样工具。

负责训练裁剪中心、错误中心与 bootstrap 中心的采样策略，
为局部训练和候选学习提供中心输入。
"""


from typing import Any

import torch
import torch.nn.functional as F

from ehfnet.graph import GraphCollator
from ehfnet.training.batch_helpers import build_local_batch_from_centers, compute_pose_rank_target
from ehfnet.training.flow_matcher import ConditionalFlowMatcher


def select_training_crop_centers(
    ligand_centers: torch.Tensor,
    proposal_logits: torch.Tensor,
    residue_pos: torch.Tensor,
    residue_batch: torch.Tensor,
    *,
    progress: float,
    positive_radius: float,
    bucket_topk: int,
    weighted_sampling: bool,
    disable_jitter: bool,
    disable_hard_negative: bool,
    proposal_start: float,
    near_miss_start: float,
    hard_negative_start: float,
) -> tuple[torch.Tensor, list[str]]:
    """
    采样训练裁剪中心。

    按照课程学习进度从不同难度桶中选择中心点，
    用于训练阶段的局部裁剪与中心监督。

    Args:
        ligand_centers: 当前 batch 中每个样本的真实配体中心。
        proposal_logits: 中心提议分支输出的 logits。
        residue_pos: 残基坐标张量。
        residue_batch: 残基所属 batch 索引。
        progress: 课程学习当前进度，通常归一化到 0 到 1。
        positive_radius: 判定正中心的距离半径。
        bucket_topk: 每个候选桶最多保留的中心数。
        weighted_sampling: 是否按分数执行加权采样。
        disable_jitter: 是否关闭中心扰动。
        disable_hard_negative: 是否关闭 hard negative 中心采样。
        proposal_start: `proposal_pos` 进入课程的训练进度阈值。
        near_miss_start: `near_miss` 进入课程的训练进度阈值。
        hard_negative_start: `hard_neg` 进入课程的训练进度阈值。

    Returns:
        tuple[Tensor, list[str]]: 训练裁剪中心张量 [B, 3] 与各样本的采样模式列表。
    """
    progress = float(max(0.0, min(1.0, progress)))
    proposal_active = progress >= float(max(0.0, min(1.0, proposal_start)))
    near_miss_active = progress >= float(max(0.0, min(1.0, near_miss_start)))
    hard_negative_active = progress >= float(max(0.0, min(1.0, hard_negative_start)))
    stage_weights = {
        "gt": max(0.10, 0.50 - 0.40 * progress),
        "jitter": max(0.15, 0.30 - 0.10 * progress),
        "proposal_pos": (0.10 + 0.20 * progress) if proposal_active else 0.0,
        "near_miss": (0.05 + 0.15 * progress) if near_miss_active else 0.0,
        "hard_neg": (
            max(0.0, -0.05 + 0.30 * progress) if hard_negative_active else 0.0
        ),
    }
    if disable_jitter:
        stage_weights["jitter"] = 0.0
    if disable_hard_negative:
        stage_weights["hard_neg"] = 0.0
    jitter_sigma = 2.0 + 6.0 * progress
    chosen_centers: list[torch.Tensor] = []
    chosen_modes: list[str] = []
    num_graphs = int(ligand_centers.size(0))

    def _sample_from_bucket(bucket_pos: torch.Tensor, bucket_logits: torch.Tensor) -> torch.Tensor:
        if bucket_pos.size(0) == 1:
            return bucket_pos[0]
        k = min(max(1, bucket_topk), bucket_pos.size(0))
        pool_pos = bucket_pos[:k]
        pool_logits = bucket_logits[:k]
        if not weighted_sampling:
            choice_idx = torch.randint(k, (1,), device=pool_pos.device).item()
            return pool_pos[choice_idx]
        weight = torch.softmax(pool_logits, dim=0)
        choice_idx = int(torch.multinomial(weight, 1).item())
        return pool_pos[choice_idx]

    for graph_idx in range(num_graphs):
        gt_center = ligand_centers[graph_idx]
        mask = residue_batch == graph_idx
        graph_pos = residue_pos[mask]
        graph_logits = proposal_logits[mask].view(-1)
        if graph_pos.numel() == 0:
            chosen_centers.append(gt_center)
            chosen_modes.append("gt_fallback")
            continue

        order = torch.argsort(graph_logits, descending=True)
        ordered_pos = graph_pos[order]
        ordered_logits = graph_logits[order]
        ordered_dist = torch.norm(ordered_pos - gt_center.unsqueeze(0), dim=-1)
        positive_mask = ordered_dist <= positive_radius
        near_mask = (ordered_dist > positive_radius) & (ordered_dist <= positive_radius * 2.0)
        hard_mask = ordered_dist > positive_radius * 2.0

        bucket_to_center: dict[str, torch.Tensor] = {
            "gt": gt_center,
        }
        if not disable_jitter:
            bucket_to_center["jitter"] = gt_center + torch.randn_like(gt_center) * jitter_sigma
        if proposal_active and positive_mask.any():
            pos_pool = ordered_pos[positive_mask]
            pos_logits = ordered_logits[positive_mask]
            bucket_to_center["proposal_pos"] = _sample_from_bucket(pos_pool, pos_logits)
        if near_miss_active and near_mask.any():
            near_pool = ordered_pos[near_mask]
            near_logits = ordered_logits[near_mask]
            bucket_to_center["near_miss"] = _sample_from_bucket(near_pool, near_logits)
        if hard_negative_active and hard_mask.any() and not disable_hard_negative:
            hard_pool = ordered_pos[hard_mask]
            hard_logits = ordered_logits[hard_mask]
            bucket_to_center["hard_neg"] = _sample_from_bucket(hard_pool, hard_logits)

        available_modes = list(bucket_to_center.keys())
        weight_tensor = torch.tensor(
            [stage_weights.get(mode, 0.0) for mode in available_modes],
            dtype=ligand_centers.dtype,
            device=ligand_centers.device,
        )
        if float(weight_tensor.sum().item()) <= 0.0:
            chosen_mode = "gt"
        else:
            chosen_mode = available_modes[int(torch.multinomial(weight_tensor / weight_tensor.sum(), 1).item())]

        chosen_centers.append(bucket_to_center[chosen_mode])
        chosen_modes.append(chosen_mode)

    return torch.stack(chosen_centers, dim=0), chosen_modes


def select_wrong_center_candidates(
    ligand_centers: torch.Tensor,
    proposal_logits: torch.Tensor,
    residue_pos: torch.Tensor,
    residue_batch: torch.Tensor,
    *,
    positive_radius: float,
    bucket_topk: int,
    weighted_sampling: bool,
    allow_negative_centers: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    采样错误中心候选。

    从 near-miss 或 hard-negative 区域中挑选错误中心，
    为 hard ranking 和 bootstrap 提供更具挑战性的样本。

    Args:
        ligand_centers: 当前 batch 中每个样本的真实配体中心。
        proposal_logits: 中心提议分支输出的 logits。
        residue_pos: 残基坐标张量。
        residue_batch: 残基所属 batch 索引。
        positive_radius: 判定正中心的距离半径。
        bucket_topk: 每个候选桶最多保留的中心数。
        weighted_sampling: 是否按分数执行加权采样。
        allow_negative_centers: 是否允许采样负例中心。

    Returns:
        tuple[Tensor, Tensor, Tensor]: 错误中心、对应分数和有效掩码。
    """
    num_graphs = int(ligand_centers.size(0))
    wrong_centers: list[torch.Tensor] = []
    wrong_center_scores: list[torch.Tensor] = []
    valid_mask: list[bool] = []

    def _sample_bucket(bucket_pos: torch.Tensor, bucket_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = min(max(1, bucket_topk), bucket_pos.size(0))
        pool_pos = bucket_pos[:k]
        pool_logits = bucket_logits[:k]
        if pool_pos.size(0) == 1:
            return pool_pos[0], pool_logits[0]
        if not weighted_sampling:
            idx = int(torch.randint(pool_pos.size(0), (1,), device=pool_pos.device).item())
            return pool_pos[idx], pool_logits[idx]
        prob = torch.softmax(pool_logits, dim=0)
        idx = int(torch.multinomial(prob, 1).item())
        return pool_pos[idx], pool_logits[idx]

    for graph_idx in range(num_graphs):
        gt_center = ligand_centers[graph_idx]
        if not allow_negative_centers:
            wrong_centers.append(gt_center)
            wrong_center_scores.append(gt_center.new_zeros(()))
            valid_mask.append(False)
            continue
        mask = residue_batch == graph_idx
        graph_pos = residue_pos[mask]
        graph_logits = proposal_logits[mask].view(-1)
        if graph_pos.numel() == 0:
            wrong_centers.append(gt_center)
            wrong_center_scores.append(gt_center.new_zeros(()))
            valid_mask.append(False)
            continue
        order = torch.argsort(graph_logits, descending=True)
        ordered_pos = graph_pos[order]
        ordered_logits = graph_logits[order]
        dist = torch.norm(ordered_pos - gt_center.unsqueeze(0), dim=-1)
        near_mask = (dist > positive_radius) & (dist <= positive_radius * 2.0)
        hard_mask = dist > positive_radius * 2.0
        if near_mask.any():
            center, score = _sample_bucket(ordered_pos[near_mask], ordered_logits[near_mask])
            valid = True
        elif hard_mask.any():
            center, score = _sample_bucket(ordered_pos[hard_mask], ordered_logits[hard_mask])
            valid = True
        else:
            center, score = gt_center, gt_center.new_zeros(())
            valid = False
        wrong_centers.append(center)
        wrong_center_scores.append(score)
        valid_mask.append(valid)

    return (
        torch.stack(wrong_centers, dim=0),
        torch.stack([score.view(1) for score in wrong_center_scores], dim=0).view(-1),
        torch.as_tensor(valid_mask, dtype=torch.bool, device=ligand_centers.device),
    )


def sample_hard_ranking_time_and_centers(
    t_anchor: torch.Tensor,
    crop_centers: torch.Tensor,
    wrong_centers: torch.Tensor,
    wrong_center_valid_mask: torch.Tensor,
    *,
    progress: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    采样 hard ranking 时间与中心。

    为排序训练选择时间步和候选中心组合，
    让模型在更困难的局部区域上学习区分能力。

    Args:
        t_anchor: hard ranking 采样使用的基准时间步。
        crop_centers: 当前已选中的裁剪中心。
        wrong_centers: 错误中心候选集合。
        wrong_center_valid_mask: 标记错误中心是否有效的布尔掩码。
        progress: 课程学习当前进度，通常归一化到 0 到 1。

    Returns:
        tuple[Tensor, Tensor]: hard ranking 使用的时间步与中心集合。
    """
    B = t_anchor.size(0)
    device = t_anchor.device

    strategy = torch.rand(B, device=device)
    t_hard = t_anchor.clone()
    hard_centers = crop_centers.clone()
    strategy_id = torch.zeros(B, device=device, dtype=torch.long)

    mask_worse_time = strategy < 0.45
    if mask_worse_time.any():
        n = int(mask_worse_time.sum().item())
        scale = 0.25 + 0.45 * torch.rand(n, device=device)
        t_hard[mask_worse_time] = t_anchor[mask_worse_time] * scale

    mask_offset = (~mask_worse_time) & wrong_center_valid_mask.to(device=device)
    if mask_offset.any():
        hard_centers[mask_offset] = wrong_centers[mask_offset]
        strategy_id[mask_offset] = 1
        scale = 0.65 + 0.25 * torch.rand(int(mask_offset.sum().item()), device=device)
        t_hard[mask_offset] = torch.clamp(t_anchor[mask_offset] * scale, max=0.85)

    sigma = 1e-3
    t_hard = t_hard.clamp(min=sigma, max=1.0 - sigma)
    return t_hard, hard_centers, strategy_id


def should_run_bootstrap(
    *,
    epoch: int,
    batch_idx: int,
    total_epochs: int,
    frequency: int,
    start_ratio: float,
) -> bool:
    """
    判断是否执行 bootstrap。

    根据当前训练轮次和 batch 位置决定是否触发 bootstrap 分支，
    避免在训练早期或频率过高时引入额外开销。

    Args:
        epoch: 当前训练轮次。
        batch_idx: 原子或样本所属的 batch 索引。
        total_epochs: 训练总轮数。
        frequency: bootstrap 或其他分支的触发频率。
        start_ratio: 开始启用该分支时对应的训练进度比例。

    Returns:
        bool: 返回布尔判断结果。
    """
    if frequency <= 0:
        return False
    progress = 1.0 if total_epochs <= 1 else epoch / max(1, total_epochs - 1)
    return progress >= start_ratio and batch_idx % frequency == 0


def select_bootstrap_blind_centers(
    ligand_centers: torch.Tensor,
    proposal_logits: torch.Tensor,
    residue_pos: torch.Tensor,
    residue_batch: torch.Tensor,
    *,
    positive_radius: float,
    bucket_topk: int,
    allow_negative_centers: bool,
) -> torch.Tensor:
    """
    选择 bootstrap 中心。

    从 blind 候选中心中挑选用于 bootstrap 训练的局部中心，
    兼顾错误中心与有效中心的采样比例。

    Args:
        ligand_centers: 当前 batch 中每个样本的真实配体中心。
        proposal_logits: 中心提议分支输出的 logits。
        residue_pos: 残基坐标张量。
        residue_batch: 残基所属 batch 索引。
        positive_radius: 判定正中心的距离半径。
        bucket_topk: 每个候选桶最多保留的中心数。
        allow_negative_centers: 是否允许采样负例中心。

    Returns:
        tuple[Tensor, Tensor]: bootstrap 使用的中心集合与有效掩码。
    """
    wrong_centers, _, wrong_valid_mask = select_wrong_center_candidates(
        ligand_centers,
        proposal_logits,
        residue_pos,
        residue_batch,
        positive_radius=positive_radius,
        bucket_topk=bucket_topk,
        weighted_sampling=True,
        allow_negative_centers=allow_negative_centers,
    )
    bootstrap_centers = ligand_centers.clone()
    if wrong_valid_mask.any():
        mix_mask = (torch.rand_like(wrong_valid_mask.float()) < 0.7) & wrong_valid_mask
        bootstrap_centers[mix_mask] = wrong_centers[mix_mask]
    return bootstrap_centers


def compute_bootstrap_pose_rank_loss(
    *,
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    matcher: ConditionalFlowMatcher,
    source_batch: Any,
    placement_centers: torch.Tensor,
    epoch: int,
    ode_steps: int,
    ode_method: str,
    graph_builder: Any,
    collator: GraphCollator,
    crop_radius: float,
    crop_min_residues: int,
    crop_atom_margin: float,
    dataset_raw_dir: str,
) -> torch.Tensor:
    """
    计算 bootstrap pose 排序损失。

    利用 teacher 轨迹构造监督目标，再由 student 预测局部排序分数，
    为单一排序头提供额外训练信号。

    Args:
        student_model: 待优化的 student 模型实例。
        teacher_model: 提供 teacher 监督的模型实例。
        matcher: 流匹配控制器或 ODE 推理控制器。
        source_batch: 作为源输入的批次对象。
        placement_centers: 指定初始放置或 bootstrap 使用的中心集合。
        epoch: 当前训练轮次。
        ode_steps: ODE 推理积分步数。
        ode_method: bootstrap teacher rollout 使用的 ODE 积分方法。
        graph_builder: 用于构图或重建局部图的图构建器。
        collator: 用于拼接局部样本的图批处理器。
        crop_radius: 局部裁剪半径。
        crop_min_residues: 局部裁剪后至少保留的残基数量。
        crop_atom_margin: 基于原子距离扩展残基裁剪范围的边界。
        dataset_raw_dir: 数据集原始样本目录。

    Returns:
        Tensor: bootstrap 排序损失标量。
    """
    blind_local_batch = build_local_batch_from_centers(
        source_batch,
        centers=placement_centers.detach().cpu(),
        crop_radius=crop_radius,
        crop_min_residues=crop_min_residues,
        crop_atom_margin=crop_atom_margin,
        graph_builder=graph_builder,
        collator=collator,
    )
    blind_local_samples = (
        blind_local_batch.to_data_list()
        if hasattr(blind_local_batch, "to_data_list")
        else [blind_local_batch]
    )
    device = next(student_model.parameters()).device
    blind_local_batch = blind_local_batch.to(device)
    x_ref = blind_local_batch["ligand_atom"].pos
    lig_batch = blind_local_batch["ligand_atom"].batch
    masses = blind_local_batch["ligand_atom"].masses
    B = int(lig_batch.max().item()) + 1
    with torch.no_grad():
        teacher_batch = blind_local_batch.clone()
        x0 = matcher._generate_random_pose(
            x_ref=x_ref,
            batch=lig_batch,
            B=B,
            masses=masses,
            torsion_indices=getattr(teacher_batch, "torsion_indices", None),
            torsion_moving_mask=getattr(teacher_batch, "torsion_moving_mask", None),
            seed_pos=teacher_batch["ligand_atom"].get("start_pos", None),
            protein_pos=teacher_batch["protein_atom"].pos,
            protein_batch=getattr(teacher_batch["protein_atom"], "batch", None),
            placement_centers=placement_centers,
            epoch=epoch,
        )
        teacher_batch["ligand_atom"].pos = x0
        final_pos, _ = matcher.ode_solve(
            model=teacher_model,
            data=teacher_batch,
            steps=ode_steps,
            method=ode_method,
            store_trajectory=False,
        )
        teacher_target = compute_pose_rank_target(
            final_pos,
            x_ref,
            batch_idx=lig_batch,
            samples=blind_local_samples,
            dataset_raw_dir=dataset_raw_dir,
        )

    student_batch = blind_local_batch.clone()
    student_batch["ligand_atom"].pos = final_pos.detach()
    student_batch.t = torch.ones(B, device=x_ref.device, dtype=x_ref.dtype)
    student_pred = student_model(student_batch, student_batch.t)
    pred_pose_rank = student_pred["pose_rank_score"].view(-1)
    target_pose_rank = teacher_target.view(-1).to(device=pred_pose_rank.device, dtype=pred_pose_rank.dtype)
    weight = 1.0 + 2.0 * target_pose_rank
    return F.binary_cross_entropy_with_logits(pred_pose_rank, target_pose_rank, weight=weight)

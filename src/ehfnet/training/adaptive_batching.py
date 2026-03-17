"""
自适应批处理工具。

负责样本级成本打包与 batch 拆分，
用于降低长尾图样本带来的显存波动与 OOM 风险。
"""


import random
from dataclasses import dataclass, field
from typing import Any

from torch.utils.data import Sampler, Subset
from torch_geometric.data import HeteroData

from ehfnet.graph import build_graph_cost_profile, estimate_graph_cost_units


def resolve_subset_root_indices(dataset_obj: Any) -> list[int]:
    """
    将任意层级的 `Subset` 映射回根数据集索引。

    Args:
        dataset_obj: 数据集对象或 `Subset`。

    Returns:
        list[int]: 与当前数据集顺序一致的根数据集索引列表。
    """
    if isinstance(dataset_obj, Subset):
        parent_indices = resolve_subset_root_indices(dataset_obj.dataset)
        return [int(parent_indices[int(local_idx)]) for local_idx in dataset_obj.indices]
    return list(range(len(dataset_obj)))


def split_collated_batch(
    batch: HeteroData,
    *,
    collator: Any,
) -> tuple[HeteroData, HeteroData] | None:
    """
    将已拼接 batch 按样本对半拆分。

    Args:
        batch: 已经 collate 完成的异构图 batch。
        collator: 当前训练使用的图拼接器。

    Returns:
        tuple[HeteroData, HeteroData] | None: 可拆分时返回左右两个子 batch；单样本时返回 `None`。
    """
    data_list = batch.to_data_list()
    if len(data_list) <= 1:
        return None
    mid = len(data_list) // 2
    if mid <= 0 or mid >= len(data_list):
        return None
    left = collator.collate(data_list[:mid])
    right = collator.collate(data_list[mid:])
    return left, right


def estimate_runtime_batch_cost(
    batch: HeteroData,
    *,
    num_gnn_blocks: int,
    dynamic_inter_max_neighbors: int,
    dynamic_residue_max_neighbors: int,
    dynamic_residue_candidate_topk: int,
    phase_multiplier: float,
) -> int:
    """
    估计已拼接 batch 的运行时成本。

    Args:
        batch: 已经 collate 完成的异构图 batch。
        num_gnn_blocks: 主干 GNN 块数量。
        dynamic_inter_max_neighbors: 动态原子跨图边的单源邻居上限。
        dynamic_residue_max_neighbors: 动态配体-残基跨图边的单源邻居上限。
        dynamic_residue_candidate_topk: 每个复合物保留的残基候选上限。
        phase_multiplier: 当前阶段的成本倍率。

    Returns:
        int: 当前 batch 的估计成本单位。
    """
    profile = build_graph_cost_profile(batch)
    return estimate_graph_cost_units(
        profile,
        num_gnn_blocks=num_gnn_blocks,
        dynamic_inter_max_neighbors=dynamic_inter_max_neighbors,
        dynamic_residue_max_neighbors=dynamic_residue_max_neighbors,
        dynamic_residue_candidate_topk=dynamic_residue_candidate_topk,
        phase_multiplier=phase_multiplier,
    )


def extract_root_dataset_indices(batch: Any) -> list[int]:
    """
    从 collate 后的 batch 中提取根数据集索引。

    Args:
        batch: 已经 collate 完成的异构图 batch，或单个样本。

    Returns:
        list[int]: 当前 batch 中每个根样本对应的底层数据集索引列表。
    """
    data_list = batch.to_data_list() if hasattr(batch, "to_data_list") else [batch]
    root_indices: list[int] = []
    for sample in data_list:
        dataset_index = getattr(sample, "dataset_index", None)
        if dataset_index is not None:
            root_indices.append(int(dataset_index))
    return root_indices


@dataclass(frozen=True)
class BudgetAdjustment:
    """
    预算状态机的一次回调结果。

    Args:
        phase_name: 所属阶段名称。
        action: 回调动作，支持 `reduce`、`recover`、`hold`。
        previous_budget: 调整前预算。
        new_budget: 调整后预算。
        window_total: 当前结算窗口内累计的根 batch 数。
        window_oom: 当前结算窗口内累计的 OOM 根 batch 数。
        offender_count: 当前仍处于冷却期的坏样本数量。
        reason: 触发本次回调的原因说明。
    """

    phase_name: str
    action: str
    previous_budget: int
    new_budget: int
    window_total: int
    window_oom: int
    offender_count: int
    reason: str


@dataclass
class WindowAimdBudgetController:
    """
    基于窗口统计的预算回调控制器。

    该控制器把 OOM 统计从“整 epoch 一刀切”改成
    “根 batch 去重 + 固定窗口结算 + AIMD 回调 + offender 冷却”。

    Args:
        phase_name: 阶段名称，仅用于日志。
        base_budget: 阶段的目标预算上限。
        min_budget: 预算允许降到的最小值。
        window_size: 统计窗口包含的根 batch 事件数。
        reduce_threshold: 单窗口内触发降档所需的最小 OOM 事件数。
        reduce_factor: 触发降档时的乘性缩放系数。
        recover_window_count: 连续多少个全干净窗口后触发一次回升。
        recover_step: 每次回升增加的预算步长。
        offender_cooldown: 坏样本冷却时长，按根 batch 事件递减。
        enable_adaptive: 是否允许控制器调整预算。
    """

    phase_name: str
    base_budget: int
    min_budget: int
    window_size: int
    reduce_threshold: int
    reduce_factor: float
    recover_window_count: int
    recover_step: int
    offender_cooldown: int
    enable_adaptive: bool = True
    current_budget: int = field(init=False)
    offender_cooldowns: dict[int, int] = field(default_factory=dict, init=False)
    _window_total: int = field(default=0, init=False)
    _window_oom: int = field(default=0, init=False)
    _clean_windows: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.base_budget = max(1, int(self.base_budget))
        self.min_budget = max(1, min(int(self.min_budget), self.base_budget))
        self.window_size = max(1, int(self.window_size))
        self.reduce_threshold = max(1, int(self.reduce_threshold))
        self.reduce_factor = float(self.reduce_factor)
        self.recover_window_count = max(1, int(self.recover_window_count))
        self.recover_step = max(1, int(self.recover_step))
        self.offender_cooldown = max(0, int(self.offender_cooldown))
        self.current_budget = self.base_budget
        if not 0.0 < self.reduce_factor < 1.0:
            raise ValueError(
                f"{self.phase_name}: reduce_factor must be in (0, 1), got {self.reduce_factor}."
            )

    @property
    def offender_count(self) -> int:
        """
        返回当前仍处于冷却期的坏样本数量。

        Returns:
            int: 坏样本数量。
        """
        return len(self.offender_cooldowns)

    def get_batch_cooldown_action(self, root_indices: list[int], num_graphs: int) -> str:
        """
        判断当前 batch 是否需要因坏样本冷却而拆分或跳过。

        Args:
            root_indices: 当前根 batch 对应的数据集索引。
            num_graphs: 当前 batch 内图数量。

        Returns:
            str: `normal`、`split` 或 `skip`。
        """
        if not root_indices:
            return "normal"
        has_offender = any(self.offender_cooldowns.get(idx, 0) > 0 for idx in root_indices)
        if not has_offender:
            return "normal"
        if num_graphs <= 1:
            return "skip"
        return "split"

    def mark_offender(
        self,
        root_indices: list[int],
        *,
        severe: bool = False,
    ) -> None:
        """
        将一组根样本标记为冷却中的坏样本。

        Args:
            root_indices: 触发异常的根样本索引。
            severe: 是否按更长冷却期处理。
        """
        if self.offender_cooldown <= 0:
            return
        cooldown = self.offender_cooldown * (2 if severe else 1)
        for idx in root_indices:
            self.offender_cooldowns[int(idx)] = max(
                self.offender_cooldowns.get(int(idx), 0),
                cooldown,
            )

    def note_cooldown_skip(self, root_indices: list[int]) -> None:
        """
        记录一次因冷却而发生的预防性跳过。

        这不会参与预算降档统计，只会延续 offender 隔离。

        Args:
            root_indices: 被冷却逻辑拦下的根样本索引。
        """
        self._advance_offender_cooldowns()
        self.mark_offender(root_indices, severe=True)

    def record_root_event(
        self,
        *,
        root_indices: list[int],
        had_oom: bool,
        irreducible: bool = False,
        count_in_window: bool = True,
    ) -> BudgetAdjustment | None:
        """
        记录一次根 batch 级事件，并按窗口规则回调预算。

        Args:
            root_indices: 参与该根事件的样本索引。
            had_oom: 本次根事件是否出现过至少一次 OOM。
            irreducible: 是否属于不可约 OOM。
            count_in_window: 是否将本次事件计入窗口统计。

        Returns:
            BudgetAdjustment | None: 预算发生变化时返回调整结果，否则返回 `None`。
        """
        self._advance_offender_cooldowns()
        if had_oom:
            self.mark_offender(root_indices, severe=irreducible)
        if not count_in_window:
            return None
        self._window_total += 1
        if had_oom:
            self._window_oom += 1
        return self._maybe_adjust_budget()

    def _advance_offender_cooldowns(self) -> None:
        """
        推进坏样本冷却计数。
        """
        if not self.offender_cooldowns:
            return
        next_state: dict[int, int] = {}
        for idx, remain in self.offender_cooldowns.items():
            new_remain = int(remain) - 1
            if new_remain > 0:
                next_state[idx] = new_remain
        self.offender_cooldowns = next_state

    def _maybe_adjust_budget(self) -> BudgetAdjustment | None:
        """
        在窗口结算点执行预算回调。

        Returns:
            BudgetAdjustment | None: 若预算发生变化则返回详细结果。
        """
        if self._window_total < self.window_size:
            return None
        window_total = self._window_total
        window_oom = self._window_oom
        self._window_total = 0
        self._window_oom = 0

        if window_oom >= self.reduce_threshold:
            self._clean_windows = 0
            if not self.enable_adaptive or self.current_budget <= self.min_budget:
                return BudgetAdjustment(
                    phase_name=self.phase_name,
                    action="hold",
                    previous_budget=self.current_budget,
                    new_budget=self.current_budget,
                    window_total=window_total,
                    window_oom=window_oom,
                    offender_count=self.offender_count,
                    reason="oom_window_reached_but_budget_already_clamped",
                )
            previous_budget = self.current_budget
            self.current_budget = max(
                self.min_budget,
                int(self.current_budget * self.reduce_factor),
            )
            return BudgetAdjustment(
                phase_name=self.phase_name,
                action="reduce",
                previous_budget=previous_budget,
                new_budget=self.current_budget,
                window_total=window_total,
                window_oom=window_oom,
                offender_count=self.offender_count,
                reason="oom_window_threshold_reached",
            )

        if window_oom == 0:
            self._clean_windows += 1
            if (
                self.enable_adaptive
                and self.current_budget < self.base_budget
                and self._clean_windows >= self.recover_window_count
            ):
                previous_budget = self.current_budget
                self.current_budget = min(
                    self.base_budget,
                    self.current_budget + self.recover_step,
                )
                self._clean_windows = 0
                return BudgetAdjustment(
                    phase_name=self.phase_name,
                    action="recover",
                    previous_budget=previous_budget,
                    new_budget=self.current_budget,
                    window_total=window_total,
                    window_oom=window_oom,
                    offender_count=self.offender_count,
                    reason="clean_windows_recovered_budget",
                )
            return None

        self._clean_windows = 0
        return None


class AdaptiveCostBatchSampler(Sampler[list[int]]):
    """
    按样本成本打包的 batch 采样器。

    将成本接近的样本尽量放入同一批次，
    以降低长尾样本导致的 batch 波动。
    """

    def __init__(
        self,
        *,
        sample_costs: list[int],
        max_cost: int,
        shuffle: bool,
        seed: int,
        window_size: int = 128,
    ) -> None:
        """
        初始化成本感知采样器。

        Args:
            sample_costs: 与局部数据集顺序对齐的样本成本列表。
            max_cost: 单个 batch 允许的最大成本预算。
            shuffle: 是否打乱样本顺序。
            seed: 随机种子。
            window_size: 局部排序窗口大小。
        """
        self.sample_costs = [max(1, int(cost)) for cost in sample_costs]
        self.max_cost = max(1, int(max_cost))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.window_size = max(8, int(window_size))

    def __len__(self) -> int:
        running_cost = 0
        batch_count = 0
        for cost in self.sample_costs:
            if running_cost > 0 and running_cost + cost > self.max_cost:
                batch_count += 1
                running_cost = 0
            running_cost += min(cost, self.max_cost)
        return batch_count + int(running_cost > 0)

    def __iter__(self):
        indices = list(range(len(self.sample_costs)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)
        ordered: list[int] = []
        for start in range(0, len(indices), self.window_size):
            window = indices[start : start + self.window_size]
            window.sort(key=lambda idx: self.sample_costs[idx], reverse=True)
            ordered.extend(window)

        batch: list[int] = []
        batch_cost = 0
        for idx in ordered:
            sample_cost = min(self.sample_costs[idx], self.max_cost)
            if batch and batch_cost + sample_cost > self.max_cost:
                yield batch
                batch = []
                batch_cost = 0
            batch.append(idx)
            batch_cost += sample_cost
        if batch:
            yield batch



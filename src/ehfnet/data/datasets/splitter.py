"""
数据划分工具。

负责按 scaffold 划分数据集并缓存划分结果，
为训练、验证和测试阶段提供可复用的切分接口。
"""


import json
import logging
import os
import random
from collections import defaultdict
from typing import Protocol, cast

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from ehfnet.data.datasets.ligand_sanitize import load_ligand_mol

logger = logging.getLogger(__name__)


class SizedDataset(Protocol):
    """
    带长度信息的数据集协议。

    定义支持 `len()` 调用的数据集最小接口，
    供划分工具在不依赖具体实现类的情况下进行类型约束。
    """

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> object: ...


class ProteinLigandLikeDataset(SizedDataset, Protocol):
    """
    兼容蛋白配体数据集的协议。

    补充 `index_df` 和 `raw_dir` 等划分所需属性约定，
    便于划分器同时支持真实数据集和兼容包装器。
    """

    index_df: pd.DataFrame
    raw_dir: str


def generate_scaffold(mol: Chem.Mol, include_chirality: bool = False) -> str:
    """
    计算分子的 Bemis-Murcko 骨架 SMILES。

    Args:
        mol: 待读取或处理的 RDKit 分子对象。
        include_chirality: 是否在骨架计算中区分手性信息。

    Returns:
        str: 骨架的 SMILES 字符串；计算失败时返回空字符串。
    """

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
        return scaffold

    except Exception:
        return ""


def _read_ligand_safe(path: str) -> Chem.Mol | None:
    """
    安全读取配体文件 (SDF/MOL2)，带自动清洗和错误处理。

    Returns:
        Chem.Mol | None: 返回成功读取的配体对象；若文件缺失或读取失败则返回 `None`。
    """

    if not os.path.exists(path):
        return None

    try:
        return load_ligand_mol(path, remove_hs=False, require_conformer=False)
    except Exception as exc:
        logger.warning("Failed to read ligand for scaffold split: %s (%s)", path, exc)
        return None


class ScaffoldSplitter:
    """
    基于 scaffold 的数据划分器。

    负责按照分子骨架对样本进行训练、验证和测试划分，
    并支持将划分结果缓存到磁盘以便复用。
    """

    def __init__(
        self,
        include_chirality: bool = False,
        seed: int = 42
    ):
        """
        初始化对象。

        Args:
            include_chirality: 是否在骨架计算中区分手性信息。
            seed: 随机种子。
        """

        self.include_chirality = include_chirality
        self.seed = seed

    def split(
        self,
        dataset: Dataset,
        *,
        frac_train: float = 0.9,
        frac_val: float = 0.1,
        frac_test: float = 0.0,
        index_df: pd.DataFrame | None = None,
    ) -> tuple[Subset, Subset, Subset]:
        """
        执行划分。

        Args:
            dataset: 参与处理或划分的数据集对象。
            frac_train: 训练集划分比例。
            frac_val: 验证集划分比例。
            frac_test: 测试集划分比例。
            index_df: 可选的索引表，用于覆盖数据集内部索引。

        Returns:
            tuple[Subset, Subset, Subset]: 训练、验证与测试子集。
        """

        split_indices = self.split_indices(
            dataset,
            frac_train=frac_train,
            frac_val=frac_val,
            frac_test=frac_test,
            index_df=index_df,
        )

        return self.subsets_from_indices(dataset, split_indices)


    def split_indices(
        self,
        dataset: Dataset,
        *,
        frac_train: float = 0.9,
        frac_val: float = 0.1,
        frac_test: float = 0.0,
        index_df: pd.DataFrame | None = None,
    ) -> dict[str, list[int]]:
        """
        仅生成并返回划分索引，不创建 Subset。

        Args:
            dataset: 参与处理或划分的数据集对象。
            frac_train: 训练集划分比例。
            frac_val: 验证集划分比例。
            frac_test: 测试集划分比例。
            index_df: 可选的索引表，用于覆盖数据集内部索引。

        Returns:
            dict[str, list[int]]: 返回按 train/val/test 分组的数据集索引字典。

        Raises:
            ValueError: 当输入参数或运行时状态不满足要求时抛出。
            AttributeError: 当访问的属性不存在或对象不满足接口约定时抛出。
        """

        if not np.isclose(frac_train + frac_val + frac_test, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {frac_train + frac_val + frac_test}")

        if index_df is None:

            if hasattr(dataset, "index_df") and hasattr(dataset, "raw_dir"):
                index_df = cast(pd.DataFrame, getattr(dataset, "index_df"))
            else:
                raise ValueError("Dataset must have 'index_df' attribute or it must be provided.")

        if not hasattr(dataset, "_pdb_to_idx"):
            raise AttributeError("Dataset must have '_pdb_to_idx' attribute mapping pdb_id to index.")

        pdb_to_idx = getattr(dataset, "_pdb_to_idx")
        raw_dir = getattr(dataset, "raw_dir", ".")

        logger.info(f"Start Scaffold Split (seed={self.seed})...")

        scaffold_map = defaultdict(list)
        invalid_count = 0
        missing_in_dataset_count = 0

        for _, row in tqdm(index_df.iterrows(), total=len(index_df), desc="Analyzing Scaffolds"):
            pdb_id = str(row["pdb_id"]).lower()

            if pdb_id not in pdb_to_idx:
                missing_in_dataset_count += 1
                continue

            real_dataset_idx = pdb_to_idx[pdb_id]

            pdb_dir = os.path.join(raw_dir, pdb_id)
            lig_sdf = os.path.join(pdb_dir, f"{pdb_id}_ligand.sdf")
            lig_mol2 = os.path.join(pdb_dir, f"{pdb_id}_ligand.mol2")
            mol_path = lig_sdf if os.path.exists(lig_sdf) else lig_mol2

            mol = _read_ligand_safe(mol_path)

            if mol is None:
                scaffold = f"invalid::{pdb_id}"
                invalid_count += 1

            else:
                scaffold = generate_scaffold(mol, self.include_chirality)
                if not scaffold:
                    scaffold = f"invalid::{pdb_id}"
                    invalid_count += 1

            scaffold_map[scaffold].append(real_dataset_idx)

        if invalid_count > 0:
            logger.warning(
                "Failed to generate valid scaffolds for %d ligands. Treating them as singleton invalid buckets.",
                invalid_count,
            )

        if missing_in_dataset_count > 0:
            logger.warning(f"Skipped {missing_in_dataset_count} entries present in CSV but missing in Dataset.")

        scaffold_sets = list(scaffold_map.values())
        scaffold_sets.sort(key=lambda x: (len(x), x[0]), reverse=True)

        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        sized_dataset = cast(SizedDataset, dataset)
        dataset_size = len(sized_dataset)
        train_cutoff = int(dataset_size * frac_train)
        val_cutoff = int(dataset_size * (frac_train + frac_val))

        for group in scaffold_sets:
            group_size = len(group)

            if len(train_indices) + group_size <= train_cutoff:
                train_indices.extend(group)

            elif len(train_indices) + len(val_indices) + group_size <= val_cutoff:
                val_indices.extend(group)

            else:
                test_indices.extend(group)

        rng = random.Random(self.seed)
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)

        logger.info(
            f"Split Result - Train: {len(train_indices)} ({len(train_indices)/dataset_size:.1%}), "
            f"Val: {len(val_indices)} ({len(val_indices)/dataset_size:.1%}), "
            f"Test: {len(test_indices)} ({len(test_indices)/dataset_size:.1%})"
        )

        if len(val_indices) == 0 and frac_val > 0:
            logger.warning("Validation set is empty! This happens when one huge scaffold dominates the dataset.")

        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }


    @staticmethod
    def subsets_from_indices(
        dataset: Dataset,
        split_indices: dict[str, list[int]],
    ) -> tuple[Subset, Subset, Subset]:
        """
        通过索引字典创建 (train, val, test) 子集。

        Args:
            dataset: 参与处理或划分的数据集对象。
            split_indices: 训练、验证和测试索引划分结果。

        Returns:
            tuple[Subset, Subset, Subset]: 返回训练、验证和测试三个 `Subset` 对象。
        """

        return (
            Subset(dataset, split_indices.get("train", [])),
            Subset(dataset, split_indices.get("val", [])),
            Subset(dataset, split_indices.get("test", [])),
        )


    @staticmethod
    def save_split(
        split_path: str,
        split_indices: dict[str, list[int]],
        *,
        metadata: dict | None = None,
    ) -> None:
        """
        将划分索引保存为 JSON，便于复现实验与专利材料留存。

        Args:
            split_path: 数据划分缓存文件路径。
            split_indices: 训练、验证和测试索引划分结果。
            metadata: 随划分或缓存一同保存的附加元数据。
        """

        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        payload = {
            "metadata": metadata or {},
            "indices": {
                "train": [int(i) for i in split_indices.get("train", [])],
                "val": [int(i) for i in split_indices.get("val", [])],
                "test": [int(i) for i in split_indices.get("test", [])],
            },
        }

        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


    @staticmethod
    def load_split(split_path: str) -> tuple[dict[str, list[int]], dict]:
        """
        从 JSON 加载划分索引与元信息。

        Args:
            split_path: 数据划分缓存文件路径。

        Returns:
            tuple[dict[str, list[int]], dict]: 返回划分索引字典及其附带元信息。
        """

        with open(split_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        indices = payload.get("indices", {})
        split_indices = {
            "train": [int(i) for i in indices.get("train", [])],
            "val": [int(i) for i in indices.get("val", [])],
            "test": [int(i) for i in indices.get("test", [])],
        }
        metadata = payload.get("metadata", {})
        return split_indices, metadata

"""
PDBBind 数据集

提供基于 PDBBind 目录结构的 Dataset 实现，并将原始复合物处理为 HeteroData 缓存。
"""

import os
import os.path as osp
import logging

import torch
import pandas as pd

from tqdm import tqdm
from typing import Any, Callable, cast

from rdkit.Chem import ChemicalFeatures, RDConfig

from torch_geometric.data import Dataset, HeteroData

from ehfnet.graph import GraphBuilder
from ehfnet.datasets.prepare import prepare_graph, get_esm_model

logger = logging.getLogger(__name__)


def load_index(index_file: str) -> pd.DataFrame:
    """
    加载数据索引文件。

    支持两种格式：
    - CSV：要求包含列 {"pdb_id", "affinity"}
    - PDBBind INDEX：按行解析，读取第 0 列 pdb_id 与第 3 列 affinity

    Args:
        index_file: 索引文件路径

    Returns:
        包含 pdb_id 与 affinity 的 DataFrame
    """

    if not osp.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")

    if index_file.endswith(".csv"):
        df = pd.read_csv(index_file)
        required_cols = {"pdb_id", "affinity"}

        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        df["pdb_id"] = df["pdb_id"].astype(str).str.lower()

        return df

    data: list[dict[str, str | float]] = []

    with open(index_file, "r") as f:

        for line in f:
            line = line.strip()

            if not line or line.startswith("#") or line.startswith("PDB"):
                continue
            parts = line.split()

            if len(parts) >= 4:
                data.append({"pdb_id": parts[0].lower(), "affinity": float(parts[3])})

    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} complexes from PDBBind INDEX format")

    return df


def ligand_path(pdb_id: str, pdb_dir: str) -> str | None:
    """
    获取配体文件路径（优先 SDF，其次 MOL2）。

    Args:
        pdb_id: 复合物 ID
        pdb_dir: 复合物目录

    Returns:
        配体文件路径；若不存在返回 None
    """

    sdf = osp.join(pdb_dir, f"{pdb_id}_ligand.sdf")

    if osp.exists(sdf):
        return sdf

    mol2 = osp.join(pdb_dir, f"{pdb_id}_ligand.mol2")

    if osp.exists(mol2):
        return mol2

    return None


def protein_path(pdb_id: str, pdb_dir: str) -> str | None:
    """
    获取蛋白质 PDB 文件路径。

    Args:
        pdb_id: 复合物 ID
        pdb_dir: 复合物目录

    Returns:
        蛋白质 PDB 路径；若不存在返回 None
    """

    p = osp.join(pdb_dir, f"{pdb_id}_protein.pdb")
    return p if osp.exists(p) else None


def esm_cache_paths(pdb_id: str, pdb_dir: str, esm_root: str | None) -> tuple[str | None, str | None]:
    """
    获取 ESM embedding 缓存的读路径与写路径。

    优先使用复合物目录下的本地缓存；若配置了 esm_root，则尝试在 esm_root 下读取/写入。

    Args:
        pdb_id: 复合物 ID
        pdb_dir: 复合物目录
        esm_root: 全局 ESM 缓存目录（可选）

    Returns:
        (read_path, write_path)
    """

    local_path = osp.join(pdb_dir, f"{pdb_id}_esm.npz")

    if osp.exists(local_path):
        return local_path, local_path

    if esm_root and osp.isdir(esm_root):
        global_path = osp.join(esm_root, f"{pdb_id}.npz")

        if osp.exists(global_path):
            return global_path, global_path

        return None, global_path

    return None, local_path


class PDBBindDataset(Dataset):
    """
    PDBBind 数据集。

    读取 index_file 并在 processed_dir 下缓存每个复合物的图数据文件 data_{pdb_id}.pt。
    """

    def __init__(
        self,
        root: str,
        index_file: str,
        *,
        esm_root: str | None = None,
        esm: str = "auto",
        esm_model_name: str = "esmc_300m",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        r_cutoff_intra: float = 5.0,
        r_cutoff_inter: float = 6.0,
        max_neighbors_intra: int = 64,
        max_neighbors_inter: int = 32,
        force_reprocess: bool = False,
    ) -> None:
        """
        Args:
            root: 数据集根目录（包含 raw/processed）
            index_file: 索引文件路径（CSV 或 PDBBind INDEX 格式）
            esm_root: 全局 ESM 缓存目录（可选）
            esm: ESM 处理模式（如 "auto"）
            esm_model_name: ESM 模型名称
            transform: PyG Dataset transform
            pre_transform: PyG Dataset pre_transform
            pre_filter: PyG Dataset pre_filter
            r_cutoff_intra: 图内边半径阈值
            r_cutoff_inter: 跨图边半径阈值
            max_neighbors_intra: 图内最大邻居数
            max_neighbors_inter: 跨图最大邻居数
            force_reprocess: 是否强制重建缓存
        """
        self.index_file = index_file
        self.esm_root = esm_root
        self.esm = esm
        self.esm_model_name = esm_model_name
        self.force_reprocess = force_reprocess

        self.index_df = load_index(index_file)

        fdef_path = osp.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        self.feature_factory = cast(Any, ChemicalFeatures).BuildFeatureFactory(fdef_path)

        self.graph_builder = GraphBuilder(
            r_cutoff_intra=r_cutoff_intra,
            r_cutoff_inter=r_cutoff_inter,
            max_neighbors_intra=max_neighbors_intra,
            max_neighbors_inter=max_neighbors_inter,
        )

        self._esm_model = None

        super().__init__(root, transform, pre_transform, pre_filter)
        self._build_valid_index()

    @property
    def raw_file_names(self) -> list[str]:
        """
        Returns:
            raw 目录需要存在的文件名列表
        """

        return []

    @property
    def processed_file_names(self) -> list[str]:
        """
        Returns:
            processed 目录期望生成的文件名列表
        """

        return [f"data_{pdb_id}.pt" for pdb_id in self.index_df["pdb_id"]]

    def download(self) -> None:
        return None

    def process(self):
        """
        处理并缓存全部样本到 processed_dir。
        """

        if self.force_reprocess:
            logger.info("Force reprocess enabled - overwriting existing cache")

        total = len(self.index_df)
        logger.info(f"Processing {total} complexes...")

        success_count = 0
        skip_count = 0
        error_count = 0

        for _, row in tqdm(self.index_df.iterrows(), total=total, desc="Processing"):
            pdb_id = row["pdb_id"]
            affinity = float(row["affinity"])
            out_path = osp.join(self.processed_dir, f"data_{pdb_id}.pt")

            if not self.force_reprocess and osp.exists(out_path):
                skip_count += 1
                continue

            try:
                data = self._process_one(pdb_id, affinity)

                if data is None:
                    error_count += 1
                    continue

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, out_path)
                success_count += 1

            except Exception as e:
                logger.warning(f"Error processing {pdb_id}: {e}")
                error_count += 1

        logger.info(
            f"Processing complete: {success_count} success, {skip_count} cached, {error_count} errors"
        )


    def _process_one(self, pdb_id: str, affinity: float) -> HeteroData | None:
        """
        处理单个复合物并构建图数据。

        Args:
            pdb_id: 复合物 ID
            affinity: 亲和力标签

        Returns:
            生成的 HeteroData；若缺文件或处理失败返回 None
        """

        pdb_dir = osp.join(self.raw_dir, pdb_id)

        lig = ligand_path(pdb_id, pdb_dir)
        pro = protein_path(pdb_id, pdb_dir)

        if lig is None or pro is None:
            return None

        esm_cache_path, esm_cache_write_path = esm_cache_paths(pdb_id, pdb_dir, self.esm_root)

        if self.esm == "auto" and esm_cache_path is None and self._esm_model is None:
            self._esm_model = get_esm_model(model_name=self.esm_model_name)

        data = prepare_graph(
            pdb_id=pdb_id,
            ligand_path=lig,
            protein_path=pro,
            affinity=affinity,
            feature_factory=self.feature_factory,
            graph_builder=self.graph_builder,
            esm_cache_path=esm_cache_path,
            esm_cache_write_path=esm_cache_write_path,
            esm=self.esm,
            esm_model=self._esm_model,
            esm_model_name=self.esm_model_name,
        )

        return data

    def _build_valid_index(self):
        """
        扫描 processed_dir 并构建可用样本索引。
        """

        os.makedirs(self.processed_dir, exist_ok=True)
        processed_files = os.listdir(self.processed_dir)
        allowed = set(self.index_df["pdb_id"].tolist())
        valid_pdb_ids = sorted(
            [
                f.replace("data_", "").replace(".pt", "")
                for f in processed_files
                if f.startswith("data_") and f.endswith(".pt")
                and f.replace("data_", "").replace(".pt", "") in allowed
            ]
        )

        self._valid_pdb_ids = valid_pdb_ids
        self._pdb_to_idx = {pdb: i for i, pdb in enumerate(valid_pdb_ids)}
        logger.info(f"Dataset ready: {len(valid_pdb_ids)} valid samples")


    def len(self) -> int:
        """
        Returns:
            可用样本数量
        """

        return len(self._valid_pdb_ids)


    def get(self, idx: int) -> HeteroData:
        """
        获取单个样本。

        Args:
            idx: 样本索引

        Returns:
            HeteroData 样本
        """

        if idx < 0 or idx >= len(self._valid_pdb_ids):
            raise IndexError(f"Index {idx} out of range [0, {len(self._valid_pdb_ids)})")

        pdb_id = self._valid_pdb_ids[idx]
        file_path = osp.join(self.processed_dir, f"data_{pdb_id}.pt")

        return cast(HeteroData, torch.load(file_path, weights_only=False))

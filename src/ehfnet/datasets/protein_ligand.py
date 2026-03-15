"""
蛋白-配体复合物数据集

通用目录结构：根目录下 cleaned/<complex_id>/ 存放配体、蛋白及可选 ESM 缓存，
配合 index.csv（表头 Concatenated ID、Log Binding Affinity）即可用于各类结合数据（如 HiqBind、PDBBind 等）。
将原始复合物处理为 HeteroData 图缓存。
"""

import os
import os.path as osp
import logging
import zlib
import json
import torch
import pandas as pd

from torch import Tensor
from tqdm import tqdm
from collections.abc import Callable
from collections import Counter
from typing import Any, cast, overload

from rdkit.Chem import ChemicalFeatures, RDConfig

from torch_geometric.data import Dataset, HeteroData

from ehfnet.graph import GraphBuilder, ESMEmbeddingFiller
from ehfnet.graph.pocket_features import build_pocket_features, pocket_feature_dim
from ehfnet.datasets.prepare import prepare_graph, get_esm_model
from ehfnet.datasets.pose_initialization import generate_decoupled_ligand_positions
from ehfnet.datasets.ligand_sanitize import LigandSanitizationError


logger = logging.getLogger(__name__)

ESM_CACHE_VERSION_TAG = "esm_chainseg"
GRAPH_CACHE_VERSION_TAG = "graph_cache"
GRAPH_CACHE_DIRNAME = "cache"
PREPROCESS_SUMMARY_FILENAME = "preprocess_summary.json"
PREPROCESS_METADATA_DIRNAME = "_preprocess_meta"


def _normalize_ligand_sanitize_mode(mode: Any) -> str:
    value = str(mode).strip().lower() if mode is not None else "unknown"
    return value if value in {"full", "partial", "rejected", "unknown"} else "unknown"


def _extract_ligand_sanitize_metadata(data: HeteroData) -> dict[str, Any]:
    mode = _normalize_ligand_sanitize_mode(getattr(data, "ligand_sanitize_mode", "unknown"))
    return {
        "ligand_sanitize_mode": mode,
        "ligand_partial_sanitize": bool(getattr(data, "ligand_partial_sanitize", mode == "partial")),
        "ligand_full_sanitize_flag": int(getattr(data, "ligand_full_sanitize_flag", -1)),
        "ligand_partial_sanitize_flag": int(getattr(data, "ligand_partial_sanitize_flag", -1)),
    }


def load_index(index_file: str) -> pd.DataFrame:
    """
    加载数据索引文件。

    表头必须为 "Concatenated ID" 与 "Log Binding Affinity"（与 prepare_data 输出一致），
    使用 pdb_id/affinity 等其它列名会报错。

    Args:
        index_file: 索引文件路径

    Returns:
        含 pdb_id、affinity 列的 DataFrame（由上述两列重命名得到）
    """

    if not osp.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")

    if not index_file.endswith(".csv"):
        raise ValueError("Index file must be a CSV file.")

    df = pd.read_csv(index_file, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    if "Concatenated ID" not in df.columns or "Log Binding Affinity" not in df.columns:
        raise ValueError(
            f"index.csv 表头必须为 Concatenated ID、Log Binding Affinity，不能使用 pdb_id/affinity 等其它列名。当前列: {list(df.columns)}"
        )
    df = df.rename(columns={"Concatenated ID": "pdb_id", "Log Binding Affinity": "affinity"})

    initial_len = len(df)
    df = df.dropna(subset=["affinity"])
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with NaN affinity from index.")

    df["pdb_id"] = df["pdb_id"].astype(str).str.lower()

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

    使用带版本标签的缓存文件名，避免继续复用旧的“按 segment 粗暴拼链”缓存。
    优先使用复合物目录下的本地缓存；若配置了 esm_root，则尝试在 esm_root 下读取/写入。

    Args:
        pdb_id: 复合物 ID
        pdb_dir: 复合物目录
        esm_root: 全局 ESM 缓存目录（可选）

    Returns:
        (read_path, write_path)
    """

    local_path = osp.join(pdb_dir, f"{pdb_id}_{ESM_CACHE_VERSION_TAG}.npz")

    if osp.exists(local_path):
        return local_path, local_path

    if esm_root and osp.isdir(esm_root):
        global_path = osp.join(esm_root, f"{pdb_id}_{ESM_CACHE_VERSION_TAG}.npz")

        if osp.exists(global_path):
            return global_path, global_path

        return None, global_path

    return None, local_path


class ProteinLigandDataset(Dataset):
    """
    蛋白-配体复合物数据集。

    目录结构：root/cleaned/<id>/ 下为各复合物；index.csv 表头为 Concatenated ID、Log Binding Affinity。
    在 processed 目录下缓存每个复合物的图数据 data_{pdb_id}.pt。
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
        max_neighbors_intra: int = 64,
        interaction_profile: str = "full",
        force_reprocess: bool = False,
        esm_dim: int = 960,
    ) -> None:
        """
        Args:
            root: 数据集根目录（其下应有 cleaned/）
            index_file: 索引 CSV 路径（表头 Concatenated ID、Log Binding Affinity）
            esm_root: 全局 ESM 缓存目录（可选）
            esm: ESM 处理模式（如 "auto"）
            esm_model_name: ESM 模型名称
            transform: PyG Dataset 的 transform（可选）
            pre_transform: PyG Dataset 的 pre_transform（可选）
            pre_filter: PyG Dataset 的 pre_filter（可选）
            r_cutoff_intra: 图内边半径阈值
            max_neighbors_intra: 图内最大邻居数
            interaction_profile: 跨图交互配置（"full" 或 "atom_only"）
            force_reprocess: 是否强制重建缓存
            esm_dim: ESM embedding 维度
        """
        self.index_file = index_file
        self.esm_root = esm_root
        self.esm = esm
        self.esm_model_name = esm_model_name
        self.force_reprocess = force_reprocess
        self.esm_dim = esm_dim

        self.index_df = load_index(index_file)

        # 强制指定 cleaned 目录
        self.cleaned_dir = osp.join(root, "cleaned")

        if not osp.exists(self.cleaned_dir):

            # 兼容性检查：如果 cleaned 不存在但 root 下直接有数据，发出警告
            if not self.index_df.empty:
                first_pdb = self.index_df.iloc[0]["pdb_id"]

                if osp.exists(osp.join(root, first_pdb)):
                    logger.warning(f"Data found in {root}, but expected in {self.cleaned_dir}. Please move data to 'cleaned' subdirectory.")
        
        fdef_path = osp.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        self.feature_factory = cast(Any, ChemicalFeatures).BuildFeatureFactory(fdef_path)

        # 默认为 960 (ESMC 300M)，如果使用 ESM-3 请改为 1152
        esm_filler = ESMEmbeddingFiller(embed_dim=self.esm_dim)

        self.graph_builder = GraphBuilder(
            r_cutoff_intra=r_cutoff_intra,
            max_neighbors_intra=max_neighbors_intra,
            esm_filler=esm_filler,
            interaction_profile=interaction_profile,
        )

        self._esm_model = None

        super().__init__(root, transform, pre_transform, pre_filter)
        self._build_valid_index()
        self.affinity_stats = self.compute_affinity_stats()


    def compute_affinity_stats(self, indices: list[int] | None = None) -> dict[str, float]:
        """
        计算亲和力标签的均值和标准差。

        Args:
            indices: Dataset 索引子集；为空时使用全部有效样本
        """
        if self.index_df.empty:
            return {"mean": 0.0, "std": 1.0}

        if indices is None:
            pdb_ids = self._valid_pdb_ids
        else:
            if not indices:
                return {"mean": 0.0, "std": 1.0}
            pdb_ids = [self._valid_pdb_ids[int(i)] for i in indices]

        affinities = self.index_df[self.index_df["pdb_id"].isin(pdb_ids)]["affinity"].to_numpy(dtype=float)
        if affinities.size == 0:
            return {"mean": 0.0, "std": 1.0}
        mean = float(affinities.mean())
        std = float(affinities.std())
        std = max(std, 1e-3)

        return {"mean": mean, "std": std}


    def set_affinity_stats(self, stats: dict[str, float]) -> None:
        """
        更新运行期使用的 affinity 归一化统计。
        """

        self.affinity_stats = {
            "mean": float(stats.get("mean", 0.0)),
            "std": max(float(stats.get("std", 1.0)), 1e-3),
        }

    @overload
    def denormalize_affinity(self, val: Tensor) -> Tensor: ...
    @overload
    def denormalize_affinity(self, val: float) -> float: ...
    def denormalize_affinity(self, val: float | Tensor) -> float | Tensor:
        """
        将归一化的亲和力值还原为 pKd。
        """
        
        return val * self.affinity_stats["std"] + self.affinity_stats["mean"]

    @property
    def raw_dir(self) -> str:
        return self.cleaned_dir

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, GRAPH_CACHE_DIRNAME)

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

    @property
    def preprocess_metadata_dir(self) -> str:
        return osp.join(self.processed_dir, PREPROCESS_METADATA_DIRNAME)

    @property
    def preprocess_summary_path(self) -> str:
        return osp.join(self.processed_dir, PREPROCESS_SUMMARY_FILENAME)

    def _preprocess_metadata_path(self, pdb_id: str) -> str:
        return osp.join(self.preprocess_metadata_dir, f"{pdb_id}.json")

    def _write_preprocess_metadata(self, pdb_id: str, metadata: dict[str, Any]) -> None:
        os.makedirs(self.preprocess_metadata_dir, exist_ok=True)
        metadata = dict(metadata)
        metadata.setdefault("graph_mode", "blind")
        metadata.setdefault("cache_dir", GRAPH_CACHE_DIRNAME)
        with open(self._preprocess_metadata_path(pdb_id), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=True, indent=2, sort_keys=True)

    def _load_cached_preprocess_metadata(self, pdb_id: str) -> dict[str, Any] | None:
        metadata_path = self._preprocess_metadata_path(pdb_id)
        if not osp.exists(metadata_path):
            return None
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning("Failed to read preprocess metadata for %s: %s", pdb_id, exc)
            return None

        payload["ligand_sanitize_mode"] = _normalize_ligand_sanitize_mode(
            payload.get("ligand_sanitize_mode")
        )
        payload.setdefault("graph_mode", "blind")
        payload.setdefault("cache_dir", GRAPH_CACHE_DIRNAME)
        return payload

    def _load_or_recover_preprocess_metadata(self, pdb_id: str, graph_path: str) -> dict[str, Any]:
        cached = self._load_cached_preprocess_metadata(pdb_id)
        if cached is not None:
            return cached

        data = cast(
            HeteroData,
            torch.load(graph_path, map_location="cpu", weights_only=False),
        )
        metadata = _extract_ligand_sanitize_metadata(data)
        self._write_preprocess_metadata(pdb_id, metadata)
        return metadata

    def _write_preprocess_summary(self, summary: dict[str, Any]) -> None:
        os.makedirs(self.processed_dir, exist_ok=True)
        with open(self.preprocess_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2, sort_keys=True)

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
        filtered_count = 0
        error_count = 0
        other_failure_count = 0
        sanitize_counts: Counter[str] = Counter(
            {
                "full": 0,
                "partial": 0,
                "rejected": 0,
                "unknown": 0,
            }
        )

        for _, row in tqdm(self.index_df.iterrows(), total=total, desc="Processing"):
            pdb_id = row["pdb_id"]
            affinity = float(row["affinity"])
            out_path = osp.join(self.processed_dir, f"data_{pdb_id}.pt")

            if not self.force_reprocess and osp.exists(out_path):
                try:
                    metadata = self._load_or_recover_preprocess_metadata(pdb_id, out_path)
                    sanitize_counts[_normalize_ligand_sanitize_mode(metadata.get("ligand_sanitize_mode"))] += 1
                except Exception as exc:
                    logger.warning("Failed to recover sanitize metadata for cached sample %s: %s", pdb_id, exc)
                    sanitize_counts["unknown"] += 1
                skip_count += 1
                continue

            try:
                data = self._process_one(pdb_id, affinity)

                if data is None:
                    sanitize_counts["unknown"] += 1
                    error_count += 1
                    other_failure_count += 1
                    continue

                metadata = _extract_ligand_sanitize_metadata(data)
                sanitize_counts[metadata["ligand_sanitize_mode"]] += 1

                if self.pre_filter is not None and not self.pre_filter(data):
                    filtered_count += 1
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, out_path)
                self._write_preprocess_metadata(pdb_id, metadata)
                success_count += 1

            except LigandSanitizationError as exc:
                self._write_preprocess_metadata(
                    pdb_id,
                    {
                        "ligand_sanitize_mode": "rejected",
                        "ligand_partial_sanitize": False,
                        "ligand_full_sanitize_flag": exc.full_flag,
                        "ligand_partial_sanitize_flag": (
                            -1 if exc.partial_flag is None else exc.partial_flag
                        ),
                    },
                )
                logger.warning("Ligand sanitize rejected %s: %s", pdb_id, exc)
                sanitize_counts["rejected"] += 1
                error_count += 1
            except Exception as e:
                logger.warning(f"Error processing {pdb_id}: {e}")
                sanitize_counts["unknown"] += 1
                error_count += 1
                other_failure_count += 1

        summary = {
            "graph_cache_version": GRAPH_CACHE_VERSION_TAG,
            "graph_mode": "blind",
            "cache_dir": GRAPH_CACHE_DIRNAME,
            "esm_cache_version": ESM_CACHE_VERSION_TAG,
            "processed_dir": osp.abspath(self.processed_dir),
            "index_file": osp.abspath(self.index_file),
            "force_reprocess": bool(self.force_reprocess),
            "totals": {
                "indexed": total,
                "success": success_count,
                "cached": skip_count,
                "filtered": filtered_count,
                "errors": error_count,
                "other_failures": other_failure_count,
            },
            "ligand_sanitize": {
                "full": int(sanitize_counts["full"]),
                "partial": int(sanitize_counts["partial"]),
                "rejected": int(sanitize_counts["rejected"]),
                "unknown": int(sanitize_counts["unknown"]),
            },
        }
        self._write_preprocess_summary(summary)

        logger.info(
            "Processing complete: %d success, %d cached, %d filtered, %d errors",
            success_count,
            skip_count,
            filtered_count,
            error_count,
        )
        logger.info(
            "Ligand sanitize summary: full=%d partial=%d rejected=%d unknown=%d",
            sanitize_counts["full"],
            sanitize_counts["partial"],
            sanitize_counts["rejected"],
            sanitize_counts["unknown"],
        )
        logger.info("Preprocess summary written to %s", self.preprocess_summary_path)

        # 释放 ESM 模型显存，因为它在训练阶段不再需要
        if self._esm_model is not None:
            logger.info("Releasing ESM model from GPU memory...")
            del self._esm_model
            self._esm_model = None
            torch.cuda.empty_cache()


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


    def _diffdock_like_cache_path(self, pdb_id: str) -> str:
        return osp.join(self.root, "candidates", "diffdock_like_init", pdb_id, "poses.pt")


    def _load_or_build_start_pos(self, pdb_id: str, expected_num_atoms: int) -> torch.Tensor | None:
        cache_path = self._diffdock_like_cache_path(pdb_id)

        if osp.exists(cache_path):
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            start_pos = cached.get("ligand_start_pos") if isinstance(cached, dict) else cached
            if isinstance(start_pos, torch.Tensor) and start_pos.ndim == 2 and start_pos.size(1) == 3:
                if int(start_pos.size(0)) == expected_num_atoms:
                    return start_pos.float()

        pdb_dir = osp.join(self.raw_dir, pdb_id)
        lig_path = ligand_path(pdb_id, pdb_dir)
        if lig_path is None:
            return None

        seed = zlib.adler32(pdb_id.encode("utf-8")) & 0xFFFFFFFF
        start_pos_np = generate_decoupled_ligand_positions(lig_path, random_seed=seed)
        start_pos = torch.as_tensor(start_pos_np, dtype=torch.float32)
        if int(start_pos.size(0)) != expected_num_atoms:
            logger.warning(
                f"Start pose atom count mismatch for {pdb_id}: expected {expected_num_atoms}, got {int(start_pos.size(0))}."
            )
            return None

        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        torch.save({"ligand_start_pos": start_pos}, cache_path)
        return start_pos


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

        data = cast(HeteroData, torch.load(file_path, weights_only=False))
        
        # 几何合理性检查已移至 preprocess build 阶段一次性完成

        # 实时归一化逻辑
        if hasattr(data, "y_energy"):
            raw_val = data.y_energy
            
            if not isinstance(raw_val, torch.Tensor):
                raw_val = torch.tensor(raw_val, dtype=torch.float)
                
            data.y_energy_raw = raw_val
            
            mean = self.affinity_stats["mean"]
            std = self.affinity_stats["std"]
            data.y_energy = (raw_val - mean) / std

        if hasattr(data["ligand_atom"], "pos"):
            expected_num_atoms = int(data["ligand_atom"].pos.size(0))
            start_pos = self._load_or_build_start_pos(pdb_id, expected_num_atoms)
            if start_pos is not None:
                data["ligand_atom"]["start_pos"] = start_pos

        expected_pocket_dim = pocket_feature_dim(int(data["protein_residue"].x_cont.size(1)))
        pocket_x_cont = getattr(data["protein_pocket"], "x_cont", None)
        if (
            pocket_x_cont is None
            or pocket_x_cont.ndim != 2
            or int(pocket_x_cont.size(1)) != expected_pocket_dim
        ):
            data["protein_pocket"].x_cont = build_pocket_features(
                residue_x_cont=data["protein_residue"].x_cont,
                residue_pos=data["protein_residue"].pos,
                protein_atom_pos=data["protein_atom"].pos,
                residue_esm_missing_mask=getattr(data["protein_residue"], "esm_missing_mask", None),
                esm_feature_start=self.graph_builder._residue_esm_feature_start,
            )
            data["protein_pocket"].num_nodes = int(data["protein_pocket"].x_cont.size(0))

        # 为 blind candidate replay 提供稳定、可重放的数据集索引（基于本 Dataset，非 DataLoader 内局部编号）
        data.dataset_index = int(idx)
        data.dataset_pdb_id = str(pdb_id)
            
        return data

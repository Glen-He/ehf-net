"""
数据集路径工具。

负责解析 index.csv、原始文件位置和各类缓存路径，
是数据目录约定和路径组织规则的集中实现。
"""


import logging
import os.path as osp

import pandas as pd

from ehfnet.contracts import (
    ESM_CACHE_VERSION_TAG,
    GRAPH_CACHE_DIRNAME,
    GRAPH_CACHE_SCHEMA_TAG,
    PREPROCESS_METADATA_DIRNAME,
    PREPROCESS_SUMMARY_FILENAME,
)

logger = logging.getLogger(__name__)


def normalize_esm_model_cache_tag(esm_model_name: str) -> str:
    """
    规范化 ESM 模型缓存标签。

    将模型名转换为适合写入文件名的稳定标识，
    避免路径分隔符和特殊字符影响缓存文件组织。

    Args:
        esm_model_name: ESM 主干模型名称。

    Returns:
        str: 返回可安全写入缓存文件名的 ESM 模型标签。

    Raises:
        ValueError: 当输入参数或运行时状态不满足要求时抛出。
    """

    normalized = esm_model_name.strip().lower()
    for old, new in (("/", "__"), ("\\", "__"), (":", "_"), (" ", "_"), ("-", "_"), (".", "_")):
        normalized = normalized.replace(old, new)
    if not normalized:
        raise ValueError("esm_model_name must not be empty.")
    return normalized


def load_index(index_file: str) -> pd.DataFrame:
    """
    加载并校验数据索引。

    读取 `index.csv` 后检查关键列名、清理无效亲和力记录，
    并统一成数据集内部使用的字段命名。

    Args:
        index_file: 数据索引文件路径。

    Returns:
        pd.DataFrame: 返回完成列名校验和基础清洗后的数据索引表。

    Raises:
        FileNotFoundError: 当依赖文件不存在时抛出。
        ValueError: 当输入参数或运行时状态不满足要求时抛出。
    """

    if not osp.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")
    if not index_file.endswith(".csv"):
        raise ValueError("Index file must be a CSV file.")

    df = pd.read_csv(index_file, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    if "Concatenated ID" not in df.columns or "Log Binding Affinity" not in df.columns:
        raise ValueError(
            "index.csv must contain the headers "
            f"'Concatenated ID' and 'Log Binding Affinity'. "
            "Do not use alternative names such as pdb_id/affinity. "
            f"Current columns: {list(df.columns)}"
        )
    df = df.rename(
        columns={
            "Concatenated ID": "pdb_id",
            "Log Binding Affinity": "affinity",
        }
    )

    initial_len = len(df)
    df = df.dropna(subset=["affinity"])
    if len(df) < initial_len:
        logger.warning(
            "Dropped %d rows with NaN affinity from index.",
            initial_len - len(df),
        )

    df["pdb_id"] = df["pdb_id"].astype(str).str.lower()
    return df


def ligand_path(pdb_id: str, pdb_dir: str) -> str | None:
    """
    解析配体文件路径。

    按既定目录约定优先查找 SDF，其次回退到 MOL2，
    为后续配体读取和预处理提供统一入口。

    Args:
        pdb_id: 复合物的 PDB 标识。
        pdb_dir: 当前复合物所在目录。

    Returns:
        str | None: 返回已定位到的配体文件路径；若不存在则返回 `None`。
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
    解析蛋白文件路径。

    根据约定的文件命名规则定位蛋白 PDB 文件，
    为预处理和图构建流程提供统一的蛋白输入路径。

    Args:
        pdb_id: 复合物的 PDB 标识。
        pdb_dir: 当前复合物所在目录。

    Returns:
        str | None: 返回已定位到的蛋白文件路径；若不存在则返回 `None`。
    """

    protein_file = osp.join(pdb_dir, f"{pdb_id}_protein.pdb")
    return protein_file if osp.exists(protein_file) else None


def esm_cache_paths(
    *,
    pdb_id: str,
    pdb_dir: str,
    esm_root: str | None,
    esm_model_name: str,
) -> tuple[str | None, str | None]:
    """
    生成 ESM 缓存路径。

    按照数据集目录和模型名约定返回缓存读写路径，
    用于定位残基嵌入的 `.npz` 文件。

    Args:
        pdb_id: 复合物的 PDB 标识。
        pdb_dir: 当前复合物所在目录。
        esm_root: ESM 缓存根目录。
        esm_model_name: ESM 主干模型名称。

    Returns:
        tuple[str | None, str | None]: 返回当前可读取的 ESM 缓存路径与建议写入的缓存路径。
    """

    cache_tag = normalize_esm_model_cache_tag(esm_model_name)
    local_path = osp.join(
        pdb_dir,
        f"{pdb_id}_{ESM_CACHE_VERSION_TAG}_{cache_tag}.npz",
    )
    if osp.exists(local_path):
        return local_path, local_path

    if esm_root and osp.isdir(esm_root):
        global_path = osp.join(
            esm_root,
            f"{pdb_id}_{ESM_CACHE_VERSION_TAG}_{cache_tag}.npz",
        )
        if osp.exists(global_path):
            return global_path, global_path
        return None, global_path

    return None, local_path

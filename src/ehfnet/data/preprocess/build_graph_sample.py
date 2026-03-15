"""
单样本预处理流程。

负责从原始蛋白配体文件构建单个图样本，
串联特征提取、图构建和缓存写入流程。
"""


import logging

from typing import TYPE_CHECKING, Any, cast

from ehfnet.data.preprocess import configure_hf_cache_env, resolve_esm_device

configure_hf_cache_env()

import MDAnalysis as mda
import torch

from torch_geometric.data import HeteroData

from ehfnet.data.datasets import load_ligand_mol
from ehfnet.data.featurizers import LigandEncoder, ProteinEncoder, load_or_compute_esm_embeddings
from ehfnet.graph import GraphBuilder

if TYPE_CHECKING:
    from esm.models.esmc import ESMC


logger = logging.getLogger(__name__)


def _ensure_hf_cache_env() -> None:
    """
    确保 HuggingFace 使用可写缓存目录。

    若用户已显式设置 `HF_HOME` 或 `HF_HUB_CACHE`，则尊重用户配置；
    否则回退到项目根目录下的 `.hf-cache`。
    """

    _, hub_cache, source = configure_hf_cache_env()
    logger.info(
        "Using HuggingFace hub cache: %s (%s)",
        hub_cache,
        source,
    )


def load_protein(protein_path: str) -> mda.Universe:
    """
    读取蛋白结构文件。

    将蛋白结构文件加载为 `MDAnalysis Universe`，
    为蛋白编码和图构建流程提供统一输入对象。

    Args:
        protein_path: 蛋白结构文件路径。

    Returns:
        mda.Universe: 返回加载后的蛋白结构对象。
    """

    return mda.Universe(protein_path)


def get_esm_model(
    model_name: str,
    *,
    device: str | torch.device | None = None,
) -> "ESMC":
    """
    加载 ESM 模型实例。

    按指定模型名创建并切换到评估模式，
    供预处理阶段的残基嵌入计算复用。

    Args:
        model_name: 待加载的模型名称。
        device: 运行所用设备，如 CPU 或 CUDA 设备。

    Returns:
        ESMC: 返回已切换到评估模式的 ESM 模型实例。
    """

    _ensure_hf_cache_env()
    resolved_device = resolve_esm_device(device)
    from esm.models.esmc import ESMC

    logger.info("Loading ESM model %s on %s", model_name, resolved_device)
    return cast("ESMC", ESMC.from_pretrained(model_name).to(resolved_device).eval())


def prepare_graph_sample(
    *,
    pdb_id: str,
    ligand_path: str,
    protein_path: str,
    affinity: float | None,
    feature_factory: Any,
    graph_builder: GraphBuilder,
    esm_cache_path: str | None,
    esm_cache_write_path: str | None,
    esm: str,
    esm_model: "ESMC | None" = None,
    esm_model_name: str,
    esm_device: str | torch.device | None = None,
) -> HeteroData:
    """
    构建单个复合物图样本。

    串联蛋白配体读取、特征提取、图构建和元数据整理流程，
    输出可直接缓存或训练使用的单样本图对象。

    Args:
        pdb_id: 复合物的 PDB 标识。
        ligand_path: 配体文件路径。
        protein_path: 蛋白结构文件路径。
        affinity: 当前样本的真实亲和力值。
        feature_factory: 负责提取化学特征的 RDKit 特征工厂。
        graph_builder: 用于构图或重建局部图的图构建器。
        esm_cache_path: ESM 缓存读取路径。
        esm_cache_write_path: ESM 缓存写入路径。
        esm: ESM 处理模式或缓存策略。
        esm_model: 已加载的 ESM 模型实例。
        esm_model_name: ESM 主干模型名称。
        esm_device: 执行 ESM 推理时使用的设备。

    Returns:
        HeteroData: 包含蛋白、配体、context 节点和元数据的单样本图对象。

    Raises:
        FileNotFoundError: 当 `esm="file"` 但缓存文件不存在时抛出。
        ValueError: 当需要写入 ESM 缓存但缺少缓存写路径时抛出。
    """

    mol = load_ligand_mol(ligand_path, remove_hs=True, require_conformer=True)
    universe = load_protein(protein_path)

    ligand_encoder = LigandEncoder(feature_factory=feature_factory)
    protein_encoder = ProteinEncoder()
    esm_embeddings = None

    if esm == "off":
        esm_embeddings = None
    elif esm == "file":
        if esm_cache_path is None:
            raise FileNotFoundError(f"ESM cache not found for {pdb_id}")
        esm_embeddings = load_or_compute_esm_embeddings(
            universe=universe,
            esm_model=None,
            cache_path=esm_cache_path,
            force_recompute=False,
        )
    else:
        if esm_cache_path is not None:
            esm_embeddings = load_or_compute_esm_embeddings(
                universe=universe,
                esm_model=None,
                cache_path=esm_cache_path,
                force_recompute=False,
            )
        else:
            if esm_cache_write_path is None:
                raise ValueError(
                    f"ESM cache_write_path is required for esm=auto ({pdb_id})"
                )
            if esm_model is None:
                esm_model = get_esm_model(
                    model_name=esm_model_name,
                    device=esm_device,
                )
            esm_embeddings = load_or_compute_esm_embeddings(
                universe=universe,
                esm_model=esm_model,
                cache_path=esm_cache_write_path,
                force_recompute=False,
            )

    ligand_result = ligand_encoder.encode(mol, strict_torsion=True)
    protein_result = protein_encoder.encode(
        universe,
        esm_embeddings=esm_embeddings,
    )

    data = graph_builder.build(ligand_result, protein_result)
    data.pdb_id = pdb_id
    data.ligand_sanitize_mode = (
        mol.GetProp("_ehfnet_sanitize_mode")
        if mol.HasProp("_ehfnet_sanitize_mode")
        else "unknown"
    )
    data.ligand_partial_sanitize = bool(data.ligand_sanitize_mode == "partial")
    data.ligand_full_sanitize_flag = (
        int(mol.GetProp("_ehfnet_full_sanitize_flag"))
        if mol.HasProp("_ehfnet_full_sanitize_flag")
        else -1
    )
    data.ligand_partial_sanitize_flag = (
        int(mol.GetProp("_ehfnet_partial_sanitize_flag"))
        if mol.HasProp("_ehfnet_partial_sanitize_flag")
        else -1
    )

    if affinity is not None:
        data.y_energy = torch.tensor([affinity], dtype=torch.float32)

    return data

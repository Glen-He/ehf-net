"""
ESM 蛋白质语言模型嵌入

提供 ESM 模型的嵌入计算、缓存和加载功能。
"""

import torch
import logging
import numpy as np

from pathlib import Path
from typing import Any, cast
from MDAnalysis import Universe
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from ehfnet.encoders.chemistry import ResidueType
from ehfnet.encoders.protein_segments import segment_residues_by_continuity


logger = logging.getLogger(__name__)


def extract_protein_chain_sequences(universe: Universe) -> list[tuple[str, str, list[int]]]:
    """
    从 Universe 中提取按真实肽链连续性切分的蛋白质链段序列及 residue.ix。

    不再直接按 segment 拼接整条蛋白，而是基于真实链标签和 backbone 连续性切段，
    避免多链或断链时把不存在的序列上下文硬拼给 ESM。

    Args:
        universe: MDAnalysis Universe 对象

    Returns:
        包含 (segment_key, sequence, residue_ixs) 的列表
    """

    protein_residues = list(universe.select_atoms("protein").residues)
    if not protein_residues:
        return []

    out: list[tuple[str, str, list[int]]] = []

    for segment in segment_residues_by_continuity(protein_residues):
        seq_list: list[str] = []
        residue_ixs: list[int] = []

        for res in segment.residues:
            res_type = ResidueType.safe_get(res.resname)

            if res_type == ResidueType.UNK:
                logger.warning(
                    f"Unknown residue '{res.resname}' at ix={res.ix} in segment {segment.key}, "
                    f"mapped to UNK ('{res_type.one_letter}')"
                )

            seq_list.append(res_type.one_letter)
            residue_ixs.append(int(res.ix))

        out.append((segment.key, "".join(seq_list), residue_ixs))

    return out


def embeddings_to_residue_ix_map(residue_ixs: list[int], embeddings: np.ndarray) -> dict[int, np.ndarray]:
    """
    将 ESM 输出的 Embeddings 映射回残基索引 (ix)

    Args:
        residue_ixs: 残基索引列表 (长度 L)
        embeddings: 对应的 Embeddings 矩阵 (形状 [L, D])

    Returns:
        {residue_ix: embedding_vector} 字典
    
    Raises:
        ValueError: 如果索引长度与 Embeddings 维度不匹配
    """

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings with shape [L, D], got {embeddings.shape}")
    
    # 强制安全检查：这是保证 GNN 节点特征对齐的最重要防线
    if len(residue_ixs) != embeddings.shape[0]:
        raise ValueError(
            f"Alignment error: Residue indices count ({len(residue_ixs)}) "
            f"does not match embedding count ({embeddings.shape[0]})."
        )

    # 使用 copy() 确保返回的 numpy 数组在内存中是独立的，防止后续修改影响源数据
    return {ix: embeddings[i].copy() for i, ix in enumerate(residue_ixs)}


def compute_esm_embeddings(universe: Universe, esm_model: ESMC) -> dict[int, np.ndarray]:
    """
    计算蛋白质的 ESM 嵌入
    
    流程：
    1. 提取序列与索引 (extract_protein_chain_sequences)
    2. ESM 模型推理
    3. 严格对齐映射 (embeddings_to_residue_ix_map)
    
    Args:
        universe: MDAnalysis Universe 对象
        esm_model: ESM 模型实例
        
    Returns:
        {residue_index: embedding} 字典，embedding 形状为 (960,)（ESMC-300M）
    """

    esm_map: dict[int, np.ndarray] = {}
    
    # 1. 数据准备阶段
    chains = extract_protein_chain_sequences(universe)
    
    if not chains:
        logger.warning("No protein segments found in universe")
        return esm_map

    device = next(esm_model.parameters()).device

    # 2. 推理阶段
    with torch.no_grad():

        for seg_idx, (segid, sequence, residue_ixs) in enumerate(chains):

            if not sequence:
                continue

            try:
                # 构造 ESM 输入
                protein_for_esm = ESMProtein(sequence=sequence)
                protein_tensor = esm_model.encode(protein_for_esm)
                
                if hasattr(protein_tensor, "to"):
                    protein_tensor = protein_tensor.to(device)

                logits_output = esm_model.logits(
                    protein_tensor, LogitsConfig(return_embeddings=True)
                )

                embeddings_tensor = logits_output.embeddings
                assert embeddings_tensor is not None, "ESM model returned None embeddings"
                raw_embeddings = embeddings_tensor.squeeze(0)

                # 智能处理 BOS/EOS tokens
                # ESM 通常会在序列首尾添加特殊 token，需要根据长度自动识别并切除
                expected_len_with_tokens = len(residue_ixs) + 2
                expected_len_without = len(residue_ixs)

                if raw_embeddings.shape[0] == expected_len_with_tokens:
                    residue_embeddings = raw_embeddings[1:-1]
                elif raw_embeddings.shape[0] == expected_len_without:
                    residue_embeddings = raw_embeddings
                else:
                    logger.error(
                        f"ESM embedding shape mismatch for segment {seg_idx}: "
                        f"expected {expected_len_with_tokens} or {expected_len_without}, "
                        f"got {raw_embeddings.shape[0]}"
                    )
                    continue

                # 3. 结果映射阶段
                cpu_embeddings = residue_embeddings.float().cpu().numpy()
                
                # 调用辅助函数进行安全映射和数据对齐
                segment_map = embeddings_to_residue_ix_map(residue_ixs, cpu_embeddings)
                esm_map.update(segment_map)

                logger.info(
                    f"Computed ESM embeddings for segment {seg_idx} "
                    f"({len(residue_ixs)} residues, segid={segid})"
                )

            except Exception as e:
                logger.error(
                    f"ESM inference failed for segment {seg_idx} (segid={segid}): "
                    f"{type(e).__name__}: {e}",
                    exc_info=True,
                )

    return esm_map


def save_esm_embeddings(embeddings: dict[int, np.ndarray], output_path: str | Path) -> None:
    """
    保存 ESM embeddings 到 .npz 文件
    
    Args:
        embeddings: {residue_index: embedding} 字典
        output_path: 输出文件路径
    """
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # npz 要求 key 为字符串
    str_key_dict = {str(k): v for k, v in embeddings.items()}
    np.savez_compressed(str(output_path), **cast(dict[str, Any], str_key_dict))
    
    logger.info(f"Saved {len(embeddings)} ESM embeddings to {output_path}")


def load_esm_embeddings(input_path: str | Path) -> dict[int, np.ndarray]:
    """
    从 .npz 文件加载 ESM embeddings
    
    Args:
        input_path: 输入文件路径
        
    Returns:
        {residue_index: embedding} 字典
    """

    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"ESM embedding file not found: {input_path}")

    with np.load(input_path) as data:
        # 加载时将 key 转回 int，适配 GNN 的索引需求
        embeddings = {int(k): data[k].copy() for k in data.files}
    
    logger.info(f"Loaded {len(embeddings)} ESM embeddings from {input_path}")
    return embeddings


def cache_esm_embeddings(
    universe: Universe, esm_model: ESMC, output_path: str | Path
) -> dict[int, np.ndarray]:
    """
    计算并保存 ESM embeddings
    
    Args:
        universe: MDAnalysis Universe 对象
        esm_model: ESM 模型实例
        output_path: 输出文件路径
        
    Returns:
        计算得到的 embeddings
    """

    embeddings = compute_esm_embeddings(universe, esm_model)
    save_esm_embeddings(embeddings, output_path)
    return embeddings


def load_or_compute_esm_embeddings(
    universe: Universe,
    esm_model: ESMC | None,
    cache_path: str | Path,
    force_recompute: bool = False,
) -> dict[int, np.ndarray]:
    """
    智能加载 ESM embeddings：优先从缓存读取，不存在则计算
    
    Args:
        universe: MDAnalysis Universe 对象
        esm_model: ESM 模型实例（计算时需要）
        cache_path: 缓存文件路径
        force_recompute: 强制重新计算
        
    Returns:
        ESM embeddings 字典
    """

    cache_path = Path(cache_path)

    # 尝试从缓存加载
    if cache_path.exists() and not force_recompute:
        
        try:
            return load_esm_embeddings(cache_path)

        except Exception as e:
            logger.warning(
                f"Failed to load cached embeddings from {cache_path}: {e}. Will recompute."
            )

    # 需要计算但没有模型
    if esm_model is None:

        raise ValueError(
            f"ESM model required to compute embeddings, but got None. "
            f"Cache file {cache_path} does not exist or force_recompute=True."
        )

    # 计算并保存
    logger.info(f"Computing ESM embeddings (cache: {cache_path})")
    return cache_esm_embeddings(universe, esm_model, cache_path)

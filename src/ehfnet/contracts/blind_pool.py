"""
Blind pool 契约定义。

集中声明候选池缓存使用的字段名、目录名和版本签名，
保证候选池文件格式在读写两端保持一致。
"""


import os

from typing import Any

from ehfnet.contracts.checkpoint import build_feature_signature


BLIND_POOL_SCHEMA_TAG = "blind_pool_context_rigid"


def build_blind_pool_signature(
    *,
    esm_dim: int,
    processed_dir: str,
    index_file: str,
    interaction_profile: str,
) -> dict[str, Any]:
    """
    构建 blind pool 签名字典。

    将候选池缓存依赖的关键结构约定整理为可持久化字典，
    供候选池写入和读取时校验当前缓存是否仍对应同一套数据与特征契约。

    Args:
        esm_dim: ESM 残基嵌入维度。
        processed_dir: processed 数据目录路径。
        index_file: 数据索引文件路径。
        interaction_profile: 跨图交互拓扑配置。

    Returns:
        dict[str, Any]: 返回描述 blind pool 缓存契约的签名字典。
    """
    return {
        "blind_pool_schema": BLIND_POOL_SCHEMA_TAG,
        "feature_signature": build_feature_signature(esm_dim=esm_dim),
        "processed_dir": os.path.abspath(processed_dir),
        "index_file": os.path.abspath(index_file),
        "interaction_profile": str(interaction_profile),
    }

"""
模型层入口。

导出编码器、嵌入层和基础算子，
统一底层网络组件的访问路径。
"""


from importlib import import_module

__all__ = [
    "AtomEmbedding",
    "LigandAtomEmbedding",
    "LigandMoleculeEmbedding",
    "ProteinAtomEmbedding",
    "ProteinContextEmbedding",
    "ProteinResidueEmbedding",
    "TimeEmbedding",
    "EHFEncoder",
    "FrameAwareConv",
    "FrameAwareHeteroConv",
    "GaussianRBF",
]

_EXPORT_MAP = {
    "AtomEmbedding": ("ehfnet.models.layers.embeddings", "AtomEmbedding"),
    "LigandAtomEmbedding": ("ehfnet.models.layers.embeddings", "LigandAtomEmbedding"),
    "LigandMoleculeEmbedding": (
        "ehfnet.models.layers.embeddings",
        "LigandMoleculeEmbedding",
    ),
    "ProteinAtomEmbedding": ("ehfnet.models.layers.embeddings", "ProteinAtomEmbedding"),
    "ProteinContextEmbedding": (
        "ehfnet.models.layers.embeddings",
        "ProteinContextEmbedding",
    ),
    "ProteinResidueEmbedding": (
        "ehfnet.models.layers.embeddings",
        "ProteinResidueEmbedding",
    ),
    "TimeEmbedding": ("ehfnet.models.layers.embeddings", "TimeEmbedding"),
    "EHFEncoder": ("ehfnet.models.layers.encoder", "EHFEncoder"),
    "FrameAwareConv": ("ehfnet.models.layers.frame_conv", "FrameAwareConv"),
    "FrameAwareHeteroConv": (
        "ehfnet.models.layers.frame_conv",
        "FrameAwareHeteroConv",
    ),
    "GaussianRBF": ("ehfnet.models.layers.rbf", "GaussianRBF"),
}


def __getattr__(name: str):
    """
    按名称返回公开对象。

    仅在首次访问时执行真实导入，
    用于避免包初始化阶段触发重模块加载或循环依赖。

    Args:
        name: 请求访问或解析的公开对象名称。

    Returns:
        object: 返回与名称对应的惰性导出对象。

    Raises:
        AttributeError: 当访问的属性不存在或对象不满足接口约定时抛出。
    """

    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = import_module(module_name, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

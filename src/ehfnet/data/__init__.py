"""
数据模块入口。

导出数据集、特征化和预处理子模块，
作为数据侧统一的公开访问入口。
"""


__all__ = [
    "ProteinLigandDataset",
]


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

    if name == "ProteinLigandDataset":
        from ehfnet.data.datasets import ProteinLigandDataset

        return ProteinLigandDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

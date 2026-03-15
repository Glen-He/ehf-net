"""
预测头模块入口。

导出预测相关子模块，
统一模型输出头的公开访问方式。
"""


from importlib import import_module

__all__ = [
    "CosineCutoff",
    "PredictionConstants",
    "PredictionHead",
]

_EXPORT_MAP = {
    "CosineCutoff": ("ehfnet.models.heads.prediction", "CosineCutoff"),
    "PredictionConstants": ("ehfnet.models.heads.prediction", "PredictionConstants"),
    "PredictionHead": ("ehfnet.models.heads.prediction", "PredictionHead"),
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

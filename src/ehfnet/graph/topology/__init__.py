"""
图拓扑模块入口。

导出图内、层级与跨图拓扑构建工具，
统一拓扑侧能力的公开访问路径。
"""


from importlib import import_module

__all__ = [
    "build_aggregate_edges",
    "build_broadcast_edges",
    "build_intra_edges",
    "build_same_type_radius_or_knn_edges",
    "build_static_inter_edges",
]

_EXPORT_MAP = {
    "build_aggregate_edges": (
        "ehfnet.graph.topology.hierarchy",
        "build_aggregate_edges",
    ),
    "build_broadcast_edges": (
        "ehfnet.graph.topology.hierarchy",
        "build_broadcast_edges",
    ),
    "build_intra_edges": ("ehfnet.graph.topology.intra", "build_intra_edges"),
    "build_same_type_radius_or_knn_edges": (
        "ehfnet.graph.topology.intra",
        "build_same_type_radius_or_knn_edges",
    ),
    "build_static_inter_edges": (
        "ehfnet.graph.topology.inter",
        "build_static_inter_edges",
    ),
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

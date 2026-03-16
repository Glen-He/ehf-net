"""
构图模块入口。

导出构图、裁剪、特征和拓扑工具，
统一图处理相关能力的公开接口。
"""


from importlib import import_module

__all__ = [
    "ESMEmbeddingFiller",
    "GraphBuilder",
    "create_graph_tools",
    "GraphCollator",
    "compute_ligand_center",
    "crop_graph_to_center",
    "build_context_features",
    "context_feature_dim",
    "build_graph_cost_profile",
    "estimate_dynamic_edge_upper_bounds",
    "estimate_graph_cost_units",
    "build_batched_bipartite_knn_edges",
    "build_batched_radius_or_knn_edges",
    "AGGREGATE_EDGES",
    "ATOM_NODE_TYPES",
    "BROADCAST_EDGES",
    "DYNAMIC_INTER_EDGES",
    "INTER_EDGES",
    "INTRA_EDGES",
    "NODE_TYPES",
    "STATIC_INTER_EDGES",
]

_EXPORT_MAP = {
    "ESMEmbeddingFiller": (
        "ehfnet.graph.builders.hetero_graph_builder",
        "ESMEmbeddingFiller",
    ),
    "GraphBuilder": ("ehfnet.graph.builders.hetero_graph_builder", "GraphBuilder"),
    "create_graph_tools": (
        "ehfnet.graph.builders.hetero_graph_builder",
        "create_graph_tools",
    ),
    "GraphCollator": ("ehfnet.graph.collate", "GraphCollator"),
    "compute_ligand_center": ("ehfnet.graph.crop.runtime_crop", "compute_ligand_center"),
    "crop_graph_to_center": ("ehfnet.graph.crop.runtime_crop", "crop_graph_to_center"),
    "build_context_features": (
        "ehfnet.graph.features.protein_context",
        "build_context_features",
    ),
    "context_feature_dim": ("ehfnet.graph.features.protein_context", "context_feature_dim"),
    "build_graph_cost_profile": ("ehfnet.graph.costs", "build_graph_cost_profile"),
    "estimate_dynamic_edge_upper_bounds": (
        "ehfnet.graph.costs",
        "estimate_dynamic_edge_upper_bounds",
    ),
    "estimate_graph_cost_units": ("ehfnet.graph.costs", "estimate_graph_cost_units"),
    "build_batched_bipartite_knn_edges": (
        "ehfnet.graph.inter_edges",
        "build_batched_bipartite_knn_edges",
    ),
    "build_batched_radius_or_knn_edges": (
        "ehfnet.graph.inter_edges",
        "build_batched_radius_or_knn_edges",
    ),
    "AGGREGATE_EDGES": ("ehfnet.graph.schema", "AGGREGATE_EDGES"),
    "ATOM_NODE_TYPES": ("ehfnet.graph.schema", "ATOM_NODE_TYPES"),
    "BROADCAST_EDGES": ("ehfnet.graph.schema", "BROADCAST_EDGES"),
    "DYNAMIC_INTER_EDGES": ("ehfnet.graph.schema", "DYNAMIC_INTER_EDGES"),
    "INTER_EDGES": ("ehfnet.graph.schema", "INTER_EDGES"),
    "INTRA_EDGES": ("ehfnet.graph.schema", "INTRA_EDGES"),
    "NODE_TYPES": ("ehfnet.graph.schema", "NODE_TYPES"),
    "STATIC_INTER_EDGES": ("ehfnet.graph.schema", "STATIC_INTER_EDGES"),
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

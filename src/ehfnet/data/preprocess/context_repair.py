"""
上下文修复工具。

负责补齐旧缓存或异常样本中的 context 节点，
保证图结构满足当前运行时的字段约定。
"""


from torch_geometric.data import HeteroData

from ehfnet.graph import GraphBuilder, context_feature_dim


def ensure_context_features(data: HeteroData, graph_builder: GraphBuilder) -> HeteroData:
    """
    补齐 context 节点特征。

    检查缓存图中的 `protein_context` 是否符合当前 schema，
    必要时重新构建该节点的连续特征以保持兼容。

    Args:
        data: 当前处理的图数据对象。
        graph_builder: 用于构图或重建局部图的图构建器。

    Returns:
        HeteroData: 返回处理后的图对象。

    Raises:
        ValueError: 当输入参数或运行时状态不满足要求时抛出。
    """

    if "protein_context" not in data.node_types:
        raise ValueError(
            "Cached graph is incompatible with the current local-context schema. "
            "Please rebuild processed graph cache."
        )

    expected_context_dim = context_feature_dim(int(data["protein_residue"].x_cont.size(1)))
    context_x_cont = getattr(data["protein_context"], "x_cont", None)
    if (
        context_x_cont is None
        or context_x_cont.ndim != 2
        or int(context_x_cont.size(1)) != expected_context_dim
    ):
        data["protein_context"].x_cont = graph_builder.build_context_node_features(data)
        data["protein_context"].num_nodes = int(data["protein_context"].x_cont.size(0))

    return data

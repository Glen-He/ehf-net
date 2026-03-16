"""
推理工具入口。

导出盲对接推理共享组件，
统一推理侧公开接口的访问方式。
"""


from importlib import import_module

__all__ = [
    "DEFAULT_FUSION_WEIGHTS",
    "calibrate_linear_fusion_weights",
    "combine_center_pose_score",
    "compute_center_guidance_scores",
    "evaluate_topn_success",
    "predict_center_proposal_logits",
    "select_diverse_center_indices",
    "summarize_blind_candidate_records",
]

_EXPORT_MAP = {
    "DEFAULT_FUSION_WEIGHTS": (
        "ehfnet.training.inference.center_utils",
        "DEFAULT_FUSION_WEIGHTS",
    ),
    "calibrate_linear_fusion_weights": (
        "ehfnet.training.inference.metrics",
        "calibrate_linear_fusion_weights",
    ),
    "combine_center_pose_score": (
        "ehfnet.training.inference.center_utils",
        "combine_center_pose_score",
    ),
    "compute_center_guidance_scores": (
        "ehfnet.training.inference.center_utils",
        "compute_center_guidance_scores",
    ),
    "evaluate_topn_success": (
        "ehfnet.training.inference.evaluator",
        "evaluate_topn_success",
    ),
    "predict_center_proposal_logits": (
        "ehfnet.training.inference.center_utils",
        "predict_center_proposal_logits",
    ),
    "select_diverse_center_indices": (
        "ehfnet.training.inference.center_utils",
        "select_diverse_center_indices",
    ),
    "summarize_blind_candidate_records": (
        "ehfnet.training.inference.metrics",
        "summarize_blind_candidate_records",
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

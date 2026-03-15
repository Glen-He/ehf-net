"""
缓存契约定义。

集中声明图缓存、ESM 缓存和预处理元数据使用的目录与版本标识，
避免缓存路径和签名在不同模块中重复散落。
"""


ESM_CACHE_VERSION_TAG = "esm_chainseg"
GRAPH_CACHE_SCHEMA_TAG = "graph_cache_context_rigid"
GRAPH_CACHE_DIRNAME = "cache"
PREPROCESS_SUMMARY_FILENAME = "preprocess_summary.json"
PREPROCESS_METADATA_DIRNAME = "_preprocess_meta"

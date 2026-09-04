"""DEPRECATED: use pipelines.paper_factor.nodes.quality directly.

Phase 1 兼容垫片（注意旧名 quality_node 对应新文件 quality.py 中的同名函数）。
"""
from pipelines.paper_factor.nodes.quality import quality_node

__all__ = ["quality_node"]
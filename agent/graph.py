"""
agent/graph.py - DEPRECATED shim

Phase 1：原 build_graph(...) 实现已搬到 harness/graph_builder.py。
这里改为薄壳调用，保持旧 import 路径继续工作。

用法（向后兼容）：
    from agent.graph import build_graph
    graph, cp = build_graph(interrupt_before_decide=True)

新用法（推荐）：
    from harness.registry import get
    from harness.graph_builder import build_graph
    pipeline = get("paper_factor")
    graph, cp = build_graph(pipeline)
"""
import sys
import warnings


def build_graph(interrupt_before_decide: bool = True):
    """
    兼容旧 API。

    - interrupt_before_decide=True（默认）：interactive 模式，interrupt_before=["decide"]
    - interrupt_before_decide=False：scheduler 模式，不中断
    """
    from harness.registry import get as _registry_get
    from harness.graph_builder import build_graph as _harness_build

    pipeline = _registry_get("paper_factor")
    if pipeline is None:
        raise RuntimeError("paper_factor pipeline not loaded")

    interrupt_before = ["decide"] if interrupt_before_decide else []
    return _harness_build(pipeline, interrupt_before=interrupt_before)


def get_default_graph(interrupt_before_decide: bool = True):
    """惰性初始化默认图（兼容旧 API）。"""
    return build_graph(interrupt_before_decide=interrupt_before_decide)[0]


# 一开始打一行 deprecation 警告（只在第一次 import 时打一次）
_warned = False

def _warn():
    global _warned
    if not _warned:
        warnings.warn(
            "agent.graph is deprecated. Use harness.graph_builder.build_graph(pipeline).",
            DeprecationWarning,
            stacklevel=2,
        )
        _warned = True

_warn()
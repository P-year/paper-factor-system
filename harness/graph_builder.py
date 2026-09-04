"""
harness/graph_builder.py - 通用 StateGraph 构造器

用法：
    from harness.registry import get
    from harness.graph_builder import build_graph

    pipeline = get("paper_factor")
    graph, checkpointer = build_graph(pipeline, interrupt_before=["decide"])

    # 或按名：
    graph, checkpointer = build_graph("paper_factor")

行为（与原 agent.graph.build_graph 等价）：
- START → plan
- plan → conditional via _route_from_plan(pipeline, state)
- 非 terminal 节点 → plan（循环）
- terminal 节点 → END
- interrupt_before 取 yaml 默认 + 显式传入的并集
"""
from typing import List, Optional, Tuple, Union

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from harness.pipeline import PipelineConfig
from harness.registry import get as _registry_get


def _route_from_plan(pipeline: PipelineConfig, state: dict) -> str:
    """plan 节点写入新决策到 state.plan[-1]，按 node 字段路由。

    - 未知节点 → END（兜底）
    - terminal_nodes → END
    - 其他 → 该节点名
    - iteration_count 超限 → END（与 plan_node 内部一致）
    """
    if not state.get("plan"):
        return END
    last = state["plan"][-1]
    next_node = last.get("node", "respond")

    if next_node not in pipeline.nodes:
        return END
    if next_node in pipeline.terminal_nodes:
        return END
    if state.get("iteration_count", 0) >= pipeline.max_iter:
        return END
    return next_node


def build_graph(
    pipeline_or_name: Union[PipelineConfig, str],
    *,
    interrupt_before: Optional[List[str]] = None,
) -> Tuple:
    """
    构造一个 StateGraph 实例 + MemorySaver checkpointer。

    Args:
        pipeline_or_name: PipelineConfig 实例 或 pipeline 名（从 registry 拉）
        interrupt_before: 显式指定的 interrupt 节点列表；与 yaml 默认取并集。

    Returns:
        (compiled_graph, checkpointer) 元组
    """
    if isinstance(pipeline_or_name, str):
        pipeline = _registry_get(pipeline_or_name)
        if pipeline is None:
            raise ValueError(f"Pipeline '{pipeline_or_name}' not found in registry")
    else:
        pipeline = pipeline_or_name

    state_schema = pipeline.state_schema

    # 构造 StateGraph
    g = StateGraph(state_schema)

    # 注册所有节点（key = node 名，value = callable(state)->dict）
    for name, fn in pipeline.nodes.items():
        g.add_node(name, fn)

    # START → plan
    g.add_edge(START, "plan")

    # plan → conditional（基于 state.plan[-1].node）
    def _route(state):
        return _route_from_plan(pipeline, state)

    # 条件边映射：节点名 → 节点名；END（字符串 "__end__"） → END
    route_map = {n: n for n in pipeline.nodes}
    route_map[END] = END

    g.add_conditional_edges(
        "plan",
        _route,
        route_map,
    )

    # 非 terminal 节点 → plan（loop back）
    terminal_set = set(pipeline.terminal_nodes)
    for name in pipeline.nodes:
        if name == "plan" or name in terminal_set:
            continue
        g.add_edge(name, "plan")

    # terminal 节点 → END
    for name in terminal_set:
        if name in pipeline.nodes:
            g.add_edge(name, END)

    # interrupt_before：yaml 默认 + 显式传入的并集
    ib = set(pipeline.interrupt_before)
    if interrupt_before:
        ib.update(interrupt_before)

    checkpointer = MemorySaver()
    compiled = g.compile(
        checkpointer=checkpointer,
        interrupt_before=list(ib) if ib else None,
    )

    return compiled, checkpointer
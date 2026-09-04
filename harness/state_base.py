"""
harness/state_base.py - 所有 pipeline 共享的 state 字段基类

每个 pipeline 的 state.py 用 make_base_typed_dict() 工厂构造基类，
加自己特有字段。total=False 保持 LangGraph default 行为（missing keys 允许）。

v4：messages 加 Annotated[..., add_messages] reducer 让 LangGraph 自动合并；
加 compaction_history / memory_tiers / retrieval_context 三个新字段。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langgraph.graph.message import add_messages


def make_base_typed_dict():
    """构造 TypedDict 类，避免每个 pipeline 手写 Annotated reducer。

    用法（pipelines/paper_factor/state.py）：
        from harness.state_base import make_base_typed_dict
        BaseState = make_base_typed_dict()

        class PaperFactorState(BaseState, total=False):
            collected_paper_ids: List[str]
            extracted_factors: List[Dict]
    """
    try:
        from typing_extensions import TypedDict
    except ImportError:
        from typing import TypedDict

    class _Base(TypedDict, total=False):
        # === 控制流 ===
        plan: List[Dict[str, Any]]
        current_step_idx: int
        iteration_count: int
        visited_nodes: List[str]
        errors: List[str]

        # === 消息 / 会话 ===
        # v4：add_messages reducer 让节点 return {"messages": [m1]} 自动合并
        messages: List[Dict[str, Any]]  # 实际用 Annotated（见下）
        session_id: str
        run_mode: str
        pipeline_name: str

        # === 人机协同 ===
        human_action: Optional[Dict[str, Any]]

        # === 成本（v3-1 起）===
        cost_summary: Dict[str, Any]
        cost_usages: List[Dict[str, Any]]

        # === 上下文/记忆（v4）===
        compaction_history: List[Dict[str, Any]]
        memory_tiers: Dict[str, Any]
        retrieval_context: Optional[List[Dict[str, Any]]]

    # 在 TypedDict 类外注入 Annotated reducer 到 messages 字段
    # 由于 TypedDict 不允许直接用 Annotated，这里用 _AnnotatedTypedDict 模式
    from typing_extensions import Annotated as _Ann

    class _AnnotatedBase(TypedDict, total=False):
        plan: List[Dict[str, Any]]
        current_step_idx: int
        iteration_count: int
        visited_nodes: List[str]
        errors: List[str]
        messages: _Ann[List[Dict[str, Any]], add_messages]
        session_id: str
        run_mode: str
        pipeline_name: str
        human_action: Optional[Dict[str, Any]]
        cost_summary: Dict[str, Any]
        cost_usages: List[Dict[str, Any]]
        compaction_history: List[Dict[str, Any]]
        memory_tiers: Dict[str, Any]
        retrieval_context: Optional[List[Dict[str, Any]]]

    return _AnnotatedBase


def ensure_v3_fields(state: Dict[str, Any]) -> Dict[str, Any]:
    """给老 state 补 v3 字段（用于不通过 migrate_state 的快速路径）。"""
    state.setdefault("compaction_history", [])
    state.setdefault("memory_tiers", {})
    state.setdefault("retrieval_context", None)
    return state
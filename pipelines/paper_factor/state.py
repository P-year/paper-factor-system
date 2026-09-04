"""
pipelines/paper_factor/state.py - PaperFactorState

v4：用 make_base_typed_dict() 工厂构造带 reducer 的基类。
继承 + 加 paper_factor 特有的字段。
"""
from typing import Any, Dict, List, Optional

from harness.state_base import make_base_typed_dict

BaseState = make_base_typed_dict()


class PaperFactorState(BaseState, total=False):
    # === 业务数据 ===
    collected_paper_ids: List[str]
    extracted_factors: List[Dict[str, Any]]
    persisted_count: int
    persist_attempted: bool
    quality_results: List[Dict[str, Any]]
    backtest_results: List[Dict[str, Any]]

    # === 审批 ===
    pending_decisions: List[Dict[str, Any]]


# 旧名兼容：AgentState
AgentState = PaperFactorState


# 常量（保留兼容）
MAX_ITER = 8
AVAILABLE_NODES = [
    "collect",
    "analyze",
    "persist_factors",
    "quality_check",
    "backtest",
    "decide",
    "respond",
]
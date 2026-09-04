"""
agent/state.py - DEPRECATED shim

Phase 1：原 AgentState 搬到 pipelines/paper_factor/state.py:PaperFactorState。
这里改为薄壳 re-export，保持旧 import 路径继续工作。
"""
from pipelines.paper_factor.state import (
    PaperFactorState,
    AgentState,            # 老名
    MAX_ITER,
    AVAILABLE_NODES,
)

__all__ = ["PaperFactorState", "AgentState", "MAX_ITER", "AVAILABLE_NODES"]
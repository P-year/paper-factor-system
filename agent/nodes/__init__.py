"""
agent/nodes - LangGraph 节点实现
"""
from .plan_node import plan_node
from .respond_node import respond_node
from .collect_node import collect_node
from .analyze_node import analyze_node
from .persist_factors_node import persist_factors_node
from .quality_node import quality_node
from .backtest_node import backtest_node
from .decide_node import decide_node

ALL_NODES = {
    "plan": plan_node,
    "collect": collect_node,
    "analyze": analyze_node,
    "persist_factors": persist_factors_node,
    "quality_check": quality_node,
    "backtest": backtest_node,
    "decide": decide_node,
    "respond": respond_node,
}

__all__ = list(ALL_NODES.keys()) + ["ALL_NODES"]
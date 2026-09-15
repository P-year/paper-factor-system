"""
pipelines/paper_factor/nodes/__init__.py - 节点集合

plan_node 在 pipeline 配置加载后动态绑定；其他 7 个直接 import。
build_nodes(pipeline) 是注册入口，调用方（pipelines/paper_factor/__init__.py）传 pipeline。
"""
from typing import Callable, Dict

from .collect import collect_node
from .analyze import analyze_node
from .persist_factors import persist_factors_node
from .quality import quality_node
from .backtest import backtest_node
from .decide import decide_node
from .respond import respond_node
from .paper_retrieve import paper_retrieve_node    # v5
from .plan import make_paper_factor_plan_node


__all__ = ["build_nodes", "collect_node", "analyze_node", "persist_factors_node",
           "quality_node", "backtest_node", "decide_node", "respond_node",
           "paper_retrieve_node", "make_paper_factor_plan_node"]


def build_nodes(pipeline) -> Dict[str, Callable]:
    """根据 pipeline 配置构造完整节点 dict（含 plan_node）。

    注意 key 名与 config.yaml:hard_rules 的 force 字段保持一致：
      plan / collect / analyze / persist_factors / quality_check / backtest / decide / respond
      paper_retrieve（v5）
    """
    plan_node = make_paper_factor_plan_node(pipeline)
    return {
        "plan": plan_node,
        "collect": collect_node,
        "paper_retrieve": paper_retrieve_node,    # v5
        "analyze": analyze_node,
        "persist_factors": persist_factors_node,
        "quality_check": quality_node,
        "backtest": backtest_node,
        "decide": decide_node,
        "respond": respond_node,
    }
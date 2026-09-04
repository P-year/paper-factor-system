"""
plan_node for paper_factor pipeline

由 harness.plan_node.make_plan_node 工厂生成，绑定 paper_factor 的：
- orchestrator prompt（来自 yaml）
- hard_rules（来自 yaml）
- tool list（从 agent.tools 聚合）

注：plan_node 必须在 pipeline 配置加载完成之后才能构造（要绑定 prompts/hard_rules），
所以这里只暴露工厂函数，节点注册由 nodes/__init__.py:build_nodes(pipeline) 完成。
"""
from typing import Callable

from harness.plan_node import make_plan_node
from harness.pipeline import PipelineConfig


def make_paper_factor_plan_node(pipeline: PipelineConfig) -> Callable:
    """根据 pipeline 配置构造 plan_node。"""
    return make_plan_node(pipeline)
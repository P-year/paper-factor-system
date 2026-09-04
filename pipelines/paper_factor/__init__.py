"""
pipelines/paper_factor/__init__.py - 注册 paper_factor pipeline

两阶段构造：
  Phase 1: 从 yaml 读声明部分（prompts / hard_rules / max_iter ...），构造 PipelineConfig（nodes 占位空 dict）
  Phase 2: 用 pipeline 实例化 plan_node（绑定 prompts/hard_rules），生成完整 nodes dict，覆盖到 pipeline

harness.registry.load_all() 在工厂返回 PipelineConfig 后立即跑 Phase 2。

用法：
    from harness.registry import get
    pipeline = get("paper_factor")        # Phase 1+2 已完成
    from harness.graph_builder import build_graph
    graph, cp = build_graph(pipeline)
"""
from pathlib import Path

from harness.pipeline import PipelineConfig, load_yaml_config
from harness.registry import register_pipeline


@register_pipeline
def _register_paper_factor():
    # Phase 1：声明部分
    from pipelines.paper_factor.state import PaperFactorState
    from pipelines.paper_factor.auto_approve import auto_approve

    yaml_path = Path(__file__).parent / "config.yaml"
    raw = load_yaml_config(yaml_path)

    pipeline = PipelineConfig(
        name=raw["name"],
        yaml_path=yaml_path,
        version=raw.get("version", 1),
        state_schema=PaperFactorState,
        nodes={},  # 占位，Phase 2 覆盖
        auto_approve_fn=auto_approve,
        max_iter=raw.get("max_iter", 8),
        interrupt_before=list(raw.get("interrupt_before", [])),
        terminal_nodes=list(raw.get("terminal_nodes", ["respond"])),
        tool_modules=list(raw.get("tool_modules", [])),
        prompts=dict(raw.get("prompts", {})),
        hard_rules=list(raw.get("hard_rules", [])),
    )

    # Phase 2：用 pipeline 实例化 plan_node，注入完整 nodes
    from pipelines.paper_factor.nodes import build_nodes
    pipeline.nodes = build_nodes(pipeline)

    return pipeline
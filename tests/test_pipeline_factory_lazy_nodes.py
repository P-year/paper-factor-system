"""
test_pipeline_factory_lazy_nodes.py - 验证 factory 时序 bug 修复

原 bug：make_plan_node 在 pipeline.nodes={} 时被调用，导致 node_names 集合永远为空。
任何 next_node 都会触发"unknown next_node, forced respond"。

修复：plan_node 内部每次从 pipeline.nodes 重新查 node_names（lazy binding）。

注：此 bug 只发生在"pipeline factory 内部"调 make_plan_node 的场景（即 paper_factor 注册路径）。
独立调 make_plan_node(pipeline) 不会触发此 bug，因为调用时 pipeline.nodes 已填充。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.pipeline import PipelineConfig
from harness.plan_node import make_plan_node


def test_plan_node_lazy_binds_to_nodes_dict(monkeypatch):
    """
    模拟 paper_factor factory 时序：
    1. 构造 pipeline 时 nodes={}
    2. 立即 make_plan_node(pipeline) → node_names 应当[]
    3. 后续 populate nodes
    4. 第一次 plan_node 调用时 node_names 必须包含已 populate 的节点

    注：当前实现修复后，plan_node 内部每次读 pipeline.nodes，所以 lazy binding 正常。
    """
    # 让 LLM 失败，强制走 fallback
    monkeypatch.setattr("harness.llm_client.call_llm_json",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no LLM")))

    # 1. 构造空 nodes pipeline
    pipeline = PipelineConfig(
        name="lazy_test",
        max_iter=8,
        nodes={},  # 空！
        prompts={"orchestrator": "{state_json} {tool_sigs} {max_iter} {goal}"},
    )

    # 2. 在 nodes 还空时构造 plan_node（模拟 factory 时序）
    plan_node = make_plan_node(pipeline)

    # 3. populate nodes（模拟 factory 的 Phase 2）
    def _collect(s):
        return {"collected_paper_ids": ["p1"]}
    pipeline.nodes["collect"] = _collect

    # 4. 调用 plan_node：fallback 应决策 collect（has_papers=False → fallback 选 collect）
    state = {
        "goal": "test lazy bind",
        "iteration_count": 0,
        "visited_nodes": [],
        "collected_paper_ids": [],
        "extracted_factors": [],
        "backtest_results": [],
        "pending_decisions": [],
        "errors": [],
    }
    out = plan_node(state)
    # 关键断言：next_node 是 collect（fallback 选择），且通过了 node_names 校验
    assert out["plan"][-1]["node"] == "collect", f"got: {out['plan'][-1]['node']}, reason: {out['plan'][-1]['reason']}"


def test_plan_node_recognizes_node_added_after_construction(monkeypatch):
    """
    验证：plan_node 构造后添加新节点，运行时能被识别。
    """
    monkeypatch.setattr("harness.llm_client.call_llm_json",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no LLM")))

    pipeline = PipelineConfig(
        name="add_after",
        max_iter=8,
        nodes={
            "collect": lambda s: {},
            "respond": lambda s: {},
        },
        prompts={"orchestrator": "{state_json} {tool_sigs} {max_iter} {goal}"},
    )

    plan_node = make_plan_node(pipeline)

    # 加一个新节点
    pipeline.nodes["analyze"] = lambda s: {}

    # fallback 会先选 collect（因为无 papers），但是我们要测试 analyze 也被识别
    state = {
        "goal": "test add after",
        "iteration_count": 0,
        "visited_nodes": [],
        "collected_paper_ids": ["p1"],  # 有论文 → fallback 选 analyze
        "extracted_factors": [],
        "backtest_results": [],
        "pending_decisions": [],
        "errors": [],
    }
    out = plan_node(state)
    # fallback 选 analyze，且新加的 analyze 在 node_names 中 → 不被 force respond
    assert out["plan"][-1]["node"] == "analyze", f"got: {out['plan'][-1]['node']}"


def test_plan_node_rejects_unknown_node(monkeypatch):
    """
    反向：fallback 决策了一个不在 pipeline.nodes 的节点时，应 force respond。
    """
    monkeypatch.setattr("harness.llm_client.call_llm_json",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no LLM")))

    pipeline = PipelineConfig(
        name="reject_unknown",
        max_iter=8,
        nodes={"respond": lambda s: {}},  # 只有 respond
        prompts={"orchestrator": "{state_json} {tool_sigs} {max_iter} {goal}"},
    )

    plan_node = make_plan_node(pipeline)

    # 直接手工设置 plan_node 模拟 LLM 返回未知 next_node（无法模拟 LLM 输出，但 fallback 不会返未知节点）
    # 替代方案：patch _fallback_decision 让它返未知节点
    import harness.plan_node as pn
    monkeypatch.setattr(pn, "_fallback_decision",
                        lambda *args, **kwargs: {"next_node": "totally_made_up", "reason": "test"})

    state = {
        "goal": "test reject",
        "iteration_count": 0,
        "visited_nodes": [],
        "errors": [],
    }
    out = plan_node(state)
    # 未知 next_node 应被 force respond
    assert out["plan"][-1]["node"] == "respond"
    assert "forced respond" in out["plan"][-1]["reason"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
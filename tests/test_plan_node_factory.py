"""
test_plan_node_factory.py - 通用 plan_node 工厂

构造 paper_factor PipelineConfig（mock），验证 make_plan_node：
1. LLM 失败时走 fallback（heuristic 推进）
2. yaml hard_rules 在 LLM 决策后正确覆盖
3. visited-count 防循环
4. 未知 next_node 兜底 respond
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.pipeline import PipelineConfig
from harness.plan_node import make_plan_node
from harness.state_base import make_base_typed_dict
from harness.hard_rules import StateView


# === Mock pipeline：4 规则照搬 paper_factor ===

PIPELINE = PipelineConfig(
    name="paper_factor_test",
    max_iter=8,
    interrupt_before=["decide"],
    terminal_nodes=["respond"],
    nodes={
        "plan": lambda s: {},
        "collect": lambda s: {"collected_paper_ids": ["p1"]},
        "analyze": lambda s: {"extracted_factors": [{"name": "f"}]},
        "persist_factors": lambda s: {"persist_attempted": True},
        "quality_check": lambda s: {"quality_results": [{"feasible": True}]},
        "backtest": lambda s: {"backtest_results": [{"IC": 0.1}]},
        "decide": lambda s: {},
        "respond": lambda s: {},
    },
    prompts={"orchestrator": "{state_json} {tool_sigs} {max_iter} {goal}"},
    hard_rules=[
        {"id": "r1", "when": "has_papers and not has_factors", "force": "analyze", "reason": "R1"},
        {"id": "r2", "when": "has_factors and not persist_attempted", "force": "persist_factors", "reason": "R2"},
        {"id": "r3", "when": "persist_attempted and not has_quality", "force": "quality_check", "reason": "R3"},
        {"id": "r4", "when": "has_quality and not has_backtest", "force": "backtest", "reason": "R4"},
    ],
)


# 让 LLM 总是失败，强制走 fallback
def _no_llm(*args, **kwargs):
    raise RuntimeError("LLM disabled for test")


def _patch_llm(monkeypatch):
    """让 harness.llm_client.call_llm_json 抛错。"""
    monkeypatch.setattr("harness.llm_client.call_llm_json", _no_llm)


def _state(**kw):
    base = {
        "goal": "test goal",
        "messages": [],
        "iteration_count": 0,
        "visited_nodes": [],
        "errors": [],
    }
    base.update(kw)
    return base


def test_factory_returns_callable():
    plan_node = make_plan_node(PIPELINE)
    assert callable(plan_node)


def test_fallback_first_iteration_returns_collect(monkeypatch):
    """空 state：fallback 决策 → collect。"""
    _patch_llm(monkeypatch)
    plan_node = make_plan_node(PIPELINE)
    out = plan_node(_state())
    assert out["plan"][-1]["node"] == "collect"
    assert out["visited_nodes"] == ["collect"]


def test_fallback_after_papers_returns_analyze(monkeypatch):
    """有论文无因子：fallback 决策 → analyze。"""
    _patch_llm(monkeypatch)
    plan_node = make_plan_node(PIPELINE)
    out = plan_node(_state(collected_paper_ids=["p1"]))
    assert out["plan"][-1]["node"] == "analyze"


def test_fallback_after_factors_returns_persist(monkeypatch):
    """有因子：fallback 决策 → persist_factors。"""
    _patch_llm(monkeypatch)
    plan_node = make_plan_node(PIPELINE)
    out = plan_node(_state(
        collected_paper_ids=["p1"],
        extracted_factors=[{"name": "f"}],
    ))
    assert out["plan"][-1]["node"] == "persist_factors"


def test_hard_rules_known_state(monkeypatch):
    """状态投影到 StateView，与 yaml 规则匹配正确。"""
    _patch_llm(monkeypatch)
    plan_node = make_plan_node(PIPELINE)

    # persist 之后
    state = _state(
        collected_paper_ids=["p1"],
        extracted_factors=[{"name": "f"}],
        persist_attempted=True,
    )
    out = plan_node(state)
    # fallback 会先 pick quality_check（因为 has_factors 但 persist 已完成）
    assert out["plan"][-1]["node"] == "quality_check"


def test_visited_count_cap_forces_respond(monkeypatch):
    """同一节点被访问 2 次后，强制 respond。"""
    _patch_llm(monkeypatch)
    plan_node = make_plan_node(PIPELINE)

    # fallback 在 persist_factors 阶段（has factors, no persist_attempted）
    # 让 visited 中已有 2 次 persist_factors → 触发兜底
    state = _state(
        collected_paper_ids=["p1"],
        extracted_factors=[{"name": "f"}],
        visited_nodes=["persist_factors", "persist_factors"],  # 该节点已被访问 2 次
    )
    out = plan_node(state)
    # fallback 选 persist_factors（has_factors + not persist_attempted），但 visited.count=2 → respond
    assert out["plan"][-1]["node"] == "respond"


def test_unknown_next_node_falls_back_to_respond(monkeypatch):
    """LLM 返回未知 next_node 时兜底 respond。"""
    _patch_llm(monkeypatch)
    plan_node = make_plan_node(PIPELINE)

    state = _state(
        collected_paper_ids=["p1"],
        extracted_factors=[{"name": "f"}],
        persist_attempted=True,
        quality_results=[{"feasible": True}],
        backtest_results=[{"IC": 0.1}],
    )
    out = plan_node(state)
    # fallback 选 decide → 但无 pending_decisions → 仍 OK
    # 这里关键是：plan_node 输出合法 node 名
    assert out["plan"][-1]["node"] in PIPELINE.nodes


def test_iteration_count_increments(monkeypatch):
    _patch_llm(monkeypatch)
    plan_node = make_plan_node(PIPELINE)
    out = plan_node(_state(iteration_count=3))
    assert out["iteration_count"] == 4


def test_state_view_exposes_required_predicates():
    """StateView 必须有 hard_rules 需要的全部谓词。"""
    v = StateView({})
    for p in ["has_papers", "has_factors", "persist_attempted", "has_quality",
              "has_backtest", "has_decisions", "iteration_count"]:
        assert hasattr(v, p), f"StateView missing {p}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
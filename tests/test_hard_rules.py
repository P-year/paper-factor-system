"""
test_hard_rules.py - 验证 DSL 决策与原 plan_node 的 4 条硬规则 if 块等价

目的：这是 Phase 1 重构的关键回归网。每条规则构造多个 fixture state，断言：
1. 原 if 块决策的 next_node / reason
2. 新 DSL apply_rules 决策的 next_node / reason
3. 二者一致
"""
import sys
from pathlib import Path

# 让 pytest 能从项目根 import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from harness.hard_rules import (
    StateView,
    build_rule,
    apply_rules,
    rule_matches,
)


# === paper_factor 的 4 条规则（yaml 草稿照搬） ===

RULES_SPEC = [
    {
        "id": "collect_must_analyze",
        "when": "has_papers and not has_factors",
        "force": "analyze",
        "reason": "硬规则：有论文必须先 analyze",
    },
    {
        "id": "analyze_must_persist",
        "when": "has_factors and not persist_attempted",
        "force": "persist_factors",
        "reason": "硬规则：analyze 后必须 persist_factors",
    },
    {
        "id": "persist_must_quality",
        "when": "persist_attempted and not has_quality",
        "force": "quality_check",
        "reason": "硬规则：persist 之后必须 quality_check",
    },
    {
        "id": "quality_must_backtest",
        "when": "has_quality and not has_backtest",
        "force": "backtest",
        "reason": "硬规则：quality_check 之后必须 backtest",
    },
]

RULES = [build_rule(s) for s in RULES_SPEC]


# === 原 plan_node 的 4 条 if 块（手工复刻，不 import plan_node 避免循环依赖） ===

def _old_apply_hard_rules(state, decision):
    """复制原 plan_node.py:87-120 的 4 条 if 块。"""
    decision = dict(decision)

    if (
        state.get("collected_paper_ids")
        and not state.get("extracted_factors")
        and decision.get("next_node") not in ("analyze", "respond")
    ):
        decision["next_node"] = "analyze"
        decision["reason"] = "硬规则：有论文必须先 analyze"

    if (
        state.get("extracted_factors")
        and not state.get("persist_attempted")
        and decision.get("next_node") not in ("persist_factors", "respond")
    ):
        decision["next_node"] = "persist_factors"
        decision["reason"] = "硬规则：analyze 后必须 persist_factors"

    if (
        state.get("persist_attempted")
        and not state.get("quality_results")
        and decision.get("next_node") not in ("quality_check", "respond")
    ):
        decision["next_node"] = "quality_check"
        decision["reason"] = "硬规则：persist 之后必须 quality_check"

    if (
        state.get("quality_results")
        and not state.get("backtest_results")
        and decision.get("next_node") not in ("backtest", "respond")
    ):
        decision["next_node"] = "backtest"
        decision["reason"] = "硬规则：quality_check 之后必须 backtest"

    return decision


# === Fixture states ===

def _state(**kw):
    """构造一个测试 state，未指定字段默认为空。"""
    base = {
        "collected_paper_ids": [],
        "extracted_factors": [],
        "persist_attempted": False,
        "quality_results": [],
        "backtest_results": [],
        "pending_decisions": [],
        "iteration_count": 0,
    }
    base.update(kw)
    return base


# === 关键 fixtures：覆盖每个规则触发 / 不触发 + LLM 选择各个节点 ===

PAPERS = ["arxiv:1", "arxiv:2"]
FACTORS = [{"factor_name": "x"}]
QUALITY = [{"factor": "x", "feasible": True}]
BACKTEST = [{"factor_name": "x", "IC": 0.1}]


# === Tests ===

def test_state_view_projection():
    """StateView 各谓词从 state 派生正确。"""
    v = StateView({
        "collected_paper_ids": PAPERS,
        "extracted_factors": FACTORS,
        "persist_attempted": True,
        "quality_results": QUALITY,
        "backtest_results": BACKTEST,
        "pending_decisions": [],
        "iteration_count": 3,
    })
    assert v.has_papers is True
    assert v.has_factors is True
    assert v.persist_attempted is True
    assert v.has_quality is True
    assert v.has_backtest is True
    assert v.has_decisions is False
    assert v.iteration_count == 3


def test_rule_1_fires_when_papers_no_factors():
    """Rule 1: 有论文无因子 → 强制 analyze（无论 LLM 选什么非终端节点）。"""
    state = _state(collected_paper_ids=PAPERS)
    llm_decision = {"next_node": "collect", "reason": "llm 觉得再 collect 一次"}
    new = apply_rules(RULES, state, llm_decision)
    assert new["next_node"] == "analyze"
    assert new["reason"] == "硬规则：有论文必须先 analyze"
    assert new["forced_by_rule"] == "collect_must_analyze"


def test_rule_1_does_not_fire_when_factors_present():
    """Rule 1 不触发：有因子时（且 persist 已完成，隔离 Rule 2）。"""
    state = _state(
        collected_paper_ids=PAPERS,
        extracted_factors=FACTORS,
        persist_attempted=True,
        quality_results=QUALITY,
        backtest_results=BACKTEST,
    )
    llm_decision = {"next_node": "respond", "reason": "llm"}
    new = apply_rules(RULES, state, llm_decision)
    assert new["next_node"] == "respond"
    assert "forced_by_rule" not in new


def test_rule_1_respects_respond_choice():
    """Rule 1 不覆盖 LLM 选择 respond（与原 if 块一致）。"""
    state = _state(collected_paper_ids=PAPERS)
    llm_decision = {"next_node": "respond", "reason": "done"}
    new = apply_rules(RULES, state, llm_decision)
    assert new["next_node"] == "respond"


def test_rule_2_fires_after_analyze():
    state = _state(collected_paper_ids=PAPERS, extracted_factors=FACTORS)
    llm_decision = {"next_node": "quality_check", "reason": "llm 跳过 persist"}
    new = apply_rules(RULES, state, llm_decision)
    assert new["next_node"] == "persist_factors"
    assert new["reason"] == "硬规则：analyze 后必须 persist_factors"


def test_rule_2_respects_already_persisted():
    state = _state(extracted_factors=FACTORS, persist_attempted=True)
    llm_decision = {"next_node": "respond", "reason": "done"}
    new = apply_rules(RULES, state, llm_decision)
    assert new["next_node"] == "respond"


def test_rule_3_fires_after_persist():
    state = _state(persist_attempted=True)
    llm_decision = {"next_node": "backtest", "reason": "skip quality"}
    new = apply_rules(RULES, state, llm_decision)
    assert new["next_node"] == "quality_check"
    assert new["reason"] == "硬规则：persist 之后必须 quality_check"


def test_rule_3_respects_already_quality_done():
    state = _state(persist_attempted=True, quality_results=QUALITY)
    llm_decision = {"next_node": "respond", "reason": "done"}
    new = apply_rules(RULES, state, llm_decision)
    assert new["next_node"] == "respond"


def test_rule_4_fires_after_quality():
    state = _state(quality_results=QUALITY)
    llm_decision = {"next_node": "decide", "reason": "skip backtest"}
    new = apply_rules(RULES, state, llm_decision)
    assert new["next_node"] == "backtest"
    assert new["reason"] == "硬规则：quality_check 之后必须 backtest"


def test_rule_4_respects_already_backtest_done():
    state = _state(quality_results=QUALITY, backtest_results=BACKTEST)
    llm_decision = {"next_node": "respond", "reason": "done"}
    new = apply_rules(RULES, state, llm_decision)
    assert new["next_node"] == "respond"


def test_rules_chain_through_full_pipeline():
    """4 条规则按顺序触发，模拟完整流程。"""
    # 阶段 1: 有论文无因子
    s = _state(collected_paper_ids=PAPERS)
    d = apply_rules(RULES, s, {"next_node": "respond", "reason": "done"})
    assert d["next_node"] == "respond"  # respond 兜底

    # 阶段 2: 有论文有因子
    s = _state(collected_paper_ids=PAPERS, extracted_factors=FACTORS)
    d = apply_rules(RULES, s, {"next_node": "backtest", "reason": "skip persist+quality"})
    assert d["next_node"] == "persist_factors"

    # 阶段 3: persist 完
    s = _state(persist_attempted=True, extracted_factors=FACTORS)
    d = apply_rules(RULES, s, {"next_node": "backtest", "reason": "skip quality"})
    assert d["next_node"] == "quality_check"

    # 阶段 4: quality 完
    s = _state(quality_results=QUALITY, persist_attempted=True)
    d = apply_rules(RULES, s, {"next_node": "decide", "reason": "skip backtest"})
    assert d["next_node"] == "backtest"

    # 阶段 5: backtest 完 → 不触发任何规则
    s = _state(backtest_results=BACKTEST, quality_results=QUALITY, persist_attempted=True)
    d = apply_rules(RULES, s, {"next_node": "respond", "reason": "done"})
    assert d["next_node"] == "respond"


# === 等价性测试：DSL vs 原 if 块 ===

EQUIVALENCE_CASES = [
    # (state, llm_decision, description)
    (_state(), {"next_node": "collect", "reason": "llm"}, "empty state"),
    (_state(collected_paper_ids=PAPERS), {"next_node": "respond", "reason": "done"}, "papers + llm=respond"),
    (_state(collected_paper_ids=PAPERS), {"next_node": "backtest", "reason": "llm skip"}, "papers + llm=backtest"),
    (_state(extracted_factors=FACTORS), {"next_node": "quality_check", "reason": "skip persist"}, "factors only + llm=quality"),
    (_state(persist_attempted=True), {"next_node": "backtest", "reason": "skip quality"}, "persisted + llm=backtest"),
    (_state(quality_results=QUALITY), {"next_node": "decide", "reason": "skip backtest"}, "quality + llm=decide"),
    (_state(backtest_results=BACKTEST), {"next_node": "respond", "reason": "done"}, "all done"),
    (_state(collected_paper_ids=PAPERS, extracted_factors=FACTORS, persist_attempted=True, quality_results=QUALITY, backtest_results=BACKTEST), {"next_node": "respond", "reason": "done"}, "all done"),
]


def test_dsl_equivalent_to_old_if_blocks():
    """核心回归网：DSL 和原 if 块在所有 fixture 上决策一致。"""
    for state, llm_decision, desc in EQUIVALENCE_CASES:
        old_d = _old_apply_hard_rules(state, llm_decision)
        new_d = apply_rules(RULES, state, llm_decision)
        assert old_d.get("next_node") == new_d.get("next_node"), (
            f"[{desc}] next_node diverge: old={old_d.get('next_node')!r} new={new_d.get('next_node')!r}"
        )
        assert old_d.get("reason") == new_d.get("reason"), (
            f"[{desc}] reason diverge: old={old_d.get('reason')!r} new={new_d.get('reason')!r}"
        )


# === DSL 解析器边缘 case ===

def test_dsl_parses_complex_expression():
    """复杂表达式解析正确。"""
    rule = build_rule({
        "id": "test_complex",
        "when": "has_papers and not has_factors or iteration_count >= 5",
        "force": "respond",
        "reason": "complex",
    })
    # 有论文无因子 → 第一个 and 为真 → 整体真
    assert rule_matches(rule, _state(collected_paper_ids=PAPERS)) is True
    # 无论文，iteration=10 → 第二个 or 为真 → 整体真
    assert rule_matches(rule, _state(iteration_count=10)) is True
    # 无论文，iteration=3 → 整体 false
    assert rule_matches(rule, _state(iteration_count=3)) is False


def test_dsl_parses_parentheses():
    rule = build_rule({
        "id": "test_paren",
        "when": "(has_papers or has_factors) and not has_backtest",
        "force": "x",
        "reason": "y",
    })
    assert rule_matches(rule, _state(collected_paper_ids=PAPERS)) is True
    assert rule_matches(rule, _state(extracted_factors=FACTORS)) is True
    assert rule_matches(rule, _state()) is False


def test_dsl_rejects_unknown_predicate():
    """未知谓词应抛 NameError（防御性）。"""
    rule = build_rule({
        "id": "bad",
        "when": "unknown_predicate",
        "force": "x",
        "reason": "y",
    })
    try:
        rule_matches(rule, _state())
        assert False, "应该抛 NameError"
    except NameError:
        pass


def test_dsl_rejects_bad_syntax():
    """语法错误应抛 SyntaxError。"""
    try:
        build_rule({"id": "bad", "when": "has_papers and", "force": "x", "reason": "y"})
        assert False, "应该抛 SyntaxError"
    except SyntaxError:
        pass


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
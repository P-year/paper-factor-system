"""
test_lineage.py - lineage 追踪单元测试

覆盖：
1. tag_factor_lineage / tag_quality_lineage / tag_backtest_lineage 写入 _lineage
2. factor_lineage_summary 摘要
3. full_lineage_chain 跨节点汇总
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.lineage import (
    tag_factor_lineage,
    tag_quality_lineage,
    tag_backtest_lineage,
    factor_lineage_summary,
    full_lineage_chain,
)


def _state(**kw):
    base = {
        "iteration_count": 5,
        "pipeline_name": "paper_factor",
        "session_id": "test_session",
        "extracted_factors": [],
        "quality_results": [],
        "backtest_results": [],
    }
    base.update(kw)
    return base


# === tag_factor_lineage ===

def test_tag_factor_lineage_basic():
    f = {"factor_name": "momentum_1m"}
    source = {"arxiv_id": "2502.12345", "title": "Momentum Paper"}
    tag_factor_lineage(f, _state(iteration_count=3), source_paper=source)

    assert "_lineage" in f
    l = f["_lineage"]
    assert l["source_paper_id"] == "2502.12345"
    assert l["source_paper_title"] == "Momentum Paper"
    assert l["extract_iteration"] == 3
    assert "extract_timestamp" in l
    assert l["pipeline_name"] == "paper_factor"
    assert l["session_id"] == "test_session"


def test_tag_factor_lineage_fallback_to_factor_fields():
    """没传 source_paper 时，从 factor 字段取。"""
    f = {"factor_name": "x", "paper_link": "https://x", "paper_title": "T"}
    tag_factor_lineage(f, _state())
    l = f["_lineage"]
    assert l["source_paper_id"] == "https://x"
    assert l["source_paper_title"] == "T"


def test_tag_factor_lineage_missing_source():
    """什么信息都没有也不会崩。"""
    f = {"factor_name": "x"}
    tag_factor_lineage(f, _state())
    l = f["_lineage"]
    assert l["source_paper_id"] == ""
    assert l["source_paper_title"] == ""


# === tag_quality_lineage ===

def test_tag_quality_lineage_heuristic():
    qr = {"factor_name": "f", "method": "heuristic"}
    tag_quality_lineage(qr, _state(iteration_count=7))
    assert qr["_lineage"]["quality_method"] == "heuristic"
    assert qr["_lineage"]["quality_iteration"] == 7
    assert "quality_timestamp" in qr["_lineage"]


def test_tag_quality_lineage_llm():
    qr = {"factor_name": "f", "method": "llm"}
    tag_quality_lineage(qr, _state(iteration_count=8))
    assert qr["_lineage"]["quality_method"] == "llm"
    assert qr["_lineage"]["quality_iteration"] == 8


# === tag_backtest_lineage ===

def test_tag_backtest_lineage_success():
    bt = {"factor_name": "f", "IC": 0.03}
    tag_backtest_lineage(bt, _state(iteration_count=10))
    l = bt["_lineage"]
    assert l["backtest_iteration"] == 10
    assert l["backtest_status"] == "success"


def test_tag_backtest_lineage_failed():
    bt = {"factor_name": "f", "IC": None}
    tag_backtest_lineage(bt, _state(iteration_count=10))
    assert bt["_lineage"]["backtest_status"] == "failed"


# === factor_lineage_summary ===

def test_factor_lineage_summary():
    f = {"factor_name": "momentum_1m"}
    tag_factor_lineage(f, _state(iteration_count=5), source_paper={"arxiv_id": "x", "title": "T"})
    s = factor_lineage_summary(f)
    assert s["source_paper_id"] == "x"
    assert s["source_paper_title"] == "T"
    assert s["extract_iteration"] == 5


# === full_lineage_chain ===

def test_full_lineage_chain_complete():
    state = _state(
        extracted_factors=[
            {"factor_name": "m1", "_lineage": {"source_paper_id": "x1", "extract_iteration": 1}},
        ],
        quality_results=[
            {"factor_name": "m1", "_lineage": {"quality_method": "heuristic", "quality_iteration": 3}},
        ],
        backtest_results=[
            {"original_factor_name": "m1", "_lineage": {"backtest_iteration": 5, "backtest_status": "success"}},
        ],
    )
    chain = full_lineage_chain(state["extracted_factors"][0], state)
    assert chain["factor_name"] == "m1"
    assert chain["factor_lineage"]["source_paper_id"] == "x1"
    assert chain["quality"]["quality_method"] == "heuristic"
    assert chain["backtest"]["backtest_status"] == "success"


def test_full_lineage_chain_partial():
    """只有 factor 没有 quality / backtest 的情况。"""
    state = _state(
        extracted_factors=[{"factor_name": "x", "_lineage": {"source_paper_id": "p"}}],
    )
    chain = full_lineage_chain(state["extracted_factors"][0], state)
    assert chain["factor_name"] == "x"
    assert chain["quality"] is None
    assert chain["backtest"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
"""
test_analyze_with_rag.py - analyze 节点 + RAG context 集成测试

覆盖：
1. analyze_node 接收 retrieval_context 后传给 extract_factor_tool
2. factor 被打上 _used_rag_context / _rag_top_k
3. retrieval_context=None 时不注入（_used_rag_context 不存在）
4. core.analyzer.extract_factor(paper, similar_papers) prompt 头部包含 RAG 块
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest


@pytest.fixture
def mock_state():
    return {
        "collected_paper_ids": ["p1"],
        "extracted_factors": [],
        "errors": [],
        "verify_log": [],
        "plan": [],
        "iteration_count": 0,
        "visited_nodes": [],
    }


def test_extract_factor_signature_accepts_similar_papers():
    """extract_factor 签名支持 similar_papers kwarg。"""
    from core.analyzer import FactorAnalyzer
    import inspect
    sig = inspect.signature(FactorAnalyzer.extract_factor)
    assert "similar_papers" in sig.parameters
    assert sig.parameters["similar_papers"].default is None


def test_extract_factor_tool_signature_accepts_similar_papers():
    """extract_factor_tool 签名支持 similar_papers kwarg。"""
    from agent.tools.analysis_tools import extract_factor_tool
    import inspect
    sig = inspect.signature(extract_factor_tool)
    assert "similar_papers" in sig.parameters


def test_verified_extracted_factor_signature_accepts_similar_papers():
    from harness.tool_call import verified_extracted_factor
    import inspect
    sig = inspect.signature(verified_extracted_factor)
    assert "similar_papers" in sig.parameters


def test_format_similar_papers():
    """_format_similar_papers 输出格式。"""
    from core.analyzer import FactorAnalyzer
    chunks = [
        {
            "paper_title": "Momentum",
            "arxiv_id": "p1",
            "score": 0.83,
            "text": "this is a momentum paper about Chinese A shares.",
        },
        {
            "paper_title": "Value",
            "arxiv_id": "p2",
            "score": 0.71,
            "text": "value investing research.",
        },
    ]
    out = FactorAnalyzer._format_similar_papers(chunks)
    assert "[1]" in out
    assert "[2]" in out
    assert "Momentum" in out
    assert "score=0.830" in out


def test_format_similar_papers_empty():
    from core.analyzer import FactorAnalyzer
    assert FactorAnalyzer._format_similar_papers([]) == ""


def test_analyze_node_passes_similar_papers(monkeypatch, mock_state):
    """analyze_node 把 state.retrieval_context 传给 extract_factor_tool。"""
    from pipelines.paper_factor.nodes import analyze as mod

    captured = {}

    def fake_load_papers(ids):
        return [{
            "arxiv_id": "p1",
            "title": "Test Paper",
            "summary": "摘要内容。" * 20,
            "link": "https://arxiv.org/abs/p1",
        }]

    def fake_verified(paper, *, similar_papers=None, max_retries=0):
        captured["paper"] = paper
        captured["similar_papers"] = similar_papers
        return {"is_factor": False, "factor": None, "error": None}

    monkeypatch.setattr(mod, "_load_papers", fake_load_papers)
    monkeypatch.setattr(mod, "verified_extracted_factor", fake_verified)

    # 有 retrieval_context
    mock_state["retrieval_context"] = [
        {"paper_title": "Ref1", "arxiv_id": "r1", "score": 0.8, "text": "ref text"}
    ]
    result = mod.analyze_node(mock_state)
    assert captured["similar_papers"] is not None
    assert len(captured["similar_papers"]) == 1


def test_analyze_node_no_retrieval_passes_none(monkeypatch, mock_state):
    from pipelines.paper_factor.nodes import analyze as mod

    captured = {}

    def fake_load_papers(ids):
        return [{
            "arxiv_id": "p1", "title": "T", "summary": "S", "link": "L",
        }]

    def fake_verified(paper, *, similar_papers=None, max_retries=0):
        captured["similar_papers"] = similar_papers
        return {"is_factor": False, "factor": None, "error": None}

    monkeypatch.setattr(mod, "_load_papers", fake_load_papers)
    monkeypatch.setattr(mod, "verified_extracted_factor", fake_verified)
    # mock_state 没 retrieval_context
    mod.analyze_node(mock_state)
    assert captured["similar_papers"] is None


def test_factor_tagged_with_rag_metadata(monkeypatch, mock_state):
    """factor 被打了 _used_rag_context + _rag_top_k。"""
    from pipelines.paper_factor.nodes import analyze as mod

    def fake_load_papers(ids):
        return [{
            "arxiv_id": "p1", "title": "T", "summary": "S", "link": "L",
        }]

    def fake_verified(paper, *, similar_papers=None, max_retries=0):
        return {
            "is_factor": True,
            "factor": {"factor_name": "test_factor", "definition": "x"},
            "error": None,
        }

    monkeypatch.setattr(mod, "_load_papers", fake_load_papers)
    monkeypatch.setattr(mod, "verified_extracted_factor", fake_verified)

    mock_state["retrieval_context"] = [{"paper_title": "R", "text": "x", "score": 0.5}]
    result = mod.analyze_node(mock_state)
    factor = result["extracted_factors"][0]
    assert factor.get("_used_rag_context") is True
    assert factor.get("_rag_top_k") == 1


def test_factor_no_rag_no_metadata(monkeypatch, mock_state):
    from pipelines.paper_factor.nodes import analyze as mod

    def fake_load_papers(ids):
        return [{"arxiv_id": "p1", "title": "T", "summary": "S", "link": "L"}]

    def fake_verified(paper, *, similar_papers=None, max_retries=0):
        return {
            "is_factor": True,
            "factor": {"factor_name": "f", "definition": "d"},
            "error": None,
        }

    monkeypatch.setattr(mod, "_load_papers", fake_load_papers)
    monkeypatch.setattr(mod, "verified_extracted_factor", fake_verified)

    result = mod.analyze_node(mock_state)
    factor = result["extracted_factors"][0]
    assert "_used_rag_context" not in factor


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
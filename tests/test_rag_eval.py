"""
test_rag_eval.py - Ground-truth 评测
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.rag.eval import (
    recall_at_k,
    mrr,
    ndcg_at_k,
    run_evaluation,
    format_report,
    write_baseline,
    load_baseline,
    _sources_of_chunk_id,
    evaluate_query,
)


QUERIES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "rag_eval" / "queries.jsonl"


# === 单元测试指标函数 ===

def test_recall_at_k_basic():
    # retrieved: [p1:c1, p2:c1, p3:c1]; relevant=[p1, p2]
    assert recall_at_k(["p1:c1", "p2:c1", "p3:c1"], ["p1", "p2"], k=3) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k(["p1:c1", "p3:c1"], ["p1", "p2"], k=3) == 0.5


def test_recall_at_k_zero():
    assert recall_at_k(["p3:c1"], ["p1"], k=3) == 0.0


def test_recall_at_k_empty_relevant():
    assert recall_at_k(["p1"], [], k=3) == 0.0


def test_sources_of_chunk_id():
    assert _sources_of_chunk_id("e38ddbd567dcdf3e:0") == "e38ddbd567dcdf3e"
    assert _sources_of_chunk_id("e38ddbd567dcdf3e") == "e38ddbd567dcdf3e"


def test_mrr_first_hit_rank_1():
    assert mrr(["p1:c1", "p2:c1"], ["p1"]) == 1.0


def test_mrr_first_hit_rank_2():
    assert mrr(["p3:c1", "p1:c1", "p2:c1"], ["p1"]) == 0.5


def test_mrr_no_hit():
    assert mrr(["p3:c1"], ["p1"]) == 0.0


def test_ndcg_perfect():
    """top-1 命中 → NDCG=1.0"""
    assert ndcg_at_k(["p1:c1", "p2:c1"], ["p1"], k=2) == 1.0


def test_ndcg_zero():
    assert ndcg_at_k(["p3:c1"], ["p1"], k=3) == 0.0


def test_ndcg_position_matters():
    """top-1 命中 > top-3 命中。"""
    s1 = ndcg_at_k(["p1:c1", "p3:c1"], ["p1"], k=2)
    s2 = ndcg_at_k(["p3:c1", "p3:c2", "p1:c1"], ["p1"], k=3)
    assert s1 > s2


# === load / save baseline ===

def test_baseline_roundtrip(tmp_path):
    report = {
        "queries_count": 30,
        "k": 5,
        "recall_at_k": 0.5,
        "mrr": 0.6,
        "ndcg_at_k": 0.55,
    }
    path = tmp_path / "baseline.json"
    write_baseline(report, path)
    loaded = load_baseline(path)
    assert loaded == report


def test_baseline_strips_per_query(tmp_path):
    report = {
        "queries_count": 30,
        "k": 5,
        "recall_at_k": 0.5,
        "mrr": 0.6,
        "ndcg_at_k": 0.55,
        "per_query": [{"query": "x", "recall_at_k": 1.0}],
    }
    path = tmp_path / "baseline.json"
    write_baseline(report, path)
    loaded = load_baseline(path)
    # per_query 不写
    assert "per_query" not in loaded


# === Ground-truth 端到端 ===

@pytest.fixture(scope="module")
def pdf_dir():
    p = PROJECT_ROOT / "tests" / "fixtures" / "pdfs"
    if not p.exists() or not list(p.glob("*.pdf")):
        pytest.skip("fixture PDFs not available")
    return p


@pytest.fixture
def store_with_pdfs(tmp_path, pdf_dir):
    from harness.rag.store import PaperRAGStore
    from harness.rag.embedder import HashBackend
    from harness.rag.pdf_loader import pdf_dir_to_papers

    s = PaperRAGStore(
        rag_dir=tmp_path / "rag",
        embedder=HashBackend(),
    )
    papers = pdf_dir_to_papers(pdf_dir, max_pages=3)
    s.index_papers(papers)
    return s


def test_run_evaluation_returns_metrics(store_with_pdfs):
    """Ground-truth eval 跑通。"""
    from harness.rag.retriever import PaperRetriever
    r = PaperRetriever(store_with_pdfs, fusion_strategy="rrf")
    report = run_evaluation(r, QUERIES_PATH, k=5)
    assert "queries_count" in report
    assert "recall_at_k" in report
    assert "mrr" in report
    assert "ndcg_at_k" in report
    assert report["queries_count"] == 100


def test_run_evaluation_at_least_some_hits(store_with_pdfs):
    """ground-truth 至少召回一些相关论文（不要求完美，但 > 0）。"""
    from harness.rag.retriever import PaperRetriever
    r = PaperRetriever(store_with_pdfs, fusion_strategy="rrf")
    report = run_evaluation(r, QUERIES_PATH, k=5)
    assert report["recall_at_k"] > 0.0, "Recall@5 应该 > 0，至少 hash+BM25 能命中一些"


def test_format_report(store_with_pdfs):
    from harness.rag.retriever import PaperRetriever
    r = PaperRetriever(store_with_pdfs, fusion_strategy="rrf")
    report = run_evaluation(r, QUERIES_PATH, k=5)
    text = format_report(report)
    assert "Recall@5" in text
    assert "MRR" in text
    assert "NDCG@5" in text


def test_per_query_breakdown(store_with_pdfs):
    """per_query 列表应包含所有 query。"""
    from harness.rag.retriever import PaperRetriever
    r = PaperRetriever(store_with_pdfs, fusion_strategy="rrf")
    report = run_evaluation(r, QUERIES_PATH, k=5)
    assert len(report["per_query"]) == 100
    for q in report["per_query"]:
        assert "query" in q
        assert "recall_at_k" in q


def test_rrf_meets_or_exceeds_weighted_baseline(store_with_pdfs):
    """RRF 不应比 weighted 更差。"""
    from harness.rag.retriever import PaperRetriever
    r_rrf = PaperRetriever(store_with_pdfs, fusion_strategy="rrf")
    r_w = PaperRetriever(store_with_pdfs, fusion_strategy="weighted")
    rep_rrf = run_evaluation(r_rrf, QUERIES_PATH, k=5)
    rep_w = run_evaluation(r_w, QUERIES_PATH, k=5)
    # RRF 在 Recall 上 ≥ weighted（容差 5%）
    assert rep_rrf["recall_at_k"] >= rep_w["recall_at_k"] - 0.05


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
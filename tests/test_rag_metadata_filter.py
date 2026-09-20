"""
test_rag_metadata_filter.py - metadata filter 测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.rag.store import PaperRAGStore
from harness.rag.embedder import HashBackend
from harness.rag.retriever import PaperRetriever
from harness.rag.chunker import Chunker


@pytest.fixture
def store(tmp_path):
    s = PaperRAGStore(
        rag_dir=tmp_path / "rag",
        embedder=HashBackend(),
        chunker=Chunker(max_chunk_chars=500, overlap_chars=100),
    )
    papers = [
        {"arxiv_id": "p1", "title": "Momentum A-shares",
         "summary": "动量因子表现。" * 30, "_meta": {"topic": "factor_model", "year": "2024"}},
        {"arxiv_id": "p2", "title": "Value investing",
         "summary": "价值投资策略。" * 30, "_meta": {"topic": "factor_model", "year": "2023"}},
        {"arxiv_id": "p3", "title": "Reversal",
         "summary": "反转因子短期表现。" * 30, "_meta": {"topic": "trading", "year": "2024"}},
    ]
    s.index_papers(papers)
    return s


# === metadata filter ===

def test_filter_by_source_id_single(store):
    """filter={"source_id": "p1"} → 只返回 p1 的 chunks。"""
    r = PaperRetriever(store, fusion_strategy="rrf")
    chunks = r.retrieve("因子", top_k=5, filter={"source_id": "p1"})
    assert len(chunks) >= 1
    assert all(c["source_id"] == "p1" for c in chunks)


def test_filter_by_source_id_list(store):
    """filter={"source_id": ["p1", "p2"]} → 只 p1/p2。"""
    r = PaperRetriever(store, fusion_strategy="rrf")
    chunks = r.retrieve("因子", top_k=10, filter={"source_id": ["p1", "p2"]})
    assert len(chunks) >= 1
    assert all(c["source_id"] in ("p1", "p2") for c in chunks)
    assert not any(c["source_id"] == "p3" for c in chunks)


def test_filter_by_meta_topic(store):
    """filter={"topic": "factor_model"} → topic 字段匹配。"""
    r = PaperRetriever(store, fusion_strategy="rrf")
    chunks = r.retrieve("因子", top_k=10, filter={"topic": "factor_model"})
    # 应该返回 p1 + p2（topic=因子），p3 应被排除
    source_ids = {c["source_id"] for c in chunks}
    assert source_ids.issubset({"p1", "p2"})
    assert "p3" not in source_ids


def test_filter_by_meta_year(store):
    """filter={"year": "2024"} → year 字段匹配。"""
    r = PaperRetriever(store, fusion_strategy="rrf")
    chunks = r.retrieve("因子", top_k=10, filter={"year": "2024"})
    source_ids = {c["source_id"] for c in chunks}
    assert source_ids.issubset({"p1", "p3"})


def test_filter_combined(store):
    """多字段 filter。"""
    r = PaperRetriever(store, fusion_strategy="rrf")
    chunks = r.retrieve("因子", top_k=10, filter={"topic": "factor_model", "year": "2024"})
    source_ids = {c["source_id"] for c in chunks}
    assert source_ids.issubset({"p1"})  # 只有 p1 同时满足


def test_filter_no_match_returns_empty(store):
    """filter 无匹配 → 返回空列表。"""
    r = PaperRetriever(store, fusion_strategy="rrf")
    chunks = r.retrieve("因子", top_k=10, filter={"topic": "nonexistent"})
    assert chunks == []


def test_filter_with_rerank(store):
    """filter 与 rerank 协同。"""
    class MockReranker:
        @property
        def model_name(self):
            return "mock"
        def rerank(self, query, chunks, *, top_k=None):
            return list(reversed(chunks))[:top_k]

    r = PaperRetriever(store, fusion_strategy="rrf", reranker=MockReranker())
    chunks = r.retrieve("因子", top_k=5, filter={"source_id": ["p1", "p2"]})
    assert all(c["source_id"] in ("p1", "p2") for c in chunks)


def test_no_filter_returns_all_topics(store):
    """无 filter 时返回所有 source。"""
    r = PaperRetriever(store, fusion_strategy="rrf")
    chunks = r.retrieve("因子", top_k=10)
    source_ids = {c["source_id"] for c in chunks}
    # 三个 paper 都可能命中
    assert source_ids.issubset({"p1", "p2", "p3"})


def test_filter_top_k_limit(store):
    """filter 后 top_k 仍生效。"""
    r = PaperRetriever(store, fusion_strategy="rrf")
    chunks = r.retrieve("因子", top_k=1, filter={"source_id": ["p1", "p2", "p3"]})
    assert len(chunks) <= 1


# === 真实 PDF ===

def test_filter_with_real_pdfs(tmp_path):
    """真实 10 篇 PDF 过滤测试。"""
    from harness.rag.pdf_loader import pdf_dir_to_papers
    pdf_dir = PROJECT_ROOT / "tests" / "fixtures" / "pdfs"
    if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
        pytest.skip("fixture PDFs not available")

    store = PaperRAGStore(rag_dir=tmp_path / "rag", embedder=HashBackend())
    papers = pdf_dir_to_papers(pdf_dir, max_pages=3)
    store.index_papers(papers)

    r = PaperRetriever(store, fusion_strategy="rrf")
    # 无 filter：返回多个 source
    chunks_all = r.retrieve("factor", top_k=10)
    source_ids_all = {c["source_id"] for c in chunks_all}

    # filter 限定特定 source
    if source_ids_all:
        first_sid = next(iter(source_ids_all))
        chunks_filt = r.retrieve("factor", top_k=10, filter={"source_id": first_sid})
        assert all(c["source_id"] == first_sid for c in chunks_filt)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
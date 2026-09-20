"""
test_rag_fusion.py - RRF fusion + PaperRetriever fusion_strategy 测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.rag.fusion import rrf_fuse, _to_rank


# === _to_rank ===

def test_to_rank_basic():
    r = _to_rank({"c1": 0.9, "c2": 0.5, "c3": 0.1})
    assert r == {"c1": 1, "c2": 2, "c3": 3}


def test_to_rank_empty():
    assert _to_rank({}) == {}


def test_to_rank_ties():
    """同分时按 dict 顺序（sorted stable）。"""
    r = _to_rank({"a": 0.5, "b": 0.5})
    # 两个都是 1 和 2（具体看 sorted 稳定性）
    assert set(r.keys()) == {"a", "b"}
    assert set(r.values()) == {1, 2}


# === rrf_fuse ===

def test_rrf_basic():
    """两路召回 → RRF 融合。"""
    result = rrf_fuse(
        embed_results={"c1": 0.9, "c2": 0.5, "c3": 0.1},
        bm25_results={"c2": 5.0, "c3": 3.0, "c4": 2.0},
    )
    # c2 出现在两路 → 分数高
    # c1, c3, c4 各只在一路
    assert isinstance(result, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
    # c2 应该是第一（两路都有）
    assert result[0][1] == "c2"
    # 全部 4 个 chunk 都应出现
    chunk_ids = {cid for _, cid in result}
    assert chunk_ids == {"c1", "c2", "c3", "c4"}


def test_rrf_scores_decrease_with_rank():
    """分数应大致按排名递减。"""
    result = rrf_fuse(
        embed_results={"c1": 0.9, "c2": 0.5, "c3": 0.1},
        bm25_results={"c1": 5.0, "c2": 3.0, "c3": 1.0},
    )
    scores = [s for s, _ in result]
    # 应降序（允许相等）
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


def test_rrf_k_const_changes_score():
    """k_const 越大，低排名贡献越小。"""
    r1 = rrf_fuse(
        embed_results={"c1": 0.9, "c2": 0.5},
        bm25_results={"c2": 0.9},
        k_const=10,
    )
    r2 = rrf_fuse(
        embed_results={"c1": 0.9, "c2": 0.5},
        bm25_results={"c2": 0.9},
        k_const=100,
    )
    # 同样排序，但分数绝对值不同
    assert [c for _, c in r1] == [c for _, c in r2]
    assert r1[0][0] > r2[0][0]


def test_rrf_weights_change_score():
    r1 = rrf_fuse({"c1": 0.9}, {}, w_embed=1.0, w_bm25=0.0)
    r2 = rrf_fuse({"c1": 0.9}, {}, w_embed=0.0, w_bm25=1.0)
    assert r1[0][0] > r2[0][0]


def test_rrf_only_embed_path():
    """只有 embedding 召回。"""
    r = rrf_fuse({"c1": 0.9, "c2": 0.5}, {})
    assert {cid for _, cid in r} == {"c1", "c2"}
    # 分数应降序
    assert r[0][0] >= r[1][0]


def test_rrf_only_bm25_path():
    """只有 BM25 召回。"""
    r = rrf_fuse({}, {"c1": 5.0, "c2": 1.0})
    assert {cid for _, cid in r} == {"c1", "c2"}


def test_rrf_extra_results():
    """额外召回路。"""
    r = rrf_fuse(
        embed_results={"c1": 0.9},
        bm25_results={"c2": 5.0},
        extra_results=[{"c3": 0.8}],
        extra_weights=[0.5],
    )
    assert {cid for _, cid in r} == {"c1", "c2", "c3"}


# === PaperRetriever fusion_strategy ===

@pytest.fixture
def store_with_papers(tmp_path):
    from harness.rag.store import PaperRAGStore
    from harness.rag.embedder import HashBackend
    from harness.rag.chunker import Chunker
    s = PaperRAGStore(
        rag_dir=tmp_path / "rag",
        embedder=HashBackend(),
        chunker=Chunker(max_chunk_chars=500, overlap_chars=100),
    )
    papers = [
        {"arxiv_id": "p1", "title": "Momentum A-shares", "summary": "动量因子表现显著。" * 30},
        {"arxiv_id": "p2", "title": "Value", "summary": "价值投资策略。" * 30},
        {"arxiv_id": "p3", "title": "Reversal", "summary": "反转因子短期表现。" * 30},
    ]
    s.index_papers(papers)
    return s


def test_retriever_default_fusion_is_rrf(store_with_papers):
    """v6-1：默认 fusion_strategy='rrf'（之前 v5 是 weighted）。"""
    from harness.rag.retriever import PaperRetriever
    r = PaperRetriever(store_with_papers)
    assert r.fusion_strategy == "rrf"


def test_retriever_weighted_still_works(store_with_papers):
    """向后兼容：fusion_strategy='weighted' 仍能用。"""
    from harness.rag.retriever import PaperRetriever
    r = PaperRetriever(store_with_papers, fusion_strategy="weighted")
    chunks = r.retrieve("动量", top_k=3)
    assert len(chunks) >= 1
    assert all("score" in c for c in chunks)


def test_retriever_rrf_works(store_with_papers):
    from harness.rag.retriever import PaperRetriever
    r = PaperRetriever(store_with_papers, fusion_strategy="rrf")
    chunks = r.retrieve("动量", top_k=3)
    assert len(chunks) >= 1


def test_retriever_invalid_fusion_strategy(store_with_papers):
    from harness.rag.retriever import PaperRetriever
    with pytest.raises(ValueError):
        PaperRetriever(store_with_papers, fusion_strategy="invalid")


def test_retriever_empty_query(store_with_papers):
    from harness.rag.retriever import PaperRetriever
    r = PaperRetriever(store_with_papers)
    assert r.retrieve("") == []
    assert r.retrieve("   ") == []


def test_retriever_empty_store(tmp_path):
    from harness.rag.store import PaperRAGStore
    from harness.rag.embedder import HashBackend
    from harness.rag.retriever import PaperRetriever
    s = PaperRAGStore(rag_dir=tmp_path / "rag", embedder=HashBackend())
    r = PaperRetriever(s)
    assert r.retrieve("test") == []


def test_retriever_rrf_vs_weighted_same_results(store_with_papers):
    """在简单场景下 RRF 和 weighted 排序应大体一致。"""
    from harness.rag.retriever import PaperRetriever
    r_rrf = PaperRetriever(store_with_papers, fusion_strategy="rrf")
    r_w = PaperRetriever(store_with_papers, fusion_strategy="weighted")
    chunks_rrf = r_rrf.retrieve("动量反转", top_k=3)
    chunks_w = r_w.retrieve("动量反转", top_k=3)
    # 至少 top-1 应相同
    if chunks_rrf and chunks_w:
        assert chunks_rrf[0]["chunk_id"] == chunks_w[0]["chunk_id"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
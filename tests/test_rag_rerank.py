"""
test_rag_rerank.py - Cross-encoder rerank 测试（mock CrossEncoder 不下载真实模型）
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.rag.rerank import (
    BGEReranker,
    _NoOpReranker,
    RerankUnavailable,
    get_default_reranker,
    reset_reranker,
)


@pytest.fixture(autouse=True)
def reset_global_reranker():
    reset_reranker()
    yield
    reset_reranker()


# === Protocol ===

def test_noop_reranker_returns_chunks_unchanged():
    r = _NoOpReranker()
    chunks = [{"chunk_id": "c1", "text": "x"}, {"chunk_id": "c2", "text": "y"}]
    out = r.rerank("query", chunks, top_k=2)
    assert out == chunks


def test_noop_reranker_top_k():
    r = _NoOpReranker()
    chunks = [{"c": i} for i in range(5)]
    out = r.rerank("q", chunks, top_k=2)
    assert len(out) == 2


# === BGEReranker ===

def test_bge_reranker_default_model():
    assert BGEReranker.DEFAULT_MODEL == "BAAI/bge-reranker-base"


def test_bge_reranker_lazy_load():
    """构造时不加载模型。"""
    r = BGEReranker()
    assert r._model is None


def test_bge_reranker_load_failure_raises(monkeypatch):
    """模型加载失败抛 RerankUnavailable（mock 避免真实网络）。"""
    # 替换 CrossEncoder 让它构造时抛异常
    class FakeBrokenCrossEncoder:
        def __init__(self, *a, **kw):
            raise RuntimeError("mocked failure")

    monkeypatch.setattr(
        "sentence_transformers.CrossEncoder",
        FakeBrokenCrossEncoder,
    )
    r = BGEReranker(model_name="any/model")
    with pytest.raises(RerankUnavailable):
        r._load()


def test_bge_reranker_with_mocked_model(monkeypatch):
    """Mock CrossEncoder.predict，验证 rerank 排序逻辑。"""
    r = BGEReranker()
    # mock model
    mock_model = MagicMock()
    # 返回 numpy array 模拟 scores
    import numpy as np
    # c1 (relevant), c2 (irrelevant), c3 (most relevant)
    mock_model.predict.return_value = np.array([0.7, 0.1, 0.9])
    r._model = mock_model

    chunks = [
        {"chunk_id": "c1", "text": "factor pricing"},
        {"chunk_id": "c2", "text": "value investing"},
        {"chunk_id": "c3", "text": "weak factor test asset"},
    ]
    out = r.rerank("factor model", chunks, top_k=3)
    # c3 排第一（0.9），c1 第二（0.7），c2 第三（0.1）
    assert [c["chunk_id"] for c in out] == ["c3", "c1", "c2"]


def test_bge_reranker_top_k(monkeypatch):
    """top_k < len(chunks) 时返回前 top_k。"""
    r = BGEReranker()
    import numpy as np
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.5, 0.9, 0.7, 0.3, 0.8])
    r._model = mock_model

    chunks = [{"chunk_id": f"c{i}", "text": f"text {i}"} for i in range(5)]
    out = r.rerank("q", chunks, top_k=2)
    assert len(out) == 2
    # 0.9 (c1) > 0.8 (c4)
    assert out[0]["chunk_id"] == "c1"
    assert out[1]["chunk_id"] == "c4"


def test_bge_reranker_empty_chunks():
    r = BGEReranker()
    assert r.rerank("q", []) == []
    assert r.rerank("q", [], top_k=5) == []


def test_bge_reranker_load_failure_falls_back(monkeypatch):
    """加载失败 → rerank 返回原顺序（降级）。"""
    class FakeBrokenCrossEncoder:
        def __init__(self, *a, **kw):
            raise RuntimeError("mocked failure")

    monkeypatch.setattr(
        "sentence_transformers.CrossEncoder",
        FakeBrokenCrossEncoder,
    )
    r = BGEReranker(model_name="any/model")
    chunks = [{"chunk_id": "c1", "text": "x"}, {"chunk_id": "c2", "text": "y"}]
    out = r.rerank("q", chunks)
    # load 失败 → 返回原顺序前 top_k
    assert [c["chunk_id"] for c in out] == ["c1", "c2"]


# === get_default_reranker ===

def test_get_default_reranker_disabled(monkeypatch):
    """HARNESS_RAG_RERANK=off → None。"""
    monkeypatch.setenv("HARNESS_RAG_RERANK", "off")
    assert get_default_reranker() is None


def test_get_default_reranker_off_explicit():
    """显式 enabled=False → None。"""
    assert get_default_reranker(enabled=False) is None


def test_get_default_reranker_enabled_lazy_loads(monkeypatch):
    """enabled=True → 构造 BGEReranker 单例（不下载因为只 load 不 predict）。"""
    monkeypatch.setenv("HARNESS_RAG_RERANK", "on")
    # 不 mock 实际 load（避免真实网络）
    # 只验证返回 BGEReranker 实例
    import os
    # monkeypatch 掉 load 避免真实失败
    monkeypatch.setattr(
        "harness.rag.rerank.BGEReranker._load",
        lambda self: None,
    )
    r = get_default_reranker()
    assert r is not None
    assert isinstance(r, BGEReranker)


def test_get_default_reranker_singleton(monkeypatch):
    """多次调用应返回同一实例。"""
    monkeypatch.setattr(
        "harness.rag.rerank.BGEReranker._load",
        lambda self: None,
    )
    r1 = get_default_reranker(enabled=True)
    r2 = get_default_reranker(enabled=True)
    assert r1 is r2


# === PaperRetriever rerank 集成 ===

@pytest.fixture
def store(tmp_path):
    from harness.rag.store import PaperRAGStore
    from harness.rag.embedder import HashBackend
    s = PaperRAGStore(rag_dir=tmp_path / "rag", embedder=HashBackend())
    papers = [
        {"arxiv_id": "p1", "title": "Momentum", "summary": "动量因子表现。" * 30},
        {"arxiv_id": "p2", "title": "Value", "summary": "价值投资。" * 30},
    ]
    s.index_papers(papers)
    return s


def test_retriever_without_reranker(store):
    """无 reranker 时不触发 _rerank_rank。"""
    from harness.rag.retriever import PaperRetriever
    r = PaperRetriever(store)
    chunks = r.retrieve("动量", top_k=2)
    assert len(chunks) >= 1
    # 无 _rerank_rank 字段
    assert all("_rerank_rank" not in c for c in chunks)


def test_retriever_with_mock_reranker(store):
    """Mock reranker 触发，_rerank_rank 应被设置。"""
    from harness.rag.retriever import PaperRetriever

    # Mock reranker
    class MockReranker:
        @property
        def model_name(self):
            return "mock"

        def rerank(self, query, chunks, *, top_k=None):
            # 反转顺序
            return list(reversed(chunks))[:top_k]

    r = PaperRetriever(store, reranker=MockReranker())
    chunks = r.retrieve("动量", top_k=2)
    assert len(chunks) >= 1
    # _rerank_rank 应有
    assert all("_rerank_rank" in c for c in chunks)
    # rank 从 1 开始
    assert chunks[0]["_rerank_rank"] == 1


def test_retriever_reranker_failure_falls_back(store):
    """reranker 抛异常时降级到原排序。"""
    from harness.rag.retriever import PaperRetriever

    class BrokenReranker:
        @property
        def model_name(self):
            return "broken"

        def rerank(self, query, chunks, *, top_k=None):
            raise RuntimeError("reranker down")

    r = PaperRetriever(store, reranker=BrokenReranker())
    chunks = r.retrieve("动量", top_k=2)
    # 不崩，返回原排序
    assert len(chunks) >= 1
    # 没有 _rerank_rank
    assert all("_rerank_rank" not in c for c in chunks)


def test_retriever_with_noop_reranker(store):
    """NoOpReranker 直接 pass-through，_rerank_rank 应被设。"""
    from harness.rag.retriever import PaperRetriever
    r = PaperRetriever(store, reranker=_NoOpReranker())
    chunks = r.retrieve("动量", top_k=2)
    assert all("_rerank_rank" in c for c in chunks)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
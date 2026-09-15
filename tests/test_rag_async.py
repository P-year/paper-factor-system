"""
test_rag_async.py - AsyncEmbedder + query cache 测试

覆盖：
1. encode_async 返回 Future
2. encode_cached 命中/未命中
3. encode_many_cached 批量
4. LRU 淘汰
5. PaperRetriever 用 async_embedder 集成
"""
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np

from harness.rag.async_embed import (
    AsyncEmbedder,
    get_async_embedder,
    reset_async_embedder,
    shutdown_executor,
    _query_hash,
    _QueryCache,
)
from harness.rag.embedder import HashBackend
from harness.rag.retriever import PaperRetriever
from harness.rag.store import PaperRAGStore
from harness.rag.chunker import Chunker


@pytest.fixture(autouse=True)
def reset_state():
    reset_async_embedder()
    shutdown_executor(wait=True)
    yield
    reset_async_embedder()
    shutdown_executor(wait=True)


# === _query_hash ===

def test_query_hash_deterministic():
    h1 = _query_hash("动量因子")
    h2 = _query_hash("动量因子")
    assert h1 == h2
    assert len(h1) == 16


def test_query_hash_different_inputs():
    assert _query_hash("a") != _query_hash("b")


# === _QueryCache ===

def test_cache_basic():
    c = _QueryCache(max_size=10)
    c.put("k1", "v1")
    assert c.get("k1") == "v1"
    assert c.get("k2") is None


def test_cache_lru_eviction():
    c = _QueryCache(max_size=2)
    c.put("k1", "v1")
    c.put("k2", "v2")
    c.put("k3", "v3")  # 触发淘汰
    assert c.get("k1") is None
    assert c.get("k2") == "v2"
    assert c.get("k3") == "v3"


def test_cache_lru_refresh_on_get():
    c = _QueryCache(max_size=2)
    c.put("k1", "v1")
    c.put("k2", "v2")
    _ = c.get("k1")  # 访问 k1 → 移到末尾
    c.put("k3", "v3")  # 应淘汰 k2
    assert c.get("k1") == "v1"
    assert c.get("k2") is None


def test_cache_thread_safe():
    c = _QueryCache(max_size=100)
    def worker(i):
        for j in range(10):
            c.put(f"k{i}_{j}", f"v{i}_{j}")
            _ = c.get(f"k{i}_{j}")
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(c) <= 100


# === AsyncEmbedder.encode_async ===

def test_encode_async_returns_future():
    from concurrent.futures import Future
    emb = HashBackend()
    ae = AsyncEmbedder(emb)
    future = ae.encode_async(["test"])
    assert isinstance(future, Future)
    vecs = future.result(timeout=5)
    assert vecs.shape == (1, emb.dim)


def test_encode_async_multiple():
    emb = HashBackend()
    ae = AsyncEmbedder(emb)
    futures = [ae.encode_async([f"text{i}"]) for i in range(3)]
    vecs_list = [f.result(timeout=5) for f in futures]
    assert all(v.shape == (1, emb.dim) for v in vecs_list)


# === AsyncEmbedder.encode_cached ===

def test_encode_cached_miss_then_hit():
    """第一次 miss 缓存 + embed；第二次 hit 缓存。"""
    emb = HashBackend()
    ae = AsyncEmbedder(emb, cache_size=10)
    assert ae.cache_stats()["size"] == 0
    v1 = ae.encode_cached("test query")
    assert ae.cache_stats()["size"] == 1
    v2 = ae.encode_cached("test query")
    # 命中：值等价
    import numpy as np
    np.testing.assert_array_equal(v1, v2)


def test_encode_cached_empty_query():
    emb = HashBackend()
    ae = AsyncEmbedder(emb)
    assert ae.encode_cached("") is None


def test_encode_many_cached():
    emb = HashBackend()
    ae = AsyncEmbedder(emb, cache_size=10)
    queries = ["q1", "q2", "q1", "q3"]  # q1 重复
    out = ae.encode_many_cached(queries)
    assert all(v is not None for v in out)
    # out[0] and out[2] 值应等价（同一 query）
    import numpy as np
    np.testing.assert_array_equal(out[0], out[2])


# === PaperRetriever + AsyncEmbedder ===

@pytest.fixture
def store_with_papers(tmp_path):
    s = PaperRAGStore(rag_dir=tmp_path / "rag", embedder=HashBackend())
    papers = [
        {"arxiv_id": "p1", "title": "Momentum", "summary": "动量因子表现。" * 30},
        {"arxiv_id": "p2", "title": "Value", "summary": "价值投资。" * 30},
    ]
    s.index_papers(papers)
    return s


def test_retriever_with_async_embedder(store_with_papers):
    """retriever 用 AsyncEmbedder 不破坏功能。"""
    emb = HashBackend()
    ae = AsyncEmbedder(emb)
    r = PaperRetriever(store_with_papers, async_embedder=ae)
    chunks = r.retrieve("动量", top_k=2)
    assert len(chunks) >= 1


def test_retriever_async_caches_query_embedding(store_with_papers):
    """相同 query 第二次走 cache（不重复 embed）。"""
    emb = HashBackend()
    ae = AsyncEmbedder(emb)
    r = PaperRetriever(store_with_papers, async_embedder=ae)
    r.retrieve("动量因子", top_k=2)
    assert ae.cache_stats()["size"] == 1
    r.retrieve("动量因子", top_k=2)  # 第二次
    assert ae.cache_stats()["size"] == 1  # 没增加


def test_retriever_without_async_embedder_works(store_with_papers):
    """async_embedder=None 时走同步 embed（向后兼容）。"""
    r = PaperRetriever(store_with_papers)
    chunks = r.retrieve("动量", top_k=2)
    assert len(chunks) >= 1


# === Executor shutdown ===

def test_executor_lifecycle():
    from harness.rag.async_embed import get_executor
    e1 = get_executor()
    e2 = get_executor()
    assert e1 is e2  # 单例

    shutdown_executor(wait=True)
    e3 = get_executor()
    assert e3 is not e1  # 重建


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
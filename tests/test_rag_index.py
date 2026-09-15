"""
test_rag_index.py - FAISSIndex 测试

覆盖：
1. add + search roundtrip
2. cosine score ∈ [0, 1]
3. save / load 持久化
4. 空索引 search 不崩
5. __len__
6. 去重（重复 add 同一 id）
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np

from harness.rag.index import FAISSIndex


def test_index_init_dim():
    idx = FAISSIndex(dim=64)
    assert idx.dim == 64
    assert idx.metric == "cosine"


def test_index_invalid_metric():
    with pytest.raises(ValueError):
        FAISSIndex(dim=64, metric="unknown")


def test_index_empty_search():
    idx = FAISSIndex(dim=64)
    vec = np.zeros(64, dtype=np.float32)
    results = idx.search(vec, top_k=5)
    assert results == []


def test_index_add_and_search():
    """add 10 个不同向量 → search 应能命中 top-1。"""
    np.random.seed(42)
    dim = 64
    vecs = np.random.rand(10, dim).astype(np.float32)
    ids = [f"chunk_{i}" for i in range(10)]
    idx = FAISSIndex(dim=dim)
    added = idx.add(vecs, ids=ids)
    assert len(added) == 10
    assert len(idx) == 10

    # search 第一条
    results = idx.search(vecs[0], top_k=3)
    assert len(results) == 3
    # top-1 应是自身
    top_score, top_id = results[0]
    assert top_id == "chunk_0"
    # cosine 模式下 self-match score ≈ 1.0
    assert top_score > 0.99


def test_index_cosine_range():
    """cosine 模式 score ∈ [0, 1]。"""
    np.random.seed(0)
    dim = 32
    idx = FAISSIndex(dim=dim)
    vecs = np.random.rand(5, dim).astype(np.float32)
    idx.add(vecs, ids=[f"c{i}" for i in range(5)])

    q = np.random.rand(dim).astype(np.float32)
    results = idx.search(q, top_k=5)
    for score, _ in results:
        assert 0.0 <= score <= 1.0


def test_index_dedup_same_id():
    """同一 id 重复 add 只算一次。"""
    vec = np.random.rand(1, 32).astype(np.float32)
    idx = FAISSIndex(dim=32)
    idx.add(vec, ids=["c1"])
    added = idx.add(vec, ids=["c1"])  # 重复
    assert added == []  # 去重后无新增
    assert len(idx) == 1


def test_index_save_load_roundtrip(tmp_path):
    np.random.seed(1)
    dim = 16
    vecs = np.random.rand(5, dim).astype(np.float32)
    ids = [f"c{i}" for i in range(5)]

    idx1 = FAISSIndex(dim=dim)
    idx1.add(vecs, ids=ids)
    path = tmp_path / "test.faiss"
    idx1.save(path)

    # 加载
    idx2 = FAISSIndex(dim=dim)
    idx2.load(path)
    assert len(idx2) == 5
    # search 应能命中
    results = idx2.search(vecs[0], top_k=1)
    assert results[0][1] == "c0"


def test_index_l2_metric():
    """l2 模式也能工作。"""
    dim = 16
    idx = FAISSIndex(dim=dim, metric="l2")
    vecs = np.array([[1.0] * dim, [2.0] * dim, [3.0] * dim], dtype=np.float32)
    idx.add(vecs, ids=["a", "b", "c"])
    q = np.array([1.0] * dim, dtype=np.float32)
    results = idx.search(q, top_k=3)
    assert len(results) == 3
    # 离自己最近
    assert results[0][1] == "a"


def test_index_search_top_k_larger_than_ntotal():
    """top_k > ntotal 时只返回 ntotal 条。"""
    idx = FAISSIndex(dim=8)
    idx.add(np.random.rand(3, 8).astype(np.float32), ids=["a", "b", "c"])
    results = idx.search(np.random.rand(8).astype(np.float32), top_k=10)
    assert len(results) == 3


def test_index_has_id():
    idx = FAISSIndex(dim=8)
    idx.add(np.random.rand(2, 8).astype(np.float32), ids=["x", "y"])
    assert idx.has_id("x") is True
    assert idx.has_id("nope") is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
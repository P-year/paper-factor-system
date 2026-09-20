"""
test_rag_hnsw.py - HNSW 索引测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np

from harness.rag.index import FAISSIndex


def test_hnsw_metric_supported():
    idx = FAISSIndex(dim=64, metric="hnsw")
    assert idx.dim == 64
    assert idx.metric == "hnsw"


def test_hnsw_add_and_search():
    np.random.seed(42)
    dim = 64
    vecs = np.random.rand(100, dim).astype(np.float32)
    ids = [f"vec_{i}" for i in range(100)]

    idx = FAISSIndex(dim=dim, metric="hnsw")
    added = idx.add(vecs, ids=ids)
    assert len(added) == 100
    assert len(idx) == 100

    results = idx.search(vecs[0], top_k=3)
    assert len(results) == 3
    # top-1 应是自身（cosine）
    assert results[0][1] == "vec_0"
    assert results[0][0] > 0.95


def test_hnsw_cosine_range():
    """HNSW + cosine → score ∈ [0, 1]。"""
    np.random.seed(0)
    dim = 32
    idx = FAISSIndex(dim=dim, metric="hnsw")
    vecs = np.random.rand(10, dim).astype(np.float32)
    idx.add(vecs, ids=[f"c{i}" for i in range(10)])

    q = np.random.rand(dim).astype(np.float32)
    results = idx.search(q, top_k=5)
    for score, _ in results:
        assert 0.0 <= score <= 1.0


def test_hnsw_save_load_roundtrip(tmp_path):
    np.random.seed(1)
    dim = 16
    vecs = np.random.rand(5, dim).astype(np.float32)
    ids = [f"c{i}" for i in range(5)]

    idx1 = FAISSIndex(dim=dim, metric="hnsw")
    idx1.add(vecs, ids=ids)
    path = tmp_path / "hnsw.faiss"
    idx1.save(path)

    idx2 = FAISSIndex(dim=dim, metric="hnsw")
    idx2.load(path)
    assert len(idx2) == 5
    # search 应能命中
    results = idx2.search(vecs[0], top_k=1)
    assert results[0][1] == "c0"


def test_hnsw_dedup():
    vec = np.random.rand(1, 32).astype(np.float32)
    idx = FAISSIndex(dim=32, metric="hnsw")
    idx.add(vec, ids=["c1"])
    added = idx.add(vec, ids=["c1"])  # 重复
    assert added == []
    assert len(idx) == 1


def test_hnsw_large_scale_basic():
    """1000 vectors 测试基本可工作性。"""
    np.random.seed(0)
    dim = 64
    n = 1000
    vecs = np.random.rand(n, dim).astype(np.float32)
    ids = [f"c{i}" for i in range(n)]

    idx = FAISSIndex(dim=dim, metric="hnsw")
    idx.add(vecs, ids=ids)
    assert len(idx) == n

    q = np.random.rand(dim).astype(np.float32)
    results = idx.search(q, top_k=10)
    assert len(results) == 10
    # top-1 score 应合理（cosine 0~1）
    assert 0 < results[0][0] <= 1.0


def test_hnsw_vs_flat_basic_equivalence():
    """同输入下，HNSW top-1 应与 Flat 接近（不是必须一致）。"""
    np.random.seed(7)
    dim = 32
    vecs = np.random.rand(50, dim).astype(np.float32)
    ids = [f"c{i}" for i in range(50)]

    idx_flat = FAISSIndex(dim=dim, metric="cosine")
    idx_flat.add(vecs, ids=ids)

    idx_hnsw = FAISSIndex(dim=dim, metric="hnsw")
    idx_hnsw.add(vecs, ids=ids)

    q = vecs[0]
    r_flat = idx_flat.search(q, top_k=3)
    r_hnsw = idx_hnsw.search(q, top_k=3)
    # top-1 应相同（自身）
    assert r_flat[0][1] == r_hnsw[0][1] == "c0"


def test_hnsw_invalid_metric_raises():
    with pytest.raises(ValueError):
        FAISSIndex(dim=64, metric="unknown_metric")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
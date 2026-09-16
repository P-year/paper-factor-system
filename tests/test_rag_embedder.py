"""
test_rag_embedder.py - EmbeddingBackend 测试

覆盖：
1. HashBackend shape / dtype / dim
2. HashBackend 确定性（相同输入 → 相同输出）
3. HashBackend 中文/英文
4. HashBackend 空输入
5. SentenceTransformerBackend 懒加载（mock）
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np

from harness.rag.embedder import (
    HashBackend,
    SentenceTransformerBackend,
    _EmbedderUnavailable,
    get_default_backend,
)


# === HashBackend ===

def test_hash_encode_shape():
    h = HashBackend(dim=256)
    vecs = h.encode(["text1", "text2", "text3"])
    assert vecs.shape == (3, 256)
    assert vecs.dtype == np.float32


def test_hash_encode_empty():
    h = HashBackend(dim=256)
    vecs = h.encode([])
    assert vecs.shape == (0, 256)


def test_hash_encode_empty_string():
    h = HashBackend(dim=256)
    vecs = h.encode(["", "x"])
    assert vecs.shape == (2, 256)
    # 空字符串 → 全 0
    assert np.allclose(vecs[0], 0)
    assert not np.allclose(vecs[1], 0)


def test_hash_deterministic():
    h = HashBackend(dim=256)
    v1 = h.encode(["测试文本"])
    v2 = h.encode(["测试文本"])
    np.testing.assert_array_equal(v1, v2)


def test_hash_l2_normalized():
    """每行 L2 归一化 → norm=1（除非全 0）。"""
    h = HashBackend(dim=256)
    vecs = h.encode(["测试文本", "momentum factor strategy"])
    norms = np.linalg.norm(vecs, axis=1)
    for n in norms:
        if n > 0:
            np.testing.assert_allclose(n, 1.0, atol=1e-5)


def test_hash_chinese():
    h = HashBackend(dim=256)
    vecs = h.encode(["动量反转因子"])
    # 中文 token 单字也能 hash
    assert vecs.shape == (1, 256)


def test_hash_dim_property():
    h = HashBackend(dim=128)
    assert h.dim == 128
    assert "hash" in h.model_name


# === SentenceTransformerBackend ===

def test_st_dim_default():
    """未加载模型时 dim 是默认值。"""
    st = SentenceTransformerBackend()
    assert st.dim == 512  # bge-small-zh 默认


def test_st_model_name():
    st = SentenceTransformerBackend()
    assert "bge" in st.model_name or "small" in st.model_name


def test_st_lazy_no_load_on_init():
    """构造时不加载模型。"""
    st = SentenceTransformerBackend()
    assert st._model is None


def test_st_load_failure_raises(monkeypatch):
    """模型加载失败抛 _EmbedderUnavailable（mock 不走真实网络）。"""
    def fake_load(self):
        raise _EmbedderUnavailable("mocked failure")
    monkeypatch.setattr(SentenceTransformerBackend, "_load", fake_load)
    st = SentenceTransformerBackend(model_name="any/model")
    with pytest.raises(_EmbedderUnavailable):
        st.encode(["test"])


def test_st_encode_unloaded_no_op(monkeypatch):
    """模型未加载时 encode 抛异常（mock 不走真实网络）。"""
    def fake_load(self):
        raise _EmbedderUnavailable("mocked not loaded")
    monkeypatch.setattr(SentenceTransformerBackend, "_load", fake_load)
    st = SentenceTransformerBackend(model_name="any/model")
    with pytest.raises(_EmbedderUnavailable):
        st.encode(["x"])


# === get_default_backend ===

def test_get_default_backend_hash():
    emb = get_default_backend(prefer="hash")
    assert isinstance(emb, HashBackend)


def test_get_default_backend_auto_fallback(monkeypatch):
    """auto 模式：尝试 ST，失败时降级 HashBackend。

    Mock ST._load 让它抛 EmbedderUnavailable，避免真实网络请求。
    """
    from harness.rag.embedder import _EmbedderUnavailable
    def fake_load(self):
        raise _EmbedderUnavailable("mocked no network")
    monkeypatch.setattr(SentenceTransformerBackend, "_load", fake_load)
    emb = get_default_backend(prefer="auto")
    assert isinstance(emb, HashBackend)
    assert emb.dim > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
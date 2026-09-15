"""
harness/rag/embedder.py - EmbeddingBackend 抽象 + 实现

设计：
- EmbeddingBackend Protocol（dim / model_name / encode）
- HashBackend：纯 Python，零依赖，dim=256，确定性（hash(token) % 256）
- SentenceTransformerBackend：sentence-transformers，懒加载 bge-small-zh-v1.5

用法：
    from harness.rag.embedder import HashBackend, SentenceTransformerBackend

    # 离线/测试
    emb = HashBackend()

    # 生产（首次 encode 时下载/加载模型 ~5-10s + 100MB）
    emb = SentenceTransformerBackend()
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable

import numpy as np


# === Protocol ===

@runtime_checkable
class EmbeddingBackend(Protocol):
    """Embedding backend 接口。"""

    @property
    def dim(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    def encode(self, texts: List[str], *, batch_size: int = 32) -> np.ndarray: ...


# === HashBackend（零依赖 fallback） ===

_TOKEN_RE = re.compile(r"[一-鿿]|[A-Za-z0-9]+")


class HashBackend:
    """纯 Python embedding：token hash → 256 维 bag-of-hash 向量 + L2 归一化。

    特性：
    - 零依赖（仅 numpy）
    - 确定性：相同输入 → 相同向量
    - dim=256
    - 用于离线 / CI / 模型下载失败时的 fallback
    """

    def __init__(self, dim: int = 256):
        self._dim = dim
        self._model_name = f"hash-{dim}d"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts: List[str], *, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            if not t:
                continue
            for tok in _TOKEN_RE.findall(t):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
                out[i, h % self._dim] += 1.0
            # L2 归一化
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


# === SentenceTransformerBackend（懒加载） ===

class _EmbedderUnavailable(RuntimeError):
    """模型加载失败时抛出。"""


class SentenceTransformerBackend:
    """sentence-transformers backend，懒加载。

    第一次 encode() 才下载/加载模型。
    加载失败抛 _EmbedderUnavailable，调用方应降级到 HashBackend。
    """

    DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
    DEFAULT_DIM = 512

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: Optional[Path] = None,
        device: str = "cpu",
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._device = device
        self._model = None  # 懒加载
        self._dim = self.DEFAULT_DIM  # bge-small-zh 实际 512

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            import os
            kwargs = {"device": self._device}
            if self._cache_dir is not None:
                kwargs["cache_folder"] = str(self._cache_dir)
            # 静默 huggingface 下载日志（可选）
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            self._model = SentenceTransformer(self._model_name, **kwargs)
            # 真实 dim
            test_vec = self._model.encode(["test"], convert_to_numpy=True)
            self._dim = int(test_vec.shape[1])
        except Exception as e:
            raise _EmbedderUnavailable(
                f"failed to load {self._model_name}: {e}"
            ) from e

    def encode(self, texts: List[str], *, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        self._load()
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 归一化，与 FAISS IndexFlatIP 配合
        )
        return vecs.astype(np.float32)


# === 工厂 ===

def get_default_backend(prefer: str = "auto") -> EmbeddingBackend:
    """获取默认 embedder。

    prefer:
        - "auto"（默认）：先尝试 SentenceTransformerBackend，失败降级 HashBackend
        - "sentence_transformer"：强制 ST（失败抛异常）
        - "hash"：直接用 HashBackend
    """
    if prefer == "hash":
        return HashBackend()
    st = SentenceTransformerBackend()
    if prefer == "sentence_transformer":
        return st
    # auto：尝试 ST
    try:
        st._load()
        return st
    except _EmbedderUnavailable:
        return HashBackend()
"""
harness/rag/rerank.py - Cross-encoder rerank 精排

设计：
- Reranker Protocol（predict score for (query, chunk_text) pairs）
- BGEReranker 实现（sentence-transformers CrossEncoder）
- 默认关闭（HARNESS_RAG_RERANK=on 启用）
- 失败降级到原排序（不阻塞）

用法：
    from harness.rag.rerank import BGEReranker
    reranker = BGEReranker()  # 懒加载 bge-reranker-base
    reranked = reranker.rerank(query, chunks, top_k=5)
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    """Rerank 抽象接口。"""

    @property
    def model_name(self) -> str: ...

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        *,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """返回重排后的 chunks（按相关性降序）。"""
        ...


class RerankUnavailable(RuntimeError):
    """reranker 加载失败。"""


class _NoOpReranker:
    """trivially pass-through（用于测试 / rerank 关闭）。"""

    @property
    def model_name(self) -> str:
        return "noop"

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        *,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return chunks[:top_k] if top_k else chunks


class BGEReranker:
    """sentence-transformers CrossEncoder reranker。

    懒加载 bge-reranker-base（~200MB）。失败抛 RerankUnavailable，调用方降级。
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-base"
    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_dir: Optional[str] = None,
    ):
        self._model_name = model_name
        self._batch_size = batch_size
        self._cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import CrossEncoder
                kwargs = {}
                if self._cache_dir:
                    kwargs["cache_folder"] = self._cache_dir
                self._model = CrossEncoder(self._model_name, **kwargs)
            except Exception as e:
                raise RerankUnavailable(
                    f"failed to load reranker {self._model_name}: {e}"
                ) from e

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        *,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """重排。返回按相关性降序的 chunks。"""
        if not chunks:
            return []
        top_k = top_k or len(chunks)
        try:
            self._load()
        except RerankUnavailable:
            # 加载失败：降级到原顺序
            return chunks[:top_k]

        # 构造 (query, text) pairs
        texts = [c.get("text", "") for c in chunks]
        pairs = [[query, t] for t in texts]
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # 排序：scores 降序
        ranked = sorted(
            zip(scores.tolist(), chunks),
            key=lambda x: -x[0],
        )
        return [c for _, c in ranked[:top_k]]


_reranker: Optional[Reranker] = None
_reranker_lock = threading.Lock()


def get_default_reranker(
    enabled: Optional[bool] = None,
    *,
    model_name: str = BGEReranker.DEFAULT_MODEL,
) -> Optional[Reranker]:
    """全局 reranker 单例（懒加载）。

    Args:
        enabled: None → 读 HARNESS_RAG_RERANK env（"on"/"off"/"auto"）
                 True/False → 显式开关
        model_name: reranker 模型名

    Returns:
        None（关闭）或 Reranker 实例
    """
    global _reranker
    if enabled is None:
        env = os.getenv("HARNESS_RAG_RERANK", "off").lower()
        if env in ("off", "0", "false", ""):
            return None
        enabled = True

    if not enabled:
        return None

    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = BGEReranker(model_name=model_name)
    return _reranker


def reset_reranker() -> None:
    """测试用：清空全局单例。"""
    global _reranker
    with _reranker_lock:
        _reranker = None
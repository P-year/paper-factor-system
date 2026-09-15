"""
harness/rag/async_embed.py - ThreadPoolExecutor 后台 embed + query cache

目的：
- 异步：index_papers() 立即返回，后台线程跑 embed + 写索引
- query cache：相同 query 复用 embed 结果（按 query string hash）

设计：
- AsyncEmbedder 包装 embedder，提供 encode_async / encode_cached
- ThreadPoolExecutor 单例（lazy init）
- query cache 用 OrderedDict（LRU）+ threading.Lock 线程安全
- 缓存默认 256 条；超限 FIFO 淘汰

用法：
    from harness.rag.async_embed import get_async_embedder

    async_emb = get_async_embedder(embedder=HashBackend())
    # 异步
    future = async_emb.encode_async(["query text"])
    vecs = future.result(timeout=30)
    # 缓存
    vecs = async_emb.encode_cached("query text")
"""
from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from collections import OrderedDict


# === Query cache ===

class _QueryCache:
    """LRU query → embedding 缓存（thread-safe）。"""

    def __init__(self, max_size: int = 256):
        self._max_size = max_size
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            # LRU: 移到末尾
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            # 超限淘汰最老
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


def _query_hash(query: str) -> str:
    """query string → stable hash（md5 16 位足够区分）。"""
    return hashlib.md5(query.encode("utf-8")).hexdigest()[:16]


# === Executor ===

_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()


def get_executor() -> ThreadPoolExecutor:
    """懒加载全局 ThreadPoolExecutor。"""
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                # 默认 2 worker（RAG 嵌入非 CPU-bound 极重任务）
                max_workers = int(os.getenv("HARNESS_RAG_EMBED_WORKERS", "2"))
                _executor = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="rag-embed",
                )
    return _executor


def shutdown_executor(wait: bool = True) -> None:
    global _executor
    if _executor is not None:
        with _executor_lock:
            if _executor is not None:
                _executor.shutdown(wait=wait)
                _executor = None


# === AsyncEmbedder ===

class AsyncEmbedder:
    """Embedder 包装：异步提交 + query 缓存。

    用法：
        ae = AsyncEmbedder(embedder)
        ae.encode_async(["text1", "text2"])   # 立即返回 Future
        ae.encode_cached("query text")        # 同步，先查缓存
    """

    def __init__(self, embedder, *, cache_size: int = 256, executor: Optional[ThreadPoolExecutor] = None):
        self.embedder = embedder
        self.cache = _QueryCache(max_size=cache_size)
        self.executor = executor or get_executor()

    def encode_async(self, texts: List[str], **kwargs) -> Future:
        """异步执行 embed，返回 Future[np.ndarray]。"""
        return self.executor.submit(self.embedder.encode, texts, **kwargs)

    def encode_cached(self, query: str) -> Any:
        """单 query 同步 embedding + 缓存。

        空字符串返回 None（不缓存）。
        """
        if not query or not query.strip():
            return None
        key = _query_hash(query)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        vecs = self.embedder.encode([query])
        if vecs is not None and len(vecs) > 0:
            self.cache.put(key, vecs[0])
            return vecs[0]
        return None

    def encode_many_cached(self, queries: List[str]) -> List[Any]:
        """批量 query embedding，未命中的并行 embed。"""
        out: List[Any] = [None] * len(queries)
        futures: Dict[int, Future] = {}
        for i, q in enumerate(queries):
            key = _query_hash(q)
            cached = self.cache.get(key)
            if cached is not None:
                out[i] = cached
            else:
                futures[i] = self.executor.submit(self.embedder.encode, [q])
        for i, f in futures.items():
            try:
                vecs = f.result(timeout=30)
                if vecs is not None and len(vecs) > 0:
                    out[i] = vecs[0]
                    self.cache.put(_query_hash(queries[i]), vecs[0])
            except Exception:
                pass
        return out

    def cache_stats(self) -> Dict[str, int]:
        return {
            "size": len(self.cache),
            "max_size": self.cache._max_size,
        }


_async_embedder: Optional[AsyncEmbedder] = None
_async_lock = threading.Lock()


def get_async_embedder(
    embedder=None,
    *,
    cache_size: int = 256,
) -> AsyncEmbedder:
    """获取/构造全局 AsyncEmbedder。"""
    global _async_embedder
    if _async_embedder is None:
        with _async_lock:
            if _async_embedder is None:
                if embedder is None:
                    from harness.rag.embedder import HashBackend
                    embedder = HashBackend()
                _async_embedder = AsyncEmbedder(embedder, cache_size=cache_size)
    return _async_embedder


def reset_async_embedder() -> None:
    """测试用：清空全局实例。"""
    global _async_embedder
    with _async_lock:
        _async_embedder = None
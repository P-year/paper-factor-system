"""
harness/rag/bm25.py - BM25 keyword scoring（与 embedding 混合用）

设计：
- 用 rank_bm25（已装）建索引
- 增量更新：append_chunk 时更新 BM25 索引
- search(query) 返回 [(bm25_score, chunk_id)]
- score 不归一化（让 retriever 自己 normalize 与 cosine 加权）

用法：
    from harness.rag.bm25 import BM25Index
    idx = BM25Index()
    idx.add_chunks([{chunk_id, text}, ...])
    results = idx.search("动量因子", top_k=5)  # [(score, chunk_id), ...]
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False


_TOKEN_RE = re.compile(r"[一-鿿]|[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """中英混合 token 化。"""
    return _TOKEN_RE.findall(text or "")


class BM25Index:
    """BM25 关键词索引（与 embedding cosine 配合做混合排序）。

    若 rank_bm25 未装，所有方法 no-op + search 返回空。
    """

    def __init__(self):
        self._chunk_ids: List[str] = []
        self._corpus: List[List[str]] = []  # 每个 chunk 的 token 列表
        self._bm25 = None
        self._dirty = True  # 是否需要重建 bm25

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """增量添加 chunks。返回新增数。"""
        if not _HAS_BM25 or not chunks:
            return 0
        added = 0
        for ch in chunks:
            cid = ch.get("chunk_id")
            text = ch.get("text", "")
            if not cid or not text:
                continue
            # 去重（已有跳过）
            if cid in self._chunk_ids:
                continue
            self._chunk_ids.append(cid)
            self._corpus.append(_tokenize(text))
            added += 1
        if added:
            self._dirty = True
        return added

    def search(self, query: str, *, top_k: int = 5) -> List[Tuple[float, str]]:
        """返回 [(bm25_score, chunk_id), ...] 按 score 降序。"""
        if not _HAS_BM25 or not self._chunk_ids:
            return []
        self._rebuild_if_needed()
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # 排序取 top_k
        ranked = sorted(
            enumerate(scores),
            key=lambda kv: kv[1],
            reverse=True,
        )
        out = []
        for idx, sc in ranked[:top_k]:
            if sc <= 0:
                continue
            out.append((float(sc), self._chunk_ids[idx]))
        return out

    def __len__(self) -> int:
        return len(self._chunk_ids)

    # === persistence ===

    def to_dict(self) -> Dict[str, Any]:
        """序列化 token corpus（可重建 BM25）。"""
        return {
            "chunk_ids": self._chunk_ids,
            "corpus": self._corpus,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        self._chunk_ids = data.get("chunk_ids", [])
        self._corpus = data.get("corpus", [])
        self._dirty = True
        # 立即重建以供首次 search
        self._rebuild_if_needed()

    # === 内部 ===

    def _rebuild_if_needed(self):
        if not self._dirty:
            return
        if not _HAS_BM25 or not self._corpus:
            self._bm25 = None
            self._dirty = False
            return
        self._bm25 = BM25Okapi(self._corpus)
        self._dirty = False


def is_bm25_available() -> bool:
    return _HAS_BM25
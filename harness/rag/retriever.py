"""
harness/rag/retriever.py - PaperRetriever（主入口）

用法：
    from harness.rag import PaperRetriever, get_default_store

    store = get_default_store()
    retriever = PaperRetriever(store)

    chunks = retriever.retrieve("动量因子在 A 股的表现", top_k=5)
    for c in chunks:
        print(c["score"], c["paper_title"], c["text"][:80])

    # 自动从 state 构造 query
    chunks = retriever.retrieve_for_state(state, top_k=5)

    # 拼成 prompt
    prompt_block = retriever.format_context(chunks)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np


class PaperRetriever:
    """论文检索器（基于 PaperRAGStore）。

    v5-3：混合排序 = embedding cosine + BM25 加权。
    v5-4：query embedding 走 AsyncEmbedder 缓存。
    权重：embedding_weight + bm25_weight = 1.0
    """

    def __init__(
        self,
        store,
        *,
        default_top_k: int = 5,
        embedding_weight: float = 0.7,
        bm25_weight: float = 0.3,
        async_embedder=None,
    ):
        self.store = store
        self.default_top_k = default_top_k
        self.embedding_weight = embedding_weight
        self.bm25_weight = bm25_weight
        self.async_embedder = async_embedder  # v5-4：None → 用 sync encode

    # === 主入口 ===

    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """按 query 文本检索（embedding + BM25 混合排序）。

        返回 List[Dict]，每个含 {chunk_id, source_id, text, arxiv_id, paper_title, paper_link, paper_published, score, _score_embed, _score_bm25}。
        """
        if not query or not query.strip():
            return []
        top_k = top_k or self.default_top_k
        if self.store.count() == 0:
            return []

        query_clean = query.strip()
        # 多取一些候选避免漏召回
        candidate_k = max(top_k * 3, 20)

        # Embedding cosine（v5-4：走 async cache）
        embed_results: Dict[str, float] = {}
        try:
            if self.async_embedder is not None:
                q_vec = self.async_embedder.encode_cached(query_clean)
            else:
                q_vec = self.store.embedder.encode([query_clean])[0]
            if q_vec is not None:
                for score, cid in self.store.search(q_vec, top_k=candidate_k):
                    embed_results[cid] = score
        except Exception:
            pass

        # BM25
        bm25_results: Dict[str, float] = {}
        try:
            for score, cid in self.store.bm25.search(query_clean, top_k=candidate_k):
                bm25_results[cid] = score
        except Exception:
            pass

        # 归一化：BM25 score 无上界，min-max normalize 到 [0, 1]
        bm25_max = max(bm25_results.values()) if bm25_results else 1.0
        if bm25_max <= 0:
            bm25_max = 1.0

        # 合并打分（weighted sum）
        all_ids = set(embed_results.keys()) | set(bm25_results.keys())
        scored: List[Tuple[float, str]] = []
        for cid in all_ids:
            emb_s = embed_results.get(cid, 0.0)
            bm_s = bm25_results.get(cid, 0.0) / bm25_max
            combined = self.embedding_weight * emb_s + self.bm25_weight * bm_s
            scored.append((combined, cid))
        scored.sort(key=lambda x: -x[0])

        out = []
        for score, cid in scored[:top_k]:
            if score < min_score:
                continue
            chunk = self.store._chunks_by_id.get(cid, {})
            out.append({
                "chunk_id": cid,
                "source_id": chunk.get("source_id", ""),
                "arxiv_id": chunk.get("arxiv_id") or chunk.get("source_id", ""),
                # 向后兼容：paper_title / paper_link / paper_published 是对外字段
                # chunk 里实际字段名是 title / pdf_link / published（来自 paper dict）
                "paper_title": chunk.get("paper_title") or chunk.get("title", ""),
                "paper_link": chunk.get("paper_link") or chunk.get("pdf_link", "") or chunk.get("link", ""),
                "paper_published": chunk.get("paper_published") or chunk.get("published", ""),
                "text": chunk.get("text", ""),
                "start": chunk.get("start", 0),
                "end": chunk.get("end", 0),
                "score": float(score),
                "_score_embed": embed_results.get(cid, 0.0),
                "_score_bm25": bm25_results.get(cid, 0.0),
            })
        return out

    def retrieve_for_state(
        self,
        state: Dict[str, Any],
        *,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """从 state 自动构造 query 并检索。

        query = state.goal + 最后 3 条 user message
        """
        top_k = top_k or self.default_top_k
        query = self._build_query_from_state(state)
        if not query:
            return []
        return self.retrieve(query, top_k=top_k, min_score=min_score)

    def format_context(
        self,
        chunks: List[Dict[str, Any]],
        *,
        max_chars: int = 2000,
    ) -> str:
        """把 chunks 拼成 prompt 友好的字符串。

        格式：
            [1] <paper_title> (<arxiv_id>) — score=0.83
            <text>

            [2] ...

        超过 max_chars 时截断 + 标注 "...(truncated, N more chunks omitted)"
        """
        if not chunks:
            return ""
        lines: List[str] = []
        total = 0
        shown = 0
        for i, ch in enumerate(chunks, start=1):
            score = ch.get("score", 0.0)
            title = ch.get("paper_title", "")
            arxiv_id = ch.get("arxiv_id", "")
            text = ch.get("text", "").strip()
            block = f"[{i}] {title} ({arxiv_id}) — score={score:.3f}\n{text}"
            if total + len(block) + 2 > max_chars:
                lines.append(f"...({len(chunks) - shown} more chunks omitted)")
                break
            lines.append(block)
            total += len(block) + 2
            shown += 1
        return "\n\n".join(lines)

    # === 内部 ===

    def _build_query_from_state(self, state: Dict[str, Any]) -> str:
        parts: List[str] = []
        goal = state.get("goal", "")
        if goal:
            parts.append(str(goal))
        # 最后 3 条 user message
        msgs = state.get("messages", []) or []
        user_msgs = []
        for m in reversed(msgs):
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(m, dict) and role in ("user", "human"):
                if content:
                    user_msgs.append(str(content))
                if len(user_msgs) >= 3:
                    break
        for u in reversed(user_msgs):
            parts.append(u)
        return " | ".join(parts)
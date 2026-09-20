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

    v5-3：混合排序 = embedding cosine + BM25 加权（"weighted" strategy）。
    v5-4：query embedding 走 AsyncEmbedder 缓存。
    v6-1：可选 RRF（Reciprocal Rank Fusion）替换加权，跨 query 分数更鲁棒。
    v6-2：bge-base-zh embedder（可选，默认仍 hash 零依赖）。
    v6-3：可选 cross-encoder rerank（默认 off，env HARNESS_RAG_RERANK=on 启用）。
    v6-5：可选 metadata filter（按 topic/year/source 过滤）。

    fusion_strategy:
        - "rrf"（默认）：Reciprocal Rank Fusion（推荐）
        - "weighted"：保留 v5 加权（向后兼容）

    权重：embedding_weight + bm25_weight = 1.0（仅 weighted 用）
    """

    def __init__(
        self,
        store,
        *,
        default_top_k: int = 5,
        embedding_weight: float = 0.7,
        bm25_weight: float = 0.3,
        async_embedder=None,
        fusion_strategy: str = "rrf",
        rrf_k: int = 60,
        reranker=None,
    ):
        self.store = store
        self.default_top_k = default_top_k
        self.embedding_weight = embedding_weight
        self.bm25_weight = bm25_weight
        self.async_embedder = async_embedder  # v5-4：None → 用 sync encode
        self.fusion_strategy = fusion_strategy  # v6-1：rrf | weighted
        self.rrf_k = rrf_k
        self.reranker = reranker  # v6-3：None → 跳过 rerank

        if fusion_strategy not in ("rrf", "weighted"):
            raise ValueError(f"fusion_strategy must be 'rrf' or 'weighted', got {fusion_strategy}")

    # === 主入口 ===

    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """按 query 文本检索（embedding + BM25 混合排序，RRF 或 weighted）。

        返回 List[Dict]，每个含 {chunk_id, source_id, text, arxiv_id, paper_title, paper_link, paper_published, score, _score_embed, _score_bm25}。
        """
        if not query or not query.strip():
            return []
        top_k = top_k or self.default_top_k
        if self.store.count() == 0:
            return []

        query_clean = query.strip()
        # 多取一些候选避免漏召回（rerank 时再多取 5x）
        base_k = max(top_k * 5, 50) if self.reranker else max(top_k * 3, 20)
        candidate_k = base_k

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

        # 融合
        if self.fusion_strategy == "rrf":
            from harness.rag.fusion import rrf_fuse
            scored = rrf_fuse(
                embed_results, bm25_results,
                k_const=self.rrf_k,
                w_embed=1.0, w_bm25=1.0,
            )
        else:
            # 兼容旧 weighted
            bm25_max = max(bm25_results.values()) if bm25_results else 1.0
            if bm25_max <= 0:
                bm25_max = 1.0
            all_ids = set(embed_results.keys()) | set(bm25_results.keys())
            scored = []
            for cid in all_ids:
                emb_s = embed_results.get(cid, 0.0)
                bm_s = bm25_results.get(cid, 0.0) / bm25_max
                combined = self.embedding_weight * emb_s + self.bm25_weight * bm_s
                scored.append((combined, cid))
            scored.sort(key=lambda x: -x[0])

        # v6-3：可选 cross-encoder rerank
        if self.reranker is not None:
            try:
                # 先把 (score, cid) 转成 chunk dict，再 rerank
                chunk_dicts = [self._chunk_to_dict(s, cid) for s, cid in scored]
                chunk_dicts = [c for c in chunk_dicts if c]  # 过滤 None
                if chunk_dicts:
                    reranked = self.reranker.rerank(query_clean, chunk_dicts, top_k=top_k)
                    # 给重排后的 chunk 打 rerank_score 标记
                    for i, c in enumerate(reranked):
                        c["_rerank_rank"] = i + 1
                    return reranked
            except Exception:
                # rerank 失败降级到原排序
                pass

        out = []
        for score, cid in scored[:top_k]:
            if score < min_score:
                continue
            chunk_dict = self._chunk_to_dict(score, cid)
            if chunk_dict:
                out.append(chunk_dict)
        return out

    def _chunk_to_dict(self, score: float, cid: str) -> Optional[Dict[str, Any]]:
        """从 chunk_id 构造对外 dict（含 paper meta）。"""
        chunk = self.store._chunks_by_id.get(cid)
        if chunk is None:
            return None
        return {
            "chunk_id": cid,
            "source_id": chunk.get("source_id", ""),
            "arxiv_id": chunk.get("arxiv_id") or chunk.get("source_id", ""),
            "paper_title": chunk.get("paper_title") or chunk.get("title", ""),
            "paper_link": chunk.get("paper_link") or chunk.get("pdf_link", "") or chunk.get("link", ""),
            "paper_published": chunk.get("paper_published") or chunk.get("published", ""),
            "text": chunk.get("text", ""),
            "start": chunk.get("start", 0),
            "end": chunk.get("end", 0),
            "score": float(score),
            "_score_embed": 0.0,  # 注入需要原始 embed_results（rrf 模式没保存）
            "_score_bm25": 0.0,
        }

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
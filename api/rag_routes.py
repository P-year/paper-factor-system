"""
api/rag_routes.py - RAG 检索 API

POST /rag/search   — 语义检索
GET  /rag/stats    — 索引状态
POST /rag/reindex  — 重建索引
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/rag", tags=["rag"])


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.0
    filter: Optional[Dict[str, Any]] = None


@router.post("/search")
async def rag_search(req: RAGSearchRequest):
    """语义检索论文 chunks。"""
    t0 = time.time()
    try:
        from harness.rag.store import get_default_store
        from harness.rag.retriever import PaperRetriever
        from harness.rag.async_embed import get_async_embedder

        store = get_default_store()
        if store.count() == 0:
            return {"results": [], "took_ms": 0, "warning": "no documents indexed"}

        ae = get_async_embedder(embedder=store.embedder)
        retriever = PaperRetriever(store, async_embedder=ae)
        chunks = retriever.retrieve(req.query, top_k=req.top_k, min_score=req.min_score, filter=req.filter)

        results = [
            {
                "chunk_id": c.get("chunk_id", ""),
                "source_id": c.get("source_id", ""),
                "arxiv_id": c.get("arxiv_id", ""),
                "paper_title": c.get("paper_title", ""),
                "paper_link": c.get("paper_link", ""),
                "score": round(c.get("score", 0.0), 4),
                "text": c.get("text", "")[:500],
            }
            for c in chunks
        ]
        took_ms = int((time.time() - t0) * 1000)
        return {"results": results, "took_ms": took_ms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed: {e}")


@router.get("/stats")
async def rag_stats():
    try:
        from harness.rag.store import get_default_store
        store = get_default_store()
        m = store.manifest()
        return {
            "chunk_count": store.count(),
            "source_count": len(store.source_ids()),
            "model": m.get("model_name", "unknown"),
            "dim": m.get("dim", 0),
            "reranker": m.get("reranker", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reindex")
async def rag_reindex():
    try:
        from harness.rag.store import get_default_store
        store = get_default_store()
        n = store.reindex_incremental()
        store.save()
        return {"reindexed": n, "total": store.count()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
harness/rag - 本地论文 RAG 检索模块

子模块：
- chunker: 段落 + 滑动窗口切分
- embedder: EmbeddingBackend (Protocol + HashBackend + SentenceTransformerBackend)
- index: FAISSIndex 封装
- store: PaperRAGStore 持久化
- retriever: PaperRetriever 主入口
- prompts: SIMILAR_PAPERS_CONTEXT_PROMPT

公共 API：
    Chunker, FAISSIndex, PaperRAGStore, PaperRetriever
    HashBackend, SentenceTransformerBackend, EmbeddingBackend
    get_default_store()
"""
from harness.rag.chunker import Chunker, _safe_id
from harness.rag.embedder import (
    EmbeddingBackend,
    HashBackend,
    SentenceTransformerBackend,
    get_default_backend,
)
from harness.rag.index import FAISSIndex
from harness.rag.store import PaperRAGStore, get_default_store
from harness.rag.retriever import PaperRetriever
from harness.rag.prompts import SIMILAR_PAPERS_CONTEXT_PROMPT

__all__ = [
    "Chunker",
    "_safe_id",
    "EmbeddingBackend",
    "HashBackend",
    "SentenceTransformerBackend",
    "get_default_backend",
    "FAISSIndex",
    "PaperRAGStore",
    "PaperRetriever",
    "get_default_store",
    "SIMILAR_PAPERS_CONTEXT_PROMPT",
]
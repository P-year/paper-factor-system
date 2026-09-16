"""
harness/rag/store.py - PaperRAGStore（组合 chunker + embedder + index）

设计：
- 持久化到 memory/rag/
  - index.faiss + index.faiss.ids.json（FAISS + id 映射）
  - chunks.jsonl（每行一个 chunk）
  - manifest.json（model_name / dim / chunk_count / version）
- 增量索引去重（chunk_id 已存在跳过）
- 自动增量从 data/papers/*.json 索引

用法：
    from harness.rag import PaperRAGStore
    store = PaperRAGStore()
    n = store.index_paper_dir(PAPER_DIR)
    store.save()
    results = store.search(query_vec, top_k=5)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from harness.rag.chunker import Chunker, _safe_id
from harness.rag.embedder import (
    EmbeddingBackend,
    HashBackend,
    SentenceTransformerBackend,
    _EmbedderUnavailable,
)
from harness.rag.index import FAISSIndex
from harness.paths import RAG_DIR, PAPER_DIR


_MANIFEST_VERSION = 1


class PaperRAGStore:
    """本地论文 RAG 持久化存储。"""

    def __init__(
        self,
        *,
        rag_dir: Optional[Path] = None,
        embedder: Optional[EmbeddingBackend] = None,
        chunker: Optional[Chunker] = None,
    ):
        self.rag_dir = Path(rag_dir) if rag_dir else RAG_DIR
        self.rag_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashBackend()  # 默认零依赖，后续可换
        self.chunker = chunker or Chunker()
        self.index = FAISSIndex(dim=self.embedder.dim, metric="cosine")
        self._chunks_by_id: Dict[str, Dict[str, Any]] = {}
        self._source_ids: set = set()  # 已索引的 source_id 集合

        # v5-3：BM25 关键词索引（与 embedding cosine 混合）
        from harness.rag.bm25 import BM25Index
        self.bm25 = BM25Index()

        # 文件路径
        self._index_path = self.rag_dir / "index.faiss"
        self._chunks_path = self.rag_dir / "chunks.jsonl"
        self._bm25_path = self.rag_dir / "bm25.json"
        self._manifest_path = self.rag_dir / "manifest.json"

        # 如果存在则加载
        if self.exists():
            try:
                self.load()
            except Exception:
                # 加载失败清空
                self._clear_chunks()

    # === 文件路径 ===

    @property
    def chunks_path(self) -> Path:
        return self._chunks_path

    @property
    def index_path(self) -> Path:
        return self._index_path

    # === 状态 ===

    def exists(self) -> bool:
        """manifest.json 是否存在。"""
        return self._manifest_path.exists()

    def count(self) -> int:
        """已索引的 chunk 数。"""
        return len(self._chunks_by_id)

    def source_ids(self) -> List[str]:
        return sorted(self._source_ids)

    def manifest(self) -> Dict[str, Any]:
        """读 manifest.json（如不存在返回当前 state 的快照）。"""
        if self._manifest_path.exists():
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        return self._build_manifest()

    # === 索引 ===

    def index_paper(self, paper: Dict[str, Any]) -> int:
        """索引单篇论文。返回新增 chunk 数（去重后）。"""
        source_id = _safe_id(paper)
        chunks = self.chunker.split_paper(paper)
        return self._add_chunks_internal(chunks, source_id, paper_meta=paper)

    def index_papers(self, papers: List[Dict[str, Any]]) -> int:
        """批量索引。返回总新增数。"""
        total = 0
        for p in papers:
            total += self.index_paper(p)
        return total

    def index_json_file(self, json_path: Path) -> int:
        """索引单个 JSON 文件。"""
        from harness.eval.paper_loader import normalize_papers
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        papers = data.get("papers", data) if isinstance(data, dict) else data
        if not papers:
            return 0
        papers = normalize_papers(papers) if isinstance(papers[0], dict) else papers
        return self.index_papers(papers)

    def index_paper_dir(
        self,
        paper_dir: Optional[Path] = None,
        *,
        only_ids: Optional[List[str]] = None,
    ) -> int:
        """扫 paper_dir 全部 papers_*.json 索引。

        only_ids: 只索引 arxiv_id 在此列表中的论文（None = 全索引）。
        """
        paper_dir = Path(paper_dir) if paper_dir else PAPER_DIR
        if not paper_dir.exists():
            return 0
        total = 0
        only_set = set(only_ids) if only_ids else None
        for fp in sorted(paper_dir.glob("papers_*.json"), reverse=True):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            papers = data.get("papers", data) if isinstance(data, dict) else data
            if not papers:
                continue
            # 过滤
            if only_set is not None:
                papers = [p for p in papers if (p.get("arxiv_id") in only_set
                                                  or _safe_id(p) in only_set)]
            if not papers:
                continue
            try:
                from harness.eval.paper_loader import normalize_papers
                papers = normalize_papers(papers)
            except Exception:
                pass
            total += self.index_papers(papers)
        return total

    def reindex_incremental(
        self,
        paper_dir: Optional[Path] = None,
    ) -> int:
        """增量索引：仅索引 source_id 不在已有 manifest 的论文。"""
        existing = set(self.source_ids())
        paper_dir = Path(paper_dir) if paper_dir else PAPER_DIR
        if not paper_dir.exists():
            return 0
        total = 0
        for fp in sorted(paper_dir.glob("papers_*.json"), reverse=True):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            papers = data.get("papers", data) if isinstance(data, dict) else data
            if not papers:
                continue
            new_papers = [p for p in papers if _safe_id(p) not in existing]
            try:
                from harness.eval.paper_loader import normalize_papers
                new_papers = normalize_papers(new_papers)
            except Exception:
                pass
            total += self.index_papers(new_papers)
        return total

    def rebuild(self) -> int:
        """全量重建：清空 + 从 paper_dir 重新索引。"""
        self._clear_chunks()
        return self.index_paper_dir()

    # === 检索 ===

    def search(self, query_vec: np.ndarray, *, top_k: int = 5) -> List[Tuple[float, str]]:
        """低阶检索。"""
        return self.index.search(query_vec, top_k=top_k)

    # === 持久化 ===

    def save(self) -> None:
        """保存索引 + chunks + manifest。"""
        self.rag_dir.mkdir(parents=True, exist_ok=True)
        # 索引
        self.index.save(self._index_path)
        # chunks
        with self._chunks_path.open("w", encoding="utf-8") as f:
            for cid, chunk in self._chunks_by_id.items():
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        # v5-3：BM25
        self._bm25_path.write_text(
            json.dumps(self.bm25.to_dict(), ensure_ascii=False),
            encoding="utf-8",
        )
        # manifest
        self._manifest_path.write_text(
            json.dumps(self._build_manifest(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        """从磁盘加载索引 + chunks + manifest。

        维度不匹配（embedder 换了）时清空 chunks + BM25，保留磁盘文件
        等调用方 rebuild / 增量索引。
        """
        # chunks
        self._chunks_by_id = {}
        dim_mismatch = False
        if self._chunks_path.exists():
            with self._chunks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ch = json.loads(line)
                        self._chunks_by_id[ch["chunk_id"]] = ch
                    except json.JSONDecodeError:
                        continue
        # 索引
        if self._index_path.exists():
            try:
                self.index.load(self._index_path)
                if self.index.dim != self.embedder.dim:
                    # 维度不匹配：清空内存里的 chunks（磁盘保留供备份/手动迁移）
                    self._chunks_by_id = {}
                    self.index = FAISSIndex(dim=self.embedder.dim, metric="cosine")
                    dim_mismatch = True
            except Exception:
                # 加载失败：当作空 store
                self.index = FAISSIndex(dim=self.embedder.dim, metric="cosine")
                dim_mismatch = True
                self._chunks_by_id = {}
        # 重建 source_ids 集合
        self._source_ids = {ch.get("source_id", "") for ch in self._chunks_by_id.values()}
        self._source_ids.discard("")
        # v5-3：重建 BM25（维度错时不读）
        if dim_mismatch:
            self.bm25 = type(self.bm25)()
        elif self._bm25_path.exists():
            try:
                self.bm25.from_dict(json.loads(self._bm25_path.read_text(encoding="utf-8")))
            except Exception:
                self.bm25 = type(self.bm25)()
                self.bm25.add_chunks(list(self._chunks_by_id.values()))

    # === 内部 ===

    def _add_chunks_internal(self, chunks: List[Dict[str, Any]], source_id: str,
                           *, paper_meta: Optional[Dict[str, Any]] = None) -> int:
        """加 chunks 到 store（去重 + embed + 写 self._chunks_by_id）。

        paper_meta：原始 paper dict（含 arxiv_id / title / link / pdf_link 等），
        把这些字段注入每个 chunk，方便 retrieve 时直接拿到 paper meta。
        """
        if not chunks:
            return 0
        # 去重（已存在 chunk_id 跳过）
        new_chunks = []
        new_chunk_ids = []
        for ch in chunks:
            cid = ch["chunk_id"]
            if cid in self._chunks_by_id:
                continue
            new_chunks.append(ch)
            new_chunk_ids.append(cid)
        if not new_chunks:
            return 0

        # embed
        texts = [c["text"] for c in new_chunks]
        try:
            vecs = self.embedder.encode(texts)
        except _EmbedderUnavailable:
            return 0
        if vecs is None or len(vecs) == 0:
            return 0

        # 加到 FAISS（按 new_chunk_ids 顺序）
        self.index.add(vecs, ids=new_chunk_ids)

        # v5-3：同步加 BM25
        self.bm25.add_chunks(new_chunks)

        # 写 self._chunks_by_id + paper meta
        # 把 paper_meta 的字段注入每个 chunk（除 chunk 自身字段外）
        paper_meta_fields = {}
        if paper_meta:
            for k, v in paper_meta.items():
                if k not in ("chunk_id", "source_id", "text", "start", "end"):
                    paper_meta_fields[k] = v
        for ch, cid in zip(new_chunks, new_chunk_ids):
            stored = dict(ch)
            stored.update(paper_meta_fields)
            self._chunks_by_id[cid] = stored

        self._source_ids.add(source_id)
        return len(new_chunks)

    def _build_manifest(self) -> Dict[str, Any]:
        return {
            "version": _MANIFEST_VERSION,
            "model_name": self.embedder.model_name,
            "dim": self.embedder.dim,
            "chunk_count": len(self._chunks_by_id),
            "source_count": len(self._source_ids),
            "metric": "cosine",
            "updated_at": datetime.now().isoformat(),
            "model_unavailable": isinstance(self.embedder, HashBackend),
        }

    def _clear_chunks(self):
        self._chunks_by_id = {}
        self._source_ids = set()
        self.index = FAISSIndex(dim=self.embedder.dim, metric="cosine")
        # 不删文件（rebuild 会覆盖；调用方负责）


def get_default_store(
    *,
    rag_dir: Optional[Path] = None,
    prefer: str = "auto",
) -> PaperRAGStore:
    """构造默认 store。

    prefer:
        - "auto"：尝试 SentenceTransformerBackend，失败降级 HashBackend
        - "hash"：强制零依赖
        - "sentence_transformer"：强制 ST（失败抛）
    """
    if prefer == "hash":
        embedder = HashBackend()
    elif prefer == "sentence_transformer":
        embedder = SentenceTransformerBackend()
    else:
        try:
            embedder = SentenceTransformerBackend()
            embedder._load()
        except _EmbedderUnavailable:
            embedder = HashBackend()
    return PaperRAGStore(rag_dir=rag_dir, embedder=embedder)
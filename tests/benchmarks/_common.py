"""
tests/benchmarks/_common.py - benchmark 脚本公共函数

抽出来减少 4 个 benchmark 脚本的重复代码。
"""
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUERIES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "rag_eval" / "queries.jsonl"
PDF_DIR = PROJECT_ROOT / "tests" / "fixtures" / "pdfs"
RESULTS_DIR = PROJECT_ROOT / "tests" / "benchmarks" / "results"


def _build_embedder_from_env():
    """从 env 构造 embedder（hash / sentence_transformer / auto）。

    Returns:
        (embedder, label) — label 用于结果文件名
    """
    prefer = os.getenv("HARNESS_RAG_PREFER", "hash")
    if prefer == "hash":
        from harness.rag.embedder import HashBackend
        return HashBackend(), "hash"
    elif prefer == "sentence_transformer":
        from harness.rag.embedder import SentenceTransformerBackend
        model = os.getenv("HARNESS_RAG_MODEL", "BAAI/bge-base-zh-v1.5")
        return SentenceTransformerBackend(model_name=model), model.split("/")[-1]
    elif prefer == "auto":
        from harness.rag.embedder import (
            HashBackend, SentenceTransformerBackend, _EmbedderUnavailable,
        )
        try:
            emb = SentenceTransformerBackend()
            emb._load()
            return emb, emb.model_name.split("/")[-1]
        except _EmbedderUnavailable:
            return HashBackend(), "hash"
    else:
        raise ValueError(f"unknown prefer: {prefer}")


def _build_store_and_retriever(
    embedder,
    *,
    fusion_strategy: str = "rrf",
    reranker=None,
    paper_dir: Optional[Path] = None,
):
    """构造 PaperRAGStore + PaperRetriever（in tempfile dir）。

    Returns:
        (store, retriever, tmpdir_path)
    """
    from harness.rag.store import PaperRAGStore
    from harness.rag.pdf_loader import pdf_dir_to_papers
    from harness.rag.retriever import PaperRetriever

    tmp = Path(tempfile.mkdtemp(prefix="rag_bench_"))
    store = PaperRAGStore(rag_dir=tmp, embedder=embedder)
    papers = pdf_dir_to_papers(paper_dir or PDF_DIR, max_pages=3)
    store.index_papers(papers)
    retriever = PaperRetriever(
        store, fusion_strategy=fusion_strategy, reranker=reranker,
    )
    return store, retriever, tmp


def _run_evaluation(retriever, k: int = 5):
    """跑 ground-truth eval，返回 report dict。"""
    from harness.rag.eval import run_evaluation
    return run_evaluation(retriever, QUERIES_PATH, k=k)


def _save_result(report: Dict[str, Any], label: str, suffix: str = ""):
    """把 report（去 per_query）写到 results/，filename sanitized。"""
    import json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace("/", "_").replace(":", "_") + (f"_{suffix}" if suffix else "")
    out_file = RESULTS_DIR / f"{safe_label}.json"
    summary = {k: v for k, v in report.items() if k != "per_query"}
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_file


def _print_header(embed_label: str, rerank_label: str = "none"):
    print(f"Embedder: {embed_label}")
    print(f"Reranker: {rerank_label}")
    print()


def _check_fixtures_or_die():
    """CI 环境没 PDF fixture 会 fail；统一报错。"""
    if not PDF_DIR.exists() or not list(PDF_DIR.glob("*.pdf")):
        print(f"ERROR: PDF fixtures not found at {PDF_DIR}")
        sys.exit(1)
    if not QUERIES_PATH.exists():
        print(f"ERROR: queries.jsonl not found at {QUERIES_PATH}")
        sys.exit(1)
"""
tests/benchmarks/benchmark_rerank.py - Cross-encoder rerank benchmark

跑 100 个 ground-truth query，对比：
1. baseline：embedding + RRF（不 rerank）
2. + cross-encoder rerank（bge-reranker-base / mock）

环境要求：tests/fixtures/pdfs/ 存在 10 篇真实 PDF

用法：
    # Mock rerank（无需下载模型）
    HARNESS_RAG_PREFER=sentence_transformer \\
    HARNESS_RAG_MODEL=BAAI/bge-small-zh-v1.5 \\
    HARNESS_RAG_RERANK=mock \\
    python tests/benchmarks/benchmark_rerank.py

    # 真实 cross-encoder（首次需下载 ~200MB）
    HARNESS_RAG_RERANK=on \\
    python tests/benchmarks/benchmark_rerank.py
"""
import os
import sys
import json
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUERIES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "rag_eval" / "queries.jsonl"
PDF_DIR = PROJECT_ROOT / "tests" / "fixtures" / "pdfs"


def _build_mock_reranker():
    """Mock rerank：随机重排（用于测试逻辑不依赖真实模型）。"""
    import random
    from harness.rag.rerank import Reranker

    class MockReranker(Reranker):
        @property
        def model_name(self):
            return "mock-random"

        def rerank(self, query, chunks, *, top_k=None):
            random.seed(hash(query) % 2**32)
            shuffled = list(chunks)
            random.shuffle(shuffled)
            return shuffled[:top_k]

    return MockReranker()


def _build_pass_through_reranker():
    """Pass-through：保留 RRF 排序（用于 baseline 对照）。"""
    from harness.rag.rerank import _NoOpReranker
    return _NoOpReranker()


def main():
    if not PDF_DIR.exists() or not list(PDF_DIR.glob("*.pdf")):
        print(f"ERROR: PDF fixtures not found at {PDF_DIR}")
        sys.exit(1)
    if not QUERIES_PATH.exists():
        print(f"ERROR: queries.jsonl not found at {QUERIES_PATH}")
        sys.exit(1)

    # embedder
    prefer = os.getenv("HARNESS_RAG_PREFER", "hash")
    if prefer == "hash":
        from harness.rag.embedder import HashBackend
        embedder = HashBackend()
    else:
        model_name = os.getenv("HARNESS_RAG_MODEL", "BAAI/bge-base-zh-v1.5")
        from harness.rag.embedder import SentenceTransformerBackend
        embedder = SentenceTransformerBackend(model_name=model_name)

    # reranker
    rerank_mode = os.getenv("HARNESS_RAG_RERANK", "off").lower()
    if rerank_mode == "on":
        try:
            from harness.rag.rerank import get_default_reranker
            reranker = get_default_reranker(enabled=True)
            if reranker is None:
                print("WARN: get_default_reranker() returned None; falling back to noop")
                reranker = _build_pass_through_reranker()
        except Exception as e:
            print(f"WARN: failed to init cross-encoder: {e}; falling back to noop")
            reranker = _build_pass_through_reranker()
    elif rerank_mode == "mock":
        reranker = _build_mock_reranker()
    elif rerank_mode == "noop":
        reranker = _build_pass_through_reranker()
    else:
        reranker = None  # baseline

    # 跑 benchmark
    from harness.rag.store import PaperRAGStore
    from harness.rag.pdf_loader import pdf_dir_to_papers
    from harness.rag.retriever import PaperRetriever
    from harness.rag.eval import run_evaluation, format_report

    rerank_label = "none" if reranker is None else (
        "noop" if hasattr(reranker, 'model_name') and reranker.model_name == "noop"
        else getattr(reranker, 'model_name', 'unknown')
    )
    embed_label = embedder.model_name.split("/")[-1] if "/" in embedder.model_name else embedder.model_name
    print(f"Embedder: {embed_label}")
    print(f"Reranker: {rerank_label}")
    print()

    with tempfile.TemporaryDirectory() as td:
        papers = pdf_dir_to_papers(PDF_DIR, max_pages=3)
        store = PaperRAGStore(rag_dir=Path(td) / "bench", embedder=embedder)
        store.index_papers(papers)
        retriever = PaperRetriever(store, fusion_strategy="rrf", reranker=reranker)

        t0 = time.time()
        report = run_evaluation(retriever, QUERIES_PATH, k=5)
        elapsed = time.time() - t0

        print(format_report(report))
        print(f"Eval time: {elapsed:.1f}s")

        # 保存结果
        results_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{embed_label}_rerank-{rerank_label}.json"
        out_file = results_dir / out_name
        summary = {k: v for k, v in report.items() if k != "per_query"}
        summary["elapsed_s"] = elapsed
        summary["embedder"] = embed_label
        summary["reranker"] = rerank_label
        out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()
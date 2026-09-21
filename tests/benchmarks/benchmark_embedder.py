"""
tests/benchmarks/benchmark_embedder.py - Embedder benchmark 回归脚本

跑在 30 个 ground-truth query 上，比较不同 embedder backend 的检索质量。

用法：
    HARNESS_RAG_PREFER=hash python tests/benchmarks/benchmark_embedder.py
    HARNESS_RAG_PREFER=sentence_transformer python tests/benchmarks/benchmark_embedder.py
    # 输出 baseline 文件供后续 phase 对比

环境要求：tests/fixtures/pdfs/ 存在 10 篇真实 PDF
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


def main():
    if not PDF_DIR.exists() or not list(PDF_DIR.glob("*.pdf")):
        print(f"ERROR: PDF fixtures not found at {PDF_DIR}")
        sys.exit(1)
    if not QUERIES_PATH.exists():
        print(f"ERROR: queries.jsonl not found at {QUERIES_PATH}")
        sys.exit(1)

    from harness.rag.store import PaperRAGStore
    from harness.rag.embedder import HashBackend, SentenceTransformerBackend, _EmbedderUnavailable
    from harness.rag.pdf_loader import pdf_dir_to_papers
    from harness.rag.retriever import PaperRetriever
    from harness.rag.eval import run_evaluation, format_report

    print(f"PDF dir: {PDF_DIR}")
    print(f"Queries: {QUERIES_PATH}")
    print()

    # 按 prefer 决定 embedder
    prefer = os.getenv("HARNESS_RAG_PREFER", "hash")
    if prefer == "hash":
        embedder = HashBackend()
        label = "hash"
    elif prefer == "sentence_transformer":
        model_name = os.getenv("HARNESS_RAG_MODEL", "BAAI/bge-base-zh-v1.5")
        embedder = SentenceTransformerBackend(model_name=model_name)
        label = f"st:{model_name.split('/')[-1]}"
    else:
        # auto：尝试 ST，失败降级 hash
        try:
            embedder = SentenceTransformerBackend()
            embedder._load()
            label = f"st:{embedder.model_name.split('/')[-1]}"
        except _EmbedderUnavailable:
            embedder = HashBackend()
            label = "hash-fallback"

    print(f"Embedder: {label}")
    print()

    with tempfile.TemporaryDirectory() as td:
        papers = pdf_dir_to_papers(PDF_DIR, max_pages=3)
        store = PaperRAGStore(rag_dir=Path(td) / "bench", embedder=embedder)
        store.index_papers(papers)

        retriever = PaperRetriever(store, fusion_strategy="rrf")

        t0 = time.time()
        report = run_evaluation(retriever, QUERIES_PATH, k=5)
        elapsed = time.time() - t0

        print(format_report(report))
        print(f"Eval time: {elapsed:.1f}s")

        # 写结果到 tests/benchmarks/results/
        results_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        # sanitize filename（Windows 不允许 `:` 等字符）
        safe_label = label.replace(":", "_").replace("/", "_")
        out_file = results_dir / f"{safe_label}.json"
        summary = {k: v for k, v in report.items() if k != "per_query"}
        summary["elapsed_s"] = elapsed
        summary["embedder"] = label
        out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()
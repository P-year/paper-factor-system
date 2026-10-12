"""
tests/benchmarks/benchmark_base.py - bge-base-zh benchmark

跑 100 个 ground-truth query，embedder 固定 BAAI/bge-base-zh-v1.5。
专门测 v6-2 embedder 升级路径。

用法（CI 自动跑）：
    HARNESS_RAG_PREFER=sentence_transformer \\
    HARNESS_RAG_MODEL=BAAI/bge-base-zh-v1.5 \\
    python tests/benchmarks/benchmark_base.py
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

    # bge-base-zh 固定
    model_name = os.getenv("HARNESS_RAG_MODEL", "BAAI/bge-base-zh-v1.5")
    from harness.rag.embedder import SentenceTransformerBackend
    embedder = SentenceTransformerBackend(model_name=model_name)

    from harness.rag.store import PaperRAGStore
    from harness.rag.pdf_loader import pdf_dir_to_papers
    from harness.rag.retriever import PaperRetriever
    from harness.rag.eval import run_evaluation, format_report

    print(f"Embedder: {model_name}")
    print()

    with tempfile.TemporaryDirectory() as td:
        papers = pdf_dir_to_papers(PDF_DIR, max_pages=3)
        store = PaperRAGStore(rag_dir=Path(td) / "bench", embedder=embedder)
        store.index_papers(papers)
        retriever = PaperRetriever(store, fusion_strategy="rrf", reranker=None)

        t0 = time.time()
        report = run_evaluation(retriever, QUERIES_PATH, k=5)
        elapsed = time.time() - t0

        print(format_report(report))
        print(f"Eval time: {elapsed:.1f}s")

        # 保存
        results_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        out_name = "bge-base-zh-v1.5.json"
        out_file = results_dir / out_name
        summary = {k: v for k, v in report.items() if k != "per_query"}
        summary["elapsed_s"] = elapsed
        summary["embedder"] = "bge-base-zh-v1.5"
        out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()
"""
tests/benchmarks/benchmark_base.py - bge-base-zh benchmark

固定 embedder=BAAI/bge-base-zh-v1.5（768d, 400MB）。
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    _build_store_and_retriever,
    _check_fixtures_or_die,
    _run_evaluation,
    _save_result,
    _print_header,
)

from harness.rag.embedder import SentenceTransformerBackend


def main():
    _check_fixtures_or_die()
    model_name = os.getenv("HARNESS_RAG_MODEL", "BAAI/bge-base-zh-v1.5")
    embedder = SentenceTransformerBackend(model_name=model_name)
    _print_header("bge-base-zh-v1.5", "none")

    store, retriever, _ = _build_store_and_retriever(embedder, fusion_strategy="rrf")

    t0 = time.time()
    report = _run_evaluation(retriever, k=5)
    elapsed = time.time() - t0

    from harness.rag.eval import format_report
    print(format_report(report))
    print(f"Eval time: {elapsed:.1f}s")
    out = _save_result(report, "bge-base-zh-v1.5")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
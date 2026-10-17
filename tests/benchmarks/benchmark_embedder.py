"""
tests/benchmarks/benchmark_embedder.py - 跑 embedder 检索 benchmark

用法：
    HARNESS_RAG_PREFER=hash python tests/benchmarks/benchmark_embedder.py
    HARNESS_RAG_PREFER=sentence_transformer HARNESS_RAG_MODEL=BAAI/bge-base-zh-v1.5 \\
      python tests/benchmarks/benchmark_embedder.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    _build_embedder_from_env,
    _build_store_and_retriever,
    _check_fixtures_or_die,
    _run_evaluation,
    _save_result,
    _print_header,
)


def main():
    _check_fixtures_or_die()
    embedder, label = _build_embedder_from_env()
    _print_header(label, "none")

    store, retriever, _ = _build_store_and_retriever(embedder, fusion_strategy="rrf")

    t0 = time.time()
    report = _run_evaluation(retriever, k=5)
    elapsed = time.time() - t0

    from harness.rag.eval import format_report
    print(format_report(report))
    print(f"Eval time: {elapsed:.1f}s")
    out = _save_result(report, label)
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
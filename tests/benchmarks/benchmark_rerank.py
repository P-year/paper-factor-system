"""
tests/benchmarks/benchmark_rerank.py - embedder × rerank 组合 benchmark

HARNESS_RAG_RERANK env: off | noop | mock-random | on
"""
import os
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


def _build_reranker(mode: str):
    if mode == "off":
        return None, "none"
    if mode == "noop":
        from harness.rag.rerank import _NoOpReranker
        return _NoOpReranker(), "noop"
    if mode == "mock":
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

        return MockReranker(), "mock-random"
    if mode == "on":
        try:
            from harness.rag.rerank import get_default_reranker
            r = get_default_reranker(enabled=True)
            if r is None:
                return None, "noop"
            return r, r.model_name.split("/")[-1] if "/" in r.model_name else r.model_name
        except Exception as e:
            print(f"WARN: failed to init cross-encoder: {e}; falling back to noop")
            from harness.rag.rerank import _NoOpReranker
            return _NoOpReranker(), "noop"
    raise ValueError(f"unknown rerank mode: {mode}")


def main():
    _check_fixtures_or_die()
    embedder, embed_label = _build_embedder_from_env()
    rerank_mode = os.getenv("HARNESS_RAG_RERANK", "off").lower()
    reranker, rerank_label = _build_reranker(rerank_mode)
    _print_header(embed_label, rerank_label)

    store, retriever, _ = _build_store_and_retriever(
        embedder, fusion_strategy="rrf", reranker=reranker,
    )

    t0 = time.time()
    report = _run_evaluation(retriever, k=5)
    elapsed = time.time() - t0

    from harness.rag.eval import format_report
    print(format_report(report))
    print(f"Eval time: {elapsed:.1f}s")
    out = _save_result(report, embed_label, suffix=f"rerank-{rerank_label}")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
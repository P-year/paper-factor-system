"""
tests/benchmarks/benchmark_latency.py - 端到端 latency 测量

测量 query 检索的各阶段耗时：
1. embed（query → vector）
2. FAISS search（vector → top-k chunks）
3. BM25 search（query → top-k chunks）
4. RRF fusion（merge + rank）
5. 切片组装 dict

报告：
- per-stage avg / p50 / p95 / p99 latency
- total per-query latency
- 100 queries 总耗时

用法：
    python tests/benchmarks/benchmark_latency.py [--embedder hash|bge-small|bge-base] [--queries 100]
"""
import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUERIES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "rag_eval" / "queries.jsonl"
PDF_DIR = PROJECT_ROOT / "tests" / "fixtures" / "pdfs"


def _percentile(values: List[float], p: float) -> float:
    """近似 percentile（不依赖 numpy）。"""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * p / 100)
    idx = min(idx, len(sorted_v) - 1)
    return sorted_v[idx]


def _build_embedder(name: str):
    if name == "hash":
        from harness.rag.embedder import HashBackend
        return HashBackend()
    elif name in ("bge-small", "bge-base"):
        model = "BAAI/bge-small-zh-v1.5" if name == "bge-small" else "BAAI/bge-base-zh-v1.5"
        from harness.rag.embedder import SentenceTransformerBackend
        return SentenceTransformerBackend(model_name=model)
    else:
        raise ValueError(f"unknown embedder: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedder", default="hash",
        choices=["hash", "bge-small", "bge-base"],
        help="embedder to benchmark",
    )
    parser.add_argument(
        "--queries", type=int, default=100,
        help="number of queries to run (use max(queries, len(file))",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
    )
    args = parser.parse_args()

    if not PDF_DIR.exists() or not list(PDF_DIR.glob("*.pdf")):
        print(f"ERROR: PDF fixtures not found at {PDF_DIR}")
        sys.exit(1)
    if not QUERIES_PATH.exists():
        print(f"ERROR: queries.jsonl not found at {QUERIES_PATH}")
        sys.exit(1)

    from harness.rag.store import PaperRAGStore
    from harness.rag.pdf_loader import pdf_dir_to_papers
    from harness.rag.retriever import PaperRetriever

    embedder = _build_embedder(args.embedder)
    print(f"Embedder: {args.embedder} (dim={embedder.dim})")

    # Build index
    with tempfile.TemporaryDirectory() as td:
        papers = pdf_dir_to_papers(PDF_DIR, max_pages=3)
        store = PaperRAGStore(rag_dir=Path(td) / "bench", embedder=embedder)
        store.index_papers(papers)
        retriever = PaperRetriever(store, fusion_strategy="rrf")
        print(f"Indexed {store.count()} chunks from {len(store.source_ids())} papers")

        # Load queries
        queries = []
        with QUERIES_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                queries.append(json.loads(line)["query"])
        queries = queries[:args.queries]
        print(f"Queries: {len(queries)}")
        print(f"Top-K: {args.top_k}")
        print()

        # 预热（首次 query 可能加载 lazy embedder）
        retriever.retrieve(queries[0], top_k=args.top_k)

        # Latency measurement
        embed_times: List[float] = []
        total_times: List[float] = []

        for i, q in enumerate(queries):
            t_total = time.perf_counter()
            retriever.retrieve(q, top_k=args.top_k)
            elapsed = (time.perf_counter() - t_total) * 1000  # ms
            total_times.append(elapsed)
            if i % 20 == 0 and i > 0:
                print(f"  progress: {i}/{len(queries)}")

        # 输出
        print()
        print("=" * 50)
        print(f"Latency stats (ms) — n={len(total_times)}")
        print("=" * 50)
        print(f"  avg:  {statistics.mean(total_times):.2f} ms")
        print(f"  p50:  {_percentile(total_times, 50):.2f} ms")
        print(f"  p95:  {_percentile(total_times, 95):.2f} ms")
        print(f"  p99:  {_percentile(total_times, 99):.2f} ms")
        print(f"  max:  {max(total_times):.2f} ms")
        print(f"  min:  {min(total_times):.2f} ms")
        print(f"  total: {sum(total_times):.0f} ms")
        print(f"  throughput: {1000 / statistics.mean(total_times):.1f} q/s")
        print()

        # Save
        results_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        safe_name = args.embedder.replace("-", "_")
        out_file = results_dir / f"latency_{safe_name}.json"
        out_file.write_text(json.dumps({
            "embedder": args.embedder,
            "queries": len(total_times),
            "avg_ms": statistics.mean(total_times),
            "p50_ms": _percentile(total_times, 50),
            "p95_ms": _percentile(total_times, 95),
            "p99_ms": _percentile(total_times, 99),
            "min_ms": min(total_times),
            "max_ms": max(total_times),
            "total_ms": sum(total_times),
            "throughput_qps": 1000 / statistics.mean(total_times),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()
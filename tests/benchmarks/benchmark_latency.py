"""
tests/benchmarks/benchmark_latency.py - 端到端 latency 测量

用法：
    python tests/benchmarks/benchmark_latency.py --embedder hash
    python tests/benchmarks/benchmark_latency.py --embedder bge-small
    python tests/benchmarks/benchmark_latency.py --embedder bge-base
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    _build_store_and_retriever,
    _check_fixtures_or_die,
    RESULTS_DIR,
)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


def _build_named_embedder(name: str):
    if name == "hash":
        from harness.rag.embedder import HashBackend
        return HashBackend()
    if name in ("bge-small", "bge-base"):
        from harness.rag.embedder import SentenceTransformerBackend
        model = "BAAI/bge-small-zh-v1.5" if name == "bge-small" else "BAAI/bge-base-zh-v1.5"
        return SentenceTransformerBackend(model_name=model)
    raise ValueError(f"unknown embedder: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedder", default="hash",
        choices=["hash", "bge-small", "bge-base"],
    )
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    _check_fixtures_or_die()
    embedder = _build_named_embedder(args.embedder)
    print(f"Embedder: {args.embedder} (dim={embedder.dim})")

    store, retriever, _ = _build_store_and_retriever(embedder, paper_dir=None)
    print(f"Indexed {store.count()} chunks from {len(store.source_ids())} papers")

    # Load queries
    from _common import QUERIES_PATH
    queries = []
    with QUERIES_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            import json
            queries.append(json.loads(line)["query"])
    queries = queries[: args.queries]
    print(f"Queries: {len(queries)} | Top-K: {args.top_k}")
    print()

    # 预热
    retriever.retrieve(queries[0], top_k=args.top_k)

    # 测
    total_times: List[float] = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        retriever.retrieve(q, top_k=args.top_k)
        total_times.append((time.perf_counter() - t0) * 1000)
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
    print(f"  max:  {max(total_times):.2f} ms | min: {min(total_times):.2f} ms")
    print(f"  total: {sum(total_times):.0f} ms")
    print(f"  throughput: {1000 / statistics.mean(total_times):.1f} q/s")
    print()

    # 保存
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = args.embedder.replace("-", "_")
    out = RESULTS_DIR / f"latency_{safe}.json"
    out.write_text(json.dumps({
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
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
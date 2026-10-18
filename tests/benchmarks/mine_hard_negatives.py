"""
tests/benchmarks/mine_hard_negatives.py - v9-3：挖 hard negative candidates

用当前 retriever 跑 195 query，标记 top-k 中"召回但 ground-truth 不相关"的
chunk 为 false positive，按 rank × query 写入 hard_negatives_candidates.jsonl，
供人工 review 后扩 ground-truth 的 irrelevant 列表。

用法：
    python tests/benchmarks/mine_hard_negatives.py
    python tests/benchmarks/mine_hard_negatives.py --top-k 10 --embedder bge-base-zh-v1.5

输出：
    tests/benchmarks/results/hard_negatives_candidates.jsonl
    每行：{"query", "chunk_id", "source_id", "paper_title", "rank", "score", "text_preview"}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from harness.paths import RAG_DIR


def _load_queries(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _short(text: str, n: int = 120) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:n] + ("..." if len(text) > n else "")


def _build_retriever(embedder: str, fusion: str = "rrf"):
    from harness.rag.store import get_default_store
    from harness.rag.retriever import PaperRetriever
    from harness.rag.async_embed import get_async_embedder
    store = get_default_store(prefer=embedder)
    if store.count() == 0:
        store.index_paper_dir()
        store.save()
    ae = get_async_embedder(embedder=store.embedder)
    return PaperRetriever(store, async_embedder=ae, fusion_strategy=fusion), store


def mine(
    *,
    queries_path: Path,
    top_k: int,
    embedder: str,
    fusion: str,
    out_path: Path,
) -> Dict[str, Any]:
    queries = _load_queries(queries_path)
    # 只处理有 relevant 的 query（空 list 时没有 false positive 概念）
    queries = [q for q in queries if q.get("relevant")]
    if not queries:
        raise SystemExit("No queries with 'relevant' field found")

    retriever, store = _build_retriever(embedder, fusion)

    candidates: List[Dict[str, Any]] = []
    fp_counts: List[int] = []
    t0 = time.time()
    for qi, q in enumerate(queries):
        qtext = q["query"]
        relevant_set = set(q.get("relevant") or [])
        try:
            chunks = retriever.retrieve(qtext, top_k=top_k, min_score=-1e9)
        except Exception as e:
            print(f"WARN: query #{qi} failed: {e}", file=sys.stderr)
            continue
        n_fp = 0
        for rank, c in enumerate(chunks, start=1):
            sid = c.get("source_id", "")
            # relevant 用的是 arxiv_id 形式或 chunk_id 形式；source_id 通常是 _safe_id(paper)
            # 这里兼容两种
            if sid and (sid in relevant_set or c.get("arxiv_id", "") in relevant_set):
                continue
            n_fp += 1
            candidates.append({
                "query": qtext,
                "relevant_expected": sorted(relevant_set),
                "chunk_id": c.get("chunk_id", ""),
                "source_id": sid,
                "arxiv_id": c.get("arxiv_id", ""),
                "paper_title": c.get("paper_title", ""),
                "rank": rank,
                "score": round(c.get("score", 0.0), 4),
                "text_preview": _short(c.get("text", "")),
            })
        fp_counts.append(n_fp)
        if (qi + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{qi+1}/{len(queries)}] fp={n_fp} elapsed={elapsed:.1f}s")

    # Sort: rank asc → 越靠前的 FP 越有价值（top-1 false positive 最严重）
    candidates.sort(key=lambda c: (c["rank"], c["query"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    avg_fp = (sum(fp_counts) / len(fp_counts)) if fp_counts else 0.0
    print()
    print(f"挖到 {len(candidates)} FP candidates（avg {avg_fp:.2f} FP / query @ k={top_k}）")
    print(f"   分布：rank=1  {sum(1 for c in candidates if c['rank']==1)}  个（最严重）")
    print(f"   分布：rank≤3  {sum(1 for c in candidates if c['rank']<=3)}  个")
    print(f"   分布：rank≤5  {sum(1 for c in candidates if c['rank']<=5)}  个")
    print(f"   耗时 {elapsed:.1f}s")
    print(f"写入 {out_path}")
    return {
        "n_candidates": len(candidates),
        "avg_fp_per_query": round(avg_fp, 3),
        "n_queries": len(queries),
        "top_k": top_k,
        "embedder": embedder,
        "fusion": fusion,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="tests/fixtures/rag_eval/queries.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--embedder", default=os.getenv("HARNESS_RAG_PREFER", "hash"),
                       help="hash | sentence_transformer | auto")
    parser.add_argument("--fusion", default="rrf", choices=["rrf", "weighted"])
    parser.add_argument("--out", default="tests/benchmarks/results/hard_negatives_candidates.jsonl")
    args = parser.parse_args()
    summary = mine(
        queries_path=Path(args.queries),
        top_k=args.top_k,
        embedder=args.embedder,
        fusion=args.fusion,
        out_path=Path(args.out),
    )
    summary_path = Path(args.out).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

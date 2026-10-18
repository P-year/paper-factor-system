"""
tests/benchmarks/run_rerank_ab.py - 真 reranker A/B benchmark

用 bge-base-zh-v1.5 (或 hash) embedder + bge-reranker-base cross-encoder，
在 195 ground-truth query 上 A/B 对比：
- baseline：当前 retriever（无 rerank）
- + rerank：同一 retriever 加 BGEReranker

输出：Recall@5 / MRR / NDCG@5 + Δ。

用法：
    python tests/benchmarks/run_rerank_ab.py
    python tests/benchmarks/run_rerank_ab.py --embedder hash      # 零依赖 fallback
    python tests/benchmarks/run_rerank_ab.py --embedder bge-base-zh-v1.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "benchmarks"))

from _common import (
    QUERIES_PATH,
    PDF_DIR,
    RESULTS_DIR,
    _build_store_and_retriever,
    _check_fixtures_or_die,
    _run_evaluation,
    _save_result,
)


def _build_embedder(name: str):
    if name == "hash":
        from harness.rag.embedder import HashBackend
        return HashBackend(), "hash"
    if name.startswith("bge-"):
        from harness.rag.embedder import SentenceTransformerBackend
        model = f"BAAI/{name}" if not name.startswith("BAAI") else name
        emb = SentenceTransformerBackend(model_name=model)
        emb._load()
        return emb, name
    raise ValueError(f"unknown embedder: {name}")


def _build_real_reranker(model_name: str = "BAAI/bge-reranker-base"):
    from harness.rag.rerank import BGEReranker
    r = BGEReranker(model_name=model_name)
    r._load()
    return r


def _summarize(report: Dict[str, Any]) -> Dict[str, float]:
    return {
        "recall_at_k": report.get("recall_at_k", 0.0),
        "mrr": report.get("mrr", 0.0),
        "ndcg_at_k": report.get("ndcg_at_k", 0.0),
    }


def _diff(baseline: Dict[str, float], rerank: Dict[str, float]) -> Dict[str, float]:
    return {k: round(rerank[k] - baseline[k], 4) for k in baseline}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", default="bge-base-zh-v1.5",
                       help="hash | bge-base-zh-v1.5 | bge-small-zh-v1.5")
    parser.add_argument("--reranker", default="BAAI/bge-reranker-base")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-baseline", action="store_true",
                       help="只跑 rerank 一端（省时间；对比时手动读 results/）")
    args = parser.parse_args()

    _check_fixtures_or_die()

    print(f"Loading embedder: {args.embedder}")
    embedder, embed_label = _build_embedder(args.embedder)
    print(f"Loading reranker: {args.reranker} (real cross-encoder)")
    reranker = _build_real_reranker(args.reranker)
    print(f"OK: embedder={embed_label}, reranker={reranker.model_name}")
    print()

    # === Baseline (no rerank) ===
    baseline_report = None
    if not args.no_baseline:
        print("=== BASELINE (no rerank) ===")
        store_b, retr_b, tmp_b = _build_store_and_retriever(
            embedder, fusion_strategy="rrf", reranker=None,
        )
        t0 = time.time()
        baseline_report = _run_evaluation(retr_b, k=args.k)
        elapsed = time.time() - t0
        baseline_summary = _summarize(baseline_report)
        baseline_summary.update({
            "embedder": embed_label,
            "reranker": "none",
            "elapsed_s": round(elapsed, 2),
            "k": args.k,
            "queries_count": baseline_report.get("queries_count"),
        })
        out = _save_result(baseline_summary, embed_label, suffix="rerank-none")
        print(f"Recall@{args.k} = {baseline_summary['recall_at_k']:.4f}")
        print(f"MRR       = {baseline_summary['mrr']:.4f}")
        print(f"NDCG@{args.k} = {baseline_summary['ndcg_at_k']:.4f}")
        print(f"Time      = {elapsed:.1f}s")
        print(f"Saved: {out}")
        print()

    # === With real reranker ===
    print(f"=== + RERANK ({reranker.model_name}) ===")
    store_r, retr_r, tmp_r = _build_store_and_retriever(
        embedder, fusion_strategy="rrf", reranker=reranker,
    )
    t0 = time.time()
    rerank_report = _run_evaluation(retr_r, k=args.k)
    elapsed = time.time() - t0
    rerank_summary = _summarize(rerank_report)
    rerank_summary.update({
        "embedder": embed_label,
        "reranker": reranker.model_name.split("/")[-1],
        "elapsed_s": round(elapsed, 2),
        "k": args.k,
        "queries_count": rerank_report.get("queries_count"),
    })
    out = _save_result(rerank_summary, embed_label, suffix=f"rerank-{reranker.model_name.split('/')[-1]}")
    print(f"Recall@{args.k} = {rerank_summary['recall_at_k']:.4f}")
    print(f"MRR       = {rerank_summary['mrr']:.4f}")
    print(f"NDCG@{args.k} = {rerank_summary['ndcg_at_k']:.4f}")
    print(f"Time      = {elapsed:.1f}s")
    print(f"Saved: {out}")
    print()

    # === Diff report ===
    if baseline_report is not None:
        print("=== A/B DIFF ===")
        base_s = _summarize(baseline_report)
        diff = _diff(base_s, rerank_summary)
        print(f"  Recall@{args.k}: {base_s['recall_at_k']:.4f} → {rerank_summary['recall_at_k']:.4f}  (Δ {diff['recall_at_k']:+.4f})")
        print(f"  MRR       : {base_s['mrr']:.4f} → {rerank_summary['mrr']:.4f}  (Δ {diff['mrr']:+.4f})")
        print(f"  NDCG@{args.k}    : {base_s['ndcg_at_k']:.4f} → {rerank_summary['ndcg_at_k']:.4f}  (Δ {diff['ndcg_at_k']:+.4f})")
        print()
        # 保存 diff
        diff_path = RESULTS_DIR / f"rerank_ab_{embed_label}.json"
        diff_path.write_text(json.dumps({
            "embedder": embed_label,
            "reranker": reranker.model_name,
            "baseline": base_s,
            "rerank": {k: rerank_summary[k] for k in ["recall_at_k", "mrr", "ndcg_at_k"]},
            "diff": diff,
            "k": args.k,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Diff report: {diff_path}")


if __name__ == "__main__":
    main()

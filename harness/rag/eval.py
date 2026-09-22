"""
harness/rag/eval.py - RAG 检索质量评测

标准指标：
- Recall@K：top-k 中命中的相关 chunk 占比
- MRR (Mean Reciprocal Rank)：第一个相关 chunk 的排名倒数
- NDCG@K (Normalized Discounted Cumulative Gain)：位置加权，位置越高分越大

用法：
    from harness.rag.eval import run_evaluation

    report = run_evaluation(
        retriever=retriever,
        queries_path="tests/fixtures/rag_eval/queries.jsonl",
        k=5,
    )
    print(report)
    # {"recall_at_5": 0.78, "mrr": 0.83, "ndcg_at_5": 0.81, ...}
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _load_queries(queries_path: Path) -> List[Dict[str, Any]]:
    """加载 queries.jsonl。每行: {query, relevant: [...], irrelevant: [...]}"""
    out: List[Dict[str, Any]] = []
    with Path(queries_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _sources_of_chunk_id(chunk_id: str) -> str:
    """从 chunk_id (e.g. 'e38ddbd567dc:0') 提取 source_id (e.g. 'e38ddbd567dc')。"""
    return chunk_id.split(":", 1)[0] if ":" in chunk_id else chunk_id


def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    """Recall@K：top-k 中命中相关 source_id 的比例。"""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    top_k_sources = {_sources_of_chunk_id(c) for c in top_k}
    relevant_set = set(relevant)
    hits = sum(1 for r in relevant_set if r in top_k_sources)
    return hits / len(relevant_set)


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    """MRR：第一个相关 chunk 的排名倒数（无命中返回 0）。"""
    relevant_set = set(relevant)
    for rank, cid in enumerate(retrieved, start=1):
        if _sources_of_chunk_id(cid) in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    """NDCG@K：相关 source 的位置加权得分 / 理想 DCG。

    binary relevance（命中=1，不命中=0）。
    每个 source 只计一次（取最好排名），避免同 source 多 chunk 重复加分。
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)

    # 找每个 source 在 top_k 内的最好排名
    best_rank: Dict[str, int] = {}
    for rank, cid in enumerate(top_k, start=1):
        src = _sources_of_chunk_id(cid)
        if src in relevant_set and src not in best_rank:
            best_rank[src] = rank

    # DCG：每个 source 仅记最好排名一次
    dcg = sum(1.0 / math.log2(rank + 1) for rank in best_rank.values())

    # IDCG：理想情况，所有相关都在最前面
    n_rel = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(
    query_data: Dict[str, Any],
    retrieve_fn: Callable[[str, int], List[str]],
    k: int = 5,
) -> Dict[str, float]:
    """评测单条 query。retrieve_fn(query, k) → List[chunk_id]。"""
    query = query_data["query"]
    relevant = query_data.get("relevant", [])
    retrieved = retrieve_fn(query, k)
    # retrieved 含 top-k chunk_ids，但评测只关心 source_id（每篇论文可能多个 chunk）
    return {
        "query": query,
        "relevant_count": len(relevant),
        "retrieved_count": len(retrieved),
        "recall_at_k": recall_at_k(retrieved, relevant, k),
        "mrr": mrr(retrieved, relevant),
        "ndcg_at_k": ndcg_at_k(retrieved, relevant, k),
    }


def run_evaluation(
    retriever,
    queries_path: str | Path,
    *,
    k: int = 5,
    top_k_for_retrieval: Optional[int] = None,
) -> Dict[str, Any]:
    """跑全套 ground-truth queries，汇总指标。

    Args:
        retriever: PaperRetriever 实例
        queries_path: queries.jsonl 路径
        k: Recall@K / NDCG@K 的 K
        top_k_for_retrieval: 传给 retriever.retrieve 的 top_k（默认 = k）

    Returns:
        {
          "queries_count": int,
          "recall_at_k": float (mean),
          "mrr": float (mean),
          "ndcg_at_k": float (mean),
          "per_query": List[per-query dict],
        }
    """
    queries = _load_queries(Path(queries_path))
    top_k = top_k_for_retrieval or k

    def retrieve_fn(query: str, _: int) -> List[str]:
        chunks = retriever.retrieve(query, top_k=top_k)
        return [c["chunk_id"] for c in chunks]

    per_query = [evaluate_query(q, retrieve_fn, k=k) for q in queries]

    n = len(per_query) or 1
    return {
        "queries_count": len(per_query),
        "k": k,
        "recall_at_k": sum(r["recall_at_k"] for r in per_query) / n,
        "mrr": sum(r["mrr"] for r in per_query) / n,
        "ndcg_at_k": sum(r["ndcg_at_k"] for r in per_query) / n,
        "per_query": per_query,
    }


def format_report(report: Dict[str, Any]) -> str:
    """格式化报告。"""
    lines = [
        f"=== RAG Evaluation Report ===",
        f"Queries: {report['queries_count']}",
        f"K: {report['k']}",
        f"Recall@{report['k']}:  {report['recall_at_k']:.4f}",
        f"MRR:                {report['mrr']:.4f}",
        f"NDCG@{report['k']}:   {report['ndcg_at_k']:.4f}",
    ]
    return "\n".join(lines)


def write_baseline(report: Dict[str, Any], path: str | Path) -> None:
    """写 baseline metrics 文件（供后续 phase 对比）。"""
    # 只存聚合指标（per_query 太大）
    summary = {k: v for k, v in report.items() if k != "per_query"}
    Path(path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_baseline(path: str | Path) -> Dict[str, Any]:
    """读 baseline metrics。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
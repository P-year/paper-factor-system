"""
harness/rag/fusion.py - RRF (Reciprocal Rank Fusion) 多路召回融合

RRF 不依赖分数归一化，只看排名，对 embedding 极端值和 BM25 范围不敏感：

    RRF(d) = sum_i w_i / (k_const + rank_i(d))

- k_const：默认 60（论文标准值，平衡 top-heavy 和 long-tail）
- w_i：第 i 路召回的权重（通常各路相等 = 1.0）
- rank_i(d)：文档 d 在第 i 路召回中的排名（1-based，从 1 开始）

对比当前 weighted fusion（retriever.py:96-108）：
    combined = 0.7 * embed_s + 0.3 * (bm_s / bm25_max)
问题：BM25 per-query min-max 归一化，跨 query 分数不可比；
     BM25_max = 1.0 时所有 bm_s = bm_s（实际是原始分数），加权失衡。

RRF 优势：
- 不需要分数归一化
- 对单路召回异常值（0 或 ∞）不敏感
- 跨 query 可比、可复现

用法：
    from harness.rag.fusion import rrf_fuse
    ranked = rrf_fuse(
        embed_results={"c1": 0.8, "c2": 0.6, "c3": 0.4},
        bm25_results={"c2": 5.0, "c3": 3.0, "c4": 2.0},
        k_const=60,
        w_embed=1.0, w_bm25=1.0,
    )
    # ranked: [(score, chunk_id), ...]  按 score 降序
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def rrf_fuse(
    embed_results: Dict[str, float],
    bm25_results: Dict[str, float],
    *,
    k_const: int = 60,
    w_embed: float = 1.0,
    w_bm25: float = 1.0,
    extra_results: Optional[List[Dict[str, float]]] = None,
    extra_weights: Optional[List[float]] = None,
) -> List[Tuple[float, str]]:
    """RRF 多路融合。

    Args:
        embed_results: {chunk_id: score} (score 无所谓，仅用于排序)
        bm25_results: {chunk_id: score}
        k_const: RRF k 常数（默认 60）
        w_embed / w_bm25: 权重
        extra_results / extra_weights: 额外召回路（如 reranker / future query expansion）

    Returns:
        [(rrf_score, chunk_id), ...] 按 score 降序
    """
    # 每路按 score 降序排序，记录 chunk_id → rank
    embed_ranks = _to_rank(embed_results)
    bm25_ranks = _to_rank(bm25_results)

    all_ids = set(embed_ranks.keys()) | set(bm25_ranks.keys())
    scores: Dict[str, float] = {}
    for cid in all_ids:
        s = 0.0
        if cid in embed_ranks:
            s += w_embed / (k_const + embed_ranks[cid])
        if cid in bm25_ranks:
            s += w_bm25 / (k_const + bm25_ranks[cid])
        scores[cid] = s

    # 额外召回路
    if extra_results:
        weights = extra_weights or [1.0] * len(extra_results)
        for path, w in zip(extra_results, weights):
            ranks = _to_rank(path)
            for cid in ranks:
                scores[cid] = scores.get(cid, 0.0) + w / (k_const + ranks[cid])

    # 排序：score 降序 → tuple list
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [(score, cid) for cid, score in ranked]


def _to_rank(results: Dict[str, float]) -> Dict[str, int]:
    """{chunk_id: score} → {chunk_id: rank}（rank 从 1 开始，按 score 降序）。"""
    sorted_items = sorted(results.items(), key=lambda kv: -kv[1])
    return {cid: rank for rank, (cid, _) in enumerate(sorted_items, start=1)}
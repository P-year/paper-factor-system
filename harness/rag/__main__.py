"""
harness/rag/__main__.py - RAG CLI

子命令：
    status   查索引状态（model / dim / chunk 数 / source 数 / FAISS 大小）
    rebuild  全量重建索引
    search   按 query 检索
    inspect  检索 + 来源/分数/分布分析（调试用）
    clear    清空索引

用法：
    python -m harness.rag status
    python -m harness.rag rebuild
    python -m harness.rag search "动量因子" --top-k 5
    python -m harness.rag inspect "动量因子" --top-k 10
    python -m harness.rag clear
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from harness.paths import PAPER_DIR, RAG_DIR


def _get_store(prefer: str = "auto"):
    from harness.rag.store import get_default_store
    return get_default_store(prefer=prefer)


def cmd_status(args):
    store = _get_store(prefer=args.prefer)
    m = store.manifest()
    print(f"RAG_DIR:         {store.rag_dir}")
    # 用运行时 embedder 信息（prefer 可能覆盖磁盘 manifest）
    print(f"Active model:    {store.embedder.model_name}  (prefer={args.prefer})")
    print(f"Active dim:      {store.embedder.dim}")
    print(f"Stored model:    {m.get('model_name', '?')}")
    print(f"Chunk count:     {store.count()}")
    print(f"Source count:    {len(store.source_ids())}")
    print(f"FAISS total:     {len(store.index)}")
    print(f"BM25 size:       {len(store.bm25)}")
    if m.get("model_unavailable"):
        print("[WARN] Stored manifest: model unavailable")
    # v6-2: reranker + filters
    rr = m.get("reranker", {})
    print(f"Reranker:        enabled={rr.get('enabled', False)}  model={rr.get('model') or '-'}  status={rr.get('status', '?')}")
    flt = m.get("available_filters", [])
    print(f"Available filters: {', '.join(flt) if flt else '-'}")
    print(f"Updated at:      {m.get('updated_at', '?')}")
    # 文件大小
    total = 0
    for fp in (store.index_path, store.chunks_path, store.rag_dir / "bm25.json", store.rag_dir / "manifest.json"):
        if fp.exists():
            sz = fp.stat().st_size
            total += sz
            print(f"  {fp.name:20s} {sz / 1024:.1f} KB")
    print(f"Total on disk:   {total / (1024*1024):.2f} MB")


def cmd_rebuild(args):
    print(f"Rebuilding from {args.paper_dir}...")
    store = _get_store(prefer=args.prefer)
    n = store.rebuild()
    store.save()
    print(f"Indexed {n} chunks, total={store.count()}")
    print(f"Saved to {store.rag_dir}")


def cmd_search(args):
    store = _get_store(prefer=args.prefer)
    if store.count() == 0:
        print("[WARN] No chunks indexed. Run `python -m harness.rag rebuild` first.")
        sys.exit(1)
    from harness.rag.retriever import PaperRetriever
    from harness.rag.async_embed import get_async_embedder

    ae = get_async_embedder(embedder=store.embedder)
    r = PaperRetriever(store, async_embedder=ae, embedding_weight=args.embed_weight, bm25_weight=args.bm25_weight)
    chunks = r.retrieve(args.query, top_k=args.top_k, min_score=args.min_score)
    if not chunks:
        print(f"No results for: {args.query!r}")
        return
    print(f"Top {len(chunks)} results for: {args.query!r}\n")
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] score={c['score']:.3f} (embed={c.get('_score_embed', 0):.3f}, bm25={c.get('_score_bm25', 0):.1f})")
        print(f"    {c['paper_title']} ({c['arxiv_id']})")
        print(f"    {c['text'][:200].strip()}...")
        print()


def cmd_inspect(args):
    """v9-2：检索 + 分析维度（来源分布/分数/页面/字数）。"""
    store = _get_store(prefer=args.prefer)
    if store.count() == 0:
        print("[WARN] No chunks indexed. Run `python -m harness.rag rebuild` first.")
        sys.exit(1)
    from harness.rag.retriever import PaperRetriever
    from harness.rag.async_embed import get_async_embedder

    ae = get_async_embedder(embedder=store.embedder)
    r = PaperRetriever(
        store,
        async_embedder=ae,
        embedding_weight=args.embed_weight,
        bm25_weight=args.bm25_weight,
        fusion_strategy=args.fusion,
    )
    chunks = r.retrieve(args.query, top_k=args.top_k, min_score=args.min_score)

    if not chunks:
        print(f"No results for: {args.query!r}")
        return

    # 1. 标准 top-k 列表（与 search 相同）
    print(f"## Top {len(chunks)} results for: {args.query!r}\n")
    for i, c in enumerate(chunks, 1):
        page = c.get('start') or '-'
        n_chars = len(c.get('text', ''))
        print(f"[{i}] score={c['score']:.3f}  page={page}  chars={n_chars}")
        print(f"    {c['paper_title']} ({c['arxiv_id']})")
        preview = c.get('text', '')[:200].strip().replace('\n', ' ')
        print(f"    {preview}...")
        print()

    # 2. 来源分布（每篇 paper 占 top-k 多少）
    print("## Source distribution")
    src_counts: Dict[str, int] = {}
    for c in chunks:
        key = c.get('paper_title') or c.get('arxiv_id', '?')
        src_counts[key] = src_counts.get(key, 0) + 1
    max_count = max(src_counts.values()) if src_counts else 1
    for src, n in sorted(src_counts.items(), key=lambda kv: -kv[1]):
        bar = '#' * int(round(n / max_count * 20))
        print(f"  {n:2d}  {bar:20s} {src[:60]}")
    unique = len(src_counts)
    print(f"  ({len(chunks)} chunks / {unique} unique papers)")

    # 3. 分数分布（min/max/mean/std）
    if chunks:
        scores = [c['score'] for c in chunks]
        if len(scores) >= 2:
            mean = sum(scores) / len(scores)
            var = sum((s - mean) ** 2 for s in scores) / len(scores)
            std = var ** 0.5
        else:
            mean, std = scores[0], 0.0
        print()
        print("## Score distribution")
        print(f"  min={min(scores):.3f}  max={max(scores):.3f}  mean={mean:.3f}  std={std:.3f}")

    # 4. 页码分布（如有）
    pages = [c.get('start') for c in chunks if isinstance(c.get('start'), int)]
    if pages:
        print()
        print("## Page distribution")
        from collections import Counter
        for p, n in sorted(Counter(pages).items()):
            print(f"  page {p:>3}: {n} chunk(s)")

    # 5. 决策详情
    print()
    print("## Decision details")
    print(f"  fusion_strategy: {r.fusion_strategy}")
    print(f"  reranker:        {'on (' + type(r.reranker).__name__ + ')' if r.reranker else 'off'}")
    print(f"  weights:         embed={r.embedding_weight}, bm25={r.bm25_weight}")
    print(f"  expand_query:    {'yes (zh)' if any('一' <= c <= '鿿' for c in args.query) else 'no'}")


def cmd_clear(args):
    store = _get_store(prefer=args.prefer)
    if args.yes or input(f"Clear all RAG data in {store.rag_dir}? [y/N] ").lower() == "y":
        store._clear_chunks()
        store.save()
        print(f"Cleared {store.rag_dir}")
    else:
        print("Cancelled.")


def main():
    parser = argparse.ArgumentParser(description="Paper RAG CLI")
    parser.add_argument("--prefer", default=os.getenv("HARNESS_RAG_PREFER", "auto"),
                        choices=["auto", "hash", "sentence_transformer"],
                        help="embedder 偏好（默认 auto；可由 HARNESS_RAG_PREFER env 覆盖）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="查索引状态")
    p_status.set_defaults(func=cmd_status)

    p_rebuild = sub.add_parser("rebuild", help="全量重建")
    p_rebuild.add_argument("--paper-dir", default=str(PAPER_DIR))
    p_rebuild.set_defaults(func=cmd_rebuild)

    p_search = sub.add_parser("search", help="检索")
    p_search.add_argument("query", help="query 字符串")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--min-score", type=float, default=0.0)
    p_search.add_argument("--embed-weight", type=float, default=0.7)
    p_search.add_argument("--bm25-weight", type=float, default=0.3)
    p_search.set_defaults(func=cmd_search)

    p_inspect = sub.add_parser("inspect", help="检索 + 来源/分数/分布分析（调试用）")
    p_inspect.add_argument("query", help="query 字符串")
    p_inspect.add_argument("--top-k", type=int, default=5)
    p_inspect.add_argument("--min-score", type=float, default=0.0)
    p_inspect.add_argument("--embed-weight", type=float, default=0.7)
    p_inspect.add_argument("--bm25-weight", type=float, default=0.3)
    p_inspect.add_argument("--fusion", default="rrf", choices=["rrf", "weighted"],
                          help="融合策略（默认 rrf）")
    p_inspect.set_defaults(func=cmd_inspect)

    p_clear = sub.add_parser("clear", help="清空索引")
    p_clear.add_argument("--yes", action="store_true", help="跳过确认")
    p_clear.set_defaults(func=cmd_clear)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
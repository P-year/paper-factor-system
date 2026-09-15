"""
paper_retrieve_node - 在 analyze 之前检索相关参考论文（v5 RAG）

触发：hard_rule "retrieve_must_analyze"（has_papers and not has_retrieval_context）

行为：
1. 自动增量索引：state.collected_paper_ids 中未索引的论文 → 索引
2. 用 goal + 最后 user msg 构造 query → PaperRetriever.retrieve_for_state
3. 找到 → state.retrieval_context = chunks
4. 找不到 → state.retrieval_context = None + errors.append warn
5. 任何异常 → errors.append + retrieval_context=None（不阻塞 pipeline）

向后兼容：
- 默认 prefer='auto'：尝试 ST，失败降级 hash
- 测试/单线程可传 prefer='hash' 完全离线
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from harness.observability import observe_node
from harness.paths import PAPER_DIR


_RAG_PREFER = os.getenv("HARNESS_RAG_PREFER", "auto")  # "auto" | "hash" | "sentence_transformer"


def _build_store():
    """构造 RAG store（懒加载 + 优雅降级）。"""
    from harness.rag.store import get_default_store
    return get_default_store(prefer=_RAG_PREFER)


def _build_retriever(store):
    from harness.rag.retriever import PaperRetriever
    return PaperRetriever(store, default_top_k=5)


@observe_node("paper_retrieve_node")
def paper_retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """检索相关参考论文并注入到 state.retrieval_context。"""
    try:
        from harness.eval.paper_loader import normalize_papers
        from harness.rag.chunker import _safe_id

        collected_ids = state.get("collected_paper_ids", []) or []
        store = _build_store()
        retriever = _build_retriever(store)

        # 1) 自动增量索引（仅索引不在 manifest 的论文）
        try:
            if collected_ids:
                existing = set(store.source_ids())
                # 找出还没索引的 id
                missing = []
                for aid in collected_ids:
                    if aid not in existing and _safe_id({"arxiv_id": aid}) not in existing:
                        missing.append(aid)
                if missing:
                    # 读 paper JSON 找缺失的论文
                    import json
                    from pathlib import Path
                    wanted = set(missing)
                    found = []
                    for fp in sorted(Path(PAPER_DIR).glob("papers_*.json"), reverse=True):
                        try:
                            data = json.loads(fp.read_text(encoding="utf-8"))
                        except Exception:
                            continue
                        ps = data.get("papers", data) if isinstance(data, dict) else data
                        for p in ps:
                            aid = p.get("arxiv_id") or _safe_id(p)
                            if aid in wanted and aid not in {f.get("arxiv_id") for f in found}:
                                found.append(p)
                    if found:
                        try:
                            found = normalize_papers(found)
                        except Exception:
                            pass
                        store.index_papers(found)
        except Exception as e:
            # 索引失败不阻塞
            errs = list(state.get("errors", []))
            errs.append(f"paper_retrieve: auto-index failed: {e}")
            return {"errors": errs}

        # 2) 检索
        try:
            chunks = retriever.retrieve_for_state(state, top_k=5, min_score=0.0)
        except Exception as e:
            errs = list(state.get("errors", []))
            errs.append(f"paper_retrieve: retrieval failed: {e}")
            return {"retrieval_context": None, "errors": errs}

        # 3) 写回 state
        if chunks:
            return {"retrieval_context": chunks}
        else:
            errs = list(state.get("errors", []))
            errs.append("paper_retrieve: no similar papers found")
            return {"retrieval_context": None, "errors": errs}
    except Exception as e:
        # 兜底异常：返回空 retrieval_context，不阻塞 pipeline
        errs = list(state.get("errors", []))
        errs.append(f"paper_retrieve: unexpected error: {e}")
        return {"retrieval_context": None, "errors": errs}
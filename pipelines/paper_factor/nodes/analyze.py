"""
analyze_node - LLM 提取因子（调 extract_factor_tool）

v3-6：每个提取的 factor 标 _lineage（source_paper_id / extract_iteration / timestamp）。
v3-7：每个 factor 跑 verify_extracted_factor 打 _verified / _verify_issues，
      防止"假因子"（名字太宽泛 / 公式空泛 / 过度自信）流入下游。
v3-8：改用 harness/tool_call.verified_extracted_factor() 闭环 wrapper——
      自动 verify + 标记 + 跟踪每次调用的元数据（_tool_attempts / _tool_name）。
"""
from typing import Any, Dict, List

from pipelines.paper_factor.state import PaperFactorState
from harness.observability import observe_node
from harness.lineage import tag_factor_lineage
from harness.tool_call import verified_extracted_factor


def _load_papers(paper_ids: List[str]) -> List[Dict]:
    """从 data/papers/*.json 加载已采集论文（按 arxiv_id 过滤）"""
    from harness.paths import PAPER_DIR

    id_set = set(paper_ids)
    out = []
    files = sorted(PAPER_DIR.glob("papers_*.json"), reverse=True)
    for fp in files:
        import json
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            for p in data.get("papers", []):
                pid = p.get("arxiv_id") or p.get("link")
                if pid in id_set and p not in out:
                    out.append(p)
        except Exception:
            continue
    return out


@observe_node("analyze_node")
def analyze_node(state: PaperFactorState) -> Dict[str, Any]:
    """真实分析：对 collected_paper_ids 调用 extract_factor_tool（闭环 wrapper）

    v5：注入 state.retrieval_context（RAG 检索结果）作为 LLM 抽因子的参考上下文。
    """
    paper_ids = state.get("collected_paper_ids", [])

    if not paper_ids:
        return {
            "errors": state.get("errors", []) + ["analyze called with no paper_ids"],
        }

    papers = _load_papers(paper_ids)

    new_factors = list(state.get("extracted_factors", []))
    errors = list(state.get("errors", []))
    verify_log: List[Dict[str, Any]] = list(state.get("verify_log", []))
    # v5：从 state 取 RAG 检索的相似论文（paper_retrieve_node 注入）
    similar_papers = state.get("retrieval_context") or None

    for paper in papers:
        # v3-8 + v5：使用闭环 wrapper，注入 similar_papers
        r = verified_extracted_factor(
            paper, similar_papers=similar_papers, max_retries=0,
        )
        if r.get("error"):
            errors.append(f"extract_factor failed for {paper.get('title', '?')[:30]}: {r['error']}")
            continue
        if r.get("is_factor") and r.get("factor"):
            factor = r["factor"]
            # v5：标记"曾用 RAG 上下文"
            if similar_papers:
                factor["_used_rag_context"] = True
                factor["_rag_top_k"] = len(similar_papers)
            # 闭环 wrapper 已经打了 _verified / _verify_issues / _verify_confidence / _tool_attempts
            # v3-6：tag lineage
            tag_factor_lineage(factor, state, source_paper=paper)
            new_factors.append(factor)

            # 记录 verify 日志
            if not factor.get("_verified", True):
                verify_log.append({
                    "factor_name": factor.get("factor_name"),
                    "paper_title": paper.get("title", "")[:50],
                    "issues": factor.get("_verify_issues", []),
                })

    return {
        "extracted_factors": new_factors,
        "errors": errors,
        "verify_log": verify_log,
    }
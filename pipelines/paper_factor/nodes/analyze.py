"""
analyze_node - LLM 提取因子（调 extract_factor_tool）

v3-6：每个提取的 factor 标 _lineage（source_paper_id / extract_iteration / timestamp）。
"""
from typing import Any, Dict, List

from pipelines.paper_factor.state import PaperFactorState
from agent.tools import extract_factor_tool
from harness.observability import observe_node
from harness.lineage import tag_factor_lineage


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
    """真实分析：对 collected_paper_ids 调用 extract_factor_tool"""
    paper_ids = state.get("collected_paper_ids", [])

    if not paper_ids:
        return {
            "errors": state.get("errors", []) + ["analyze called with no paper_ids"],
        }

    papers = _load_papers(paper_ids)

    new_factors = list(state.get("extracted_factors", []))
    errors = list(state.get("errors", []))

    for paper in papers:
        r = extract_factor_tool(paper)
        if r.get("error"):
            errors.append(f"extract_factor failed for {paper.get('title', '?')[:30]}: {r['error']}")
            continue
        if r.get("is_factor") and r.get("factor"):
            # v3-6：tag lineage
            tag_factor_lineage(r["factor"], state, source_paper=paper)
            new_factors.append(r["factor"])

    return {
        "extracted_factors": new_factors,
        "errors": errors,
    }
"""
collect_node - 调 search_arxiv
"""
from typing import Any, Dict

from pipelines.paper_factor.state import PaperFactorState
from agent.tools import search_arxiv
from harness.observability import observe_node


@observe_node("collect_node")
def collect_node(state: PaperFactorState) -> Dict[str, Any]:
    """真实采集：调 search_arxiv"""
    args = {}
    if state.get("plan"):
        last = state["plan"][-1]
        for tc in last.get("args", []):
            if tc.get("tool") == "search_arxiv":
                args = tc.get("args", {})

    days = args.get("days", 90)
    force = args.get("force", False)

    result = search_arxiv(days=days, force=force)

    if result.get("error"):
        return {
            "errors": state.get("errors", []) + [f"collect failed: {result['error']}"],
            "collected_paper_ids": state.get("collected_paper_ids", []),
        }

    paper_ids = [p.get("arxiv_id") or p.get("link") for p in result.get("papers", [])]
    paper_ids = [pid for pid in paper_ids if pid]

    return {
        "collected_paper_ids": list(set(state.get("collected_paper_ids", []) + paper_ids)),
        "errors": state.get("errors", []),
    }
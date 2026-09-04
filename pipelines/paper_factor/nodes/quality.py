"""
quality_node - 因子质量评估 + 数据可行性

流程：
1. 先用启发式 check_data_feasibility（快、便宜）
2. 启发式失败时升级到 LLM 判断（慢、贵但能识别抽象因子）
3. LLM 也失败 → 标 infeasible

v3-6：每个 quality_result 标 _lineage（quality_iteration / quality_method）。
"""
from typing import Any, Dict

from pipelines.paper_factor.state import PaperFactorState
from agent.tools import check_data_feasibility, llm_check_data_feasibility
from harness.observability import observe_node
from harness.lineage import tag_quality_lineage


@observe_node("quality_node")
def quality_node(state: PaperFactorState) -> Dict[str, Any]:
    """对每个 extracted_factor 跑可行性检查（启发式 + LLM 升级）"""
    factors = state.get("extracted_factors", [])
    existing = {r.get("factor_name"): r for r in state.get("quality_results", [])}

    results = []
    errors = list(state.get("errors", []))

    for f in factors:
        fname = f.get("factor_name", "")

        # 缓存命中（已经检查过且不是 stub）
        if fname in existing and not existing[fname].get("_stub"):
            results.append(existing[fname])
            continue

        # 1. 启发式检查
        r = check_data_feasibility(f)

        if r.get("feasible"):
            qr = {
                "factor_name": fname,
                "feasible": True,
                "matching_factor": r.get("matching_factor"),
                "method": "heuristic",
                "approximation": "",
            }
            tag_quality_lineage(qr, state)
            results.append(qr)
            continue

        # 2. LLM 升级
        try:
            r_llm = llm_check_data_feasibility(f)
        except Exception as e:
            r_llm = {"feasible": False, "error": str(e)}

        if r_llm.get("feasible") and r_llm.get("matching_factor"):
            qr = {
                "factor_name": fname,
                "feasible": True,
                "matching_factor": r_llm["matching_factor"],
                "method": "llm",
                "approximation": r_llm.get("approximation", ""),
                "rationale": r_llm.get("rationale", ""),
            }
            tag_quality_lineage(qr, state)
            results.append(qr)
        else:
            qr = {
                "factor_name": fname,
                "feasible": False,
                "matching_factor": None,
                "method": r_llm.get("method", "llm_failed"),
                "missing": r_llm.get("missing", []),
                "rationale": r_llm.get("rationale", ""),
            }
            tag_quality_lineage(qr, state)
            results.append(qr)
            errors.append(
                f"feasibility check failed for {fname[:30]}: "
                f"{r_llm.get('rationale', 'unknown')[:80]}"
            )

    return {"quality_results": results, "errors": errors}
"""
backtest_node - 对 feasible 因子调 run_backtest

v3-6：每个 backtest_result 标 _lineage（backtest_iteration / backtest_timestamp / backtest_status）。
"""
from typing import Any, Dict

from pipelines.paper_factor.state import PaperFactorState
from agent.tools import run_backtest
from harness.observability import observe_node
from harness.lineage import tag_backtest_lineage


@observe_node("backtest_node")
def backtest_node(state: PaperFactorState) -> Dict[str, Any]:
    """对每个 feasible 因子调 run_backtest"""
    quality = {r.get("factor_name"): r for r in state.get("quality_results", [])}
    factors = state.get("extracted_factors", [])

    existing_bt = {r.get("factor_name"): r for r in state.get("backtest_results", [])}
    results = list(state.get("backtest_results", []))

    for f in factors:
        fname = f.get("factor_name", "")
        q = quality.get(fname, {})

        # 跳过不可行的
        if not q.get("feasible"):
            continue

        matching = q.get("matching_factor") or fname
        if fname in existing_bt and not existing_bt[fname].get("_stub"):
            continue

        # 真实回测
        r = run_backtest(
            factor_name=matching,
            universe="csi500",
            start_date="20230101",
            end_date="20241231",
            n_stocks=50,
        )
        r["original_factor_name"] = fname
        # v3-6：tag lineage
        tag_backtest_lineage(r, state)
        results.append(r)

    return {"backtest_results": results}
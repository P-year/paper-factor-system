"""
backtest_node - 对 feasible 因子调 run_backtest

v3-6：每个 backtest_result 标 _lineage（backtest_iteration / backtest_timestamp / backtest_status）。
v3-7：每个 backtest_result 跑 verify_backtest_result 打 _verified / _verify_issues / _verify_confidence，
      防止幻觉 / 数据错误静默传到 decide。
"""
from typing import Any, Dict, List

from pipelines.paper_factor.state import PaperFactorState
from agent.tools import run_backtest
from harness.observability import observe_node
from harness.lineage import tag_backtest_lineage
from harness.verify import verify_backtest_result


@observe_node("backtest_node")
def backtest_node(state: PaperFactorState) -> Dict[str, Any]:
    """对每个 feasible 因子调 run_backtest"""
    quality = {r.get("factor_name"): r for r in state.get("quality_results", [])}
    factors = state.get("extracted_factors", [])

    existing_bt = {r.get("factor_name"): r for r in state.get("backtest_results", [])}
    results = list(state.get("backtest_results", []))
    unverified: List[Dict[str, Any]] = list(state.get("unverified_backtests", []))

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

        # v3-7：verify 回测结果（防止幻觉 / 数据错误）
        v = verify_backtest_result(r)
        r["_verified"] = v["verified"]
        r["_verify_issues"] = v["issues"]
        r["_verify_confidence"] = v["confidence"]

        # v3-6：tag lineage
        tag_backtest_lineage(r, state)
        results.append(r)

        if not v["verified"]:
            unverified.append({
                "factor_name": fname,
                "issues": v["issues"],
                "verify_confidence": v["confidence"],
            })

    return {
        "backtest_results": results,
        "unverified_backtests": unverified,
    }
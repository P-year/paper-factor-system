"""
backtest_node - 对 feasible 因子调 run_backtest

v3-6：每个 backtest_result 标 _lineage（backtest_iteration / backtest_timestamp / backtest_status）。
v3-7：每个 backtest_result 跑 verify_backtest_result 打 _verified / _verify_issues / _verify_confidence，
      防止幻觉 / 数据错误静默传到 decide。
v3-8：改用 harness/tool_call.verified_backtest() 闭环 wrapper——
      自动 verify + 重试 + 恢复（n_dates 太少时自动拉长日期范围），
      保证每次回测结果都经过 sanity check 才往下传。
"""
from typing import Any, Dict, List

from pipelines.paper_factor.state import PaperFactorState
from harness.observability import observe_node
from harness.lineage import tag_backtest_lineage
from harness.tool_call import verified_backtest


@observe_node("backtest_node")
def backtest_node(state: PaperFactorState) -> Dict[str, Any]:
    """对每个 feasible 因子调 run_backtest（闭环 wrapper）"""
    quality = {r.get("factor_name"): r for r in state.get("quality_results", [])}
    factors = state.get("extracted_factors", [])

    existing_bt = {r.get("factor_name"): r for r in state.get("backtest_results", [])}
    results = list(state.get("backtest_results", []))
    unverified: List[Dict[str, Any]] = list(state.get("unverified_backtests", []))
    retry_log: List[Dict[str, Any]] = list(state.get("backtest_retry_log", []))

    for f in factors:
        fname = f.get("factor_name", "")
        q = quality.get(fname, {})

        # 跳过不可行的
        if not q.get("feasible"):
            continue

        matching = q.get("matching_factor") or fname
        if fname in existing_bt and not existing_bt[fname].get("_stub"):
            continue

        # v3-8：使用闭环 wrapper（verify + retry + recovery）
        r = verified_backtest(
            factor_name=matching,
            universe="csi500",
            start_date="20230101",
            end_date="20241231",
            n_stocks=50,
            max_retries=1,  # 失败重试 1 次
        )
        r["original_factor_name"] = fname

        # v3-6：tag lineage
        tag_backtest_lineage(r, state)
        results.append(r)

        # 记录重试日志（用于调试 / 上层决策）
        if r.get("_tool_attempts", 1) > 1:
            retry_log.append({
                "factor_name": fname,
                "attempts": r["_tool_attempts"],
                "final_verified": r.get("_verified"),
                "final_confidence": r.get("_verify_confidence"),
                "issues": r.get("_verify_issues", []),
            })

        if not r.get("_verified", True):
            unverified.append({
                "factor_name": fname,
                "issues": r.get("_verify_issues", []),
                "verify_confidence": r.get("_verify_confidence"),
                "attempts": r.get("_tool_attempts"),
            })

    return {
        "backtest_results": results,
        "unverified_backtests": unverified,
        "backtest_retry_log": retry_log,
    }
"""
auto_approve.py - paper_factor scheduler 模式的自动审批回调

从 main_scheduler.py:_auto_approve_callback 搬运而来。
签名：(state, config) -> List[Decision]，每条 Decision 形如：
    {"factor_name": str, "decision": "approved"|"rejected", "notes": str}
"""
from typing import Any, Dict, List


def auto_approve(state: Dict[str, Any], config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """按 BACKTEST_CONFIG 阈值对 pending_decisions 自动审批。"""
    from config import BACKTEST_CONFIG

    decisions = []
    for p in state.get("pending_decisions", []):
        bt = p.get("backtest", {})
        icir = bt.get("ICIR", 0)
        top_ret = bt.get("top_return", -100)
        min_icir = BACKTEST_CONFIG.get("min_icir", 0.3)
        min_top = BACKTEST_CONFIG.get("min_top_return", -20)

        if icir >= min_icir and top_ret >= min_top:
            decision = "approved"
        elif icir < 0 or top_ret < -10:
            decision = "rejected"
        else:
            decision = "approved"  # 默认通过，避免卡住调度

        decisions.append({
            "factor_name": p.get("factor_name"),
            "decision": decision,
            "notes": f"auto: ICIR={icir:.2f}, top={top_ret:.1f}%",
        })

    return decisions
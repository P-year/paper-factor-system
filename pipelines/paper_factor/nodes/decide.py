"""
decide_node - 因子审批（支持 interrupt + 人工审批 + 自动审批）

工作流（依赖 LangGraph interrupt_before=["decide"]）：
1. 第一次到达 decide：interrupt 触发，节点尚未执行
2. main_agent.py 检测到挂起，读取 pending_decisions，提示用户输入
3. 用户输入 `approve EP` / `reject EP reason`
4. main_agent.py 调 graph.update_state(config, {"human_action": {...}})
5. 调 graph.invoke(None, config) 恢复执行
6. decide_node 看到 human_action 后应用决策
7. 调度模式：human_action 已被 AutoApproveCB 预先填入
"""
from typing import Any, Dict

from pipelines.paper_factor.state import PaperFactorState
from agent.tools import update_factor_status
from harness.observability import observe_node


def _apply_human_action(state: PaperFactorState, action: Dict) -> Dict[str, Any]:
    """应用一条 human_action，更新因子状态 + 移除 pending"""
    name = action.get("factor_name", "")
    decision = action.get("decision", "approved")  # approved/rejected
    notes = action.get("notes", "")

    # 调真实工具（写因子库）
    update_factor_status(name=name, status=decision, notes=notes)

    # 移除该 pending
    pending = [p for p in state.get("pending_decisions", []) if p.get("factor_name") != name]

    # 加入 extracted_factors 的最终 status
    factors = list(state.get("extracted_factors", []))
    for f in factors:
        if f.get("factor_name") == name:
            f["status"] = decision
            f["notes"] = notes

    return {"pending_decisions": pending, "extracted_factors": factors}


@observe_node("decide_node")
def decide_node(state: PaperFactorState) -> Dict[str, Any]:
    """决策节点：应用 human_action 或自动决策"""

    # 收集所有待审批项
    backtest = state.get("backtest_results", [])
    pending = list(state.get("pending_decisions", []))

    # 找出还没进 pending 的回测结果
    pending_names = {p.get("factor_name") for p in pending}
    for r in backtest:
        fname = r.get("original_factor_name") or r.get("factor_name", "")
        if fname and fname not in pending_names and not r.get("_stub"):
            pending.append({
                "factor_name": fname,
                "backtest": r,
                "decision": "pending",
            })

    updates: Dict[str, Any] = {"pending_decisions": pending}

    # 应用 human_action
    action = state.get("human_action")
    if action:
        applied = _apply_human_action(state, action)
        updates.update(applied)
        updates["human_action"] = None  # 清空

    return updates
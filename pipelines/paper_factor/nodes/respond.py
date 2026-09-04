"""
respond_node - 生成最终回复给用户
"""
import json
from typing import Any, Dict

from pipelines.paper_factor.state import PaperFactorState
from harness.observability import observe_node


@observe_node("respond_node")
def respond_node(state: PaperFactorState) -> Dict[str, Any]:
    """汇总当前状态，生成最终回复消息"""
    summary_lines = [
        f"任务完成摘要：",
        f"  - 目标：{state.get('goal', '')}",
        f"  - 已执行节点：{state.get('visited_nodes', [])}",
        f"  - 采集论文：{len(state.get('collected_paper_ids', []))} 篇",
        f"  - 提取因子：{len(state.get('extracted_factors', []))} 个",
        f"  - 入库因子：{state.get('persisted_count', 0)} 个",
        f"  - 回测结果：{len(state.get('backtest_results', []))} 个",
        f"  - 待审批：{len(state.get('pending_decisions', []))} 个",
    ]

    if state.get("errors"):
        summary_lines.append(f"  - 错误：{len(state['errors'])} 个")

    if state.get("backtest_results"):
        summary_lines.append("")
        summary_lines.append("回测结果 Top 3：")
        sorted_bt = sorted(
            state["backtest_results"],
            key=lambda r: r.get("IC") if isinstance(r.get("IC"), (int, float)) else 0,
            reverse=True,
        )
        for r in sorted_bt[:3]:
            ic = r.get("IC", "?")
            icir = r.get("ICIR", "?")
            ic_s = f"{ic:.4f}" if isinstance(ic, (int, float)) else str(ic)
            icir_s = f"{icir:.4f}" if isinstance(icir, (int, float)) else str(icir)
            summary_lines.append(
                f"  - {r.get('factor_name', '?')}: "
                f"IC={ic_s} ICIR={icir_s} "
                f"[{r.get('status', '?')}]"
            )

    message = {
        "role": "assistant",
        "content": "\n".join(summary_lines),
    }

    # v4：state["messages"] 有 add_messages reducer，只需 return 新增消息
    return {"messages": [message]}
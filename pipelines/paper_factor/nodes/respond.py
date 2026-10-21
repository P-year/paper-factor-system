"""
respond_node - 生成最终回复给用户

v9-1: 加对话模式 — 简单问题直接 LLM 回答，不走 pipeline 摘要
"""
import json
from typing import Any, Dict

from pipelines.paper_factor.state import PaperFactorState
from harness.observability import observe_node


# 意图识别：是否为对话式问题（非任务执行）
_TASK_KEYWORDS = {
    "搜索", "采集", "收集", "查找", "下载",
    "提取", "挖掘", "分析论文", "提取因子",
    "回测", "测试因子", "运行", "执行",
    "上传", "导入", "提交", "对比", "比较",
}

# 明确的对话式问句模式
_CONVERSATIONAL_PATTERNS = [
    "什么是", "是什么", "怎么理解", "解释一下", "介绍一下",
    "区别是什么", "有什么不同", "为什么", "如何理解",
    "你能做什么", "你会什么", "帮我理解",
    "你觉得", "你的看法", "有什么建议",
    "你好", "谢谢", "感谢",
]


def _is_conversational(goal: str) -> bool:
    """判断是否为对话式问题（非任务执行）。

    必须满足:
    1. 含明确的对话式问句模式（What is / 解释 / 介绍 等）
    2. 不含任务关键词（搜索/提取/回测 等）
    """
    goal = goal.strip()
    has_conv = any(p in goal for p in _CONVERSATIONAL_PATTERNS)
    has_task = any(kw in goal for kw in _TASK_KEYWORDS)
    return has_conv and not has_task


def _generate_conversational_reply(goal: str) -> str:
    """用 LLM 生成对话式回答。"""
    try:
        from harness.llm_client import call_llm
        system = (
            "你是 FactorLab 的 AI 助手，一个专业的量化因子研究平台。\n"
            "你可以：搜索/上传论文、提取量化因子、公式编译、因子回测、策略回测。\n"
            "用简洁、专业、友好的语气回答用户问题。"
        )
        result = call_llm(system=system, user=goal, timeout=15)
        if result and result.strip():
            return result.strip()
    except Exception:
        pass
    return f"你好！我是 FactorLab 助手。我是你的问题「{goal}」。有什么关于因子挖掘或回测的问题可以帮你？"


@observe_node("respond_node")
def respond_node(state: PaperFactorState) -> Dict[str, Any]:
    """生成最终回复（对话模式 或 任务摘要）"""
    goal = state.get("goal", "")

    # === 对话模式 ===
    if _is_conversational(goal):
        reply = _generate_conversational_reply(goal)
        return {
            "messages": [{"type": "ai", "content": reply}],
            "final_message": reply,
        }

    # === 任务摘要模式 ===
    summary_lines = [
        f"任务完成摘要：",
        f"  - 目标：{goal}",
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

    # 相关论文推荐
    retrieval_context = state.get("retrieval_context")
    if retrieval_context:
        try:
            summary_lines.append("")
            summary_lines.append(f"相关论文推荐 Top {len(retrieval_context)}（本地 RAG）：")
            for i, ch in enumerate(retrieval_context[:5], start=1):
                title = ch.get("paper_title", "") or ch.get("arxiv_id", "")
                arxiv = ch.get("arxiv_id", "")
                score = ch.get("score", 0.0)
                text = (ch.get("text", "") or "").strip().replace("\n", " ")[:120]
                summary_lines.append(
                    f"  [{i}] {title} ({arxiv}) — score={score:.3f}\n      {text}..."
                )
        except Exception:
            pass

    message_text = "\n".join(summary_lines)
    return {
        "messages": [{"type": "ai", "content": message_text}],
        "final_message": message_text,
    }
"""
harness/plan_node.py - 通用 plan_node 工厂

用法：
    plan = get("paper_factor")
    plan_node = make_plan_node(plan)

    state = AgentState(...)
    out = plan_node(state)

行为（与原 plan_node 一致）：
1. 调 LLM 决策 next_node + tool_calls + reason（system prompt = pipeline.prompts["orchestrator"]）
2. 应用 yaml 里声明的 hard_rules（按顺序，第一条命中覆盖 LLM）
3. visited_nodes 计数防重复（同一节点出现 2 次强制 respond）
4. iteration_count >= pipeline.max_iter 强制 respond
5. next_node 必须 ∈ pipeline.nodes，否则兜底 respond

注：fallback_decision 仍是 LLM 失败时用的启发式（基于 state 推进）。
"""
import json
from typing import Any, Callable, Dict

from harness.hard_rules import apply_rules, HardRule, build_rule
from harness.observability import observe_node
from harness.pipeline import PipelineConfig


def _tool_signatures_for(tool_list) -> str:
    """把 tool 列表渲染成简短的签名描述。"""
    lines = []
    for t in tool_list:
        doc_first = (t.__doc__ or "").split("\n")[0].strip() if t.__doc__ else "no doc"
        lines.append(f"- {t.__name__}: {doc_first}")
    return "\n".join(lines)


def _truncate_state(state: Dict[str, Any], *, max_iter: int) -> str:
    """只给 LLM 看精简后的 state。"""
    return json.dumps({
        "goal": state.get("goal", ""),
        "iteration_count": state.get("iteration_count", 0),
        "max_iter": max_iter,
        "visited_nodes": state.get("visited_nodes", []),
        "collected_paper_count": len(state.get("collected_paper_ids", [])),
        "extracted_factor_count": len(state.get("extracted_factors", [])),
        "backtest_result_count": len(state.get("backtest_results", [])),
        "pending_decision_count": len(state.get("pending_decisions", [])),
        "errors": state.get("errors", [])[-3:],
    }, ensure_ascii=False, indent=2)


def _fallback_decision(state: Dict[str, Any], *, max_iter: int, node_names: list) -> Dict[str, Any]:
    """LLM 失败时的兜底（基于 state 启发式推进）。

    节点集合由 pipeline.nodes 决定（不再硬编码 paper_factor 节点名）。
    """
    if state.get("iteration_count", 0) >= max_iter:
        return {"next_node": "respond", "tool_calls": [], "reason": "max_iter reached"}

    # 按 "通常顺序" 启发式：plan 之后的第一个空槽位
    # 注意：这是兜底，yaml 的 hard_rules 是主路径
    if not state.get("collected_paper_ids") and "collect" in node_names:
        return {"next_node": "collect", "tool_calls": [{"tool": "search_arxiv", "args": {"days": 90}}], "reason": "no papers collected yet"}

    if not state.get("extracted_factors") and "analyze" in node_names:
        return {"next_node": "analyze", "tool_calls": [], "reason": "papers collected but no factors extracted"}

    if state.get("extracted_factors") and not state.get("persist_attempted") and "persist_factors" in node_names:
        return {"next_node": "persist_factors", "tool_calls": [], "reason": "factors need to be persisted"}

    if state.get("persist_attempted") and not state.get("quality_results") and "quality_check" in node_names:
        return {"next_node": "quality_check", "tool_calls": [], "reason": "persisted but no quality"}

    if not state.get("backtest_results") and "backtest" in node_names:
        return {"next_node": "backtest", "tool_calls": [], "reason": "factors extracted but no backtest"}

    if state.get("pending_decisions") and "decide" in node_names:
        return {"next_node": "decide", "tool_calls": [], "reason": "factors pending decision"}

    return {"next_node": "respond", "tool_calls": [], "reason": "all done, summarize"}


def _collect_tools(pipeline: PipelineConfig):
    """从 pipeline.tool_modules 聚合所有 tool 函数。"""
    import importlib
    tools = []
    for mod_path in pipeline.tool_modules:
        try:
            mod = importlib.import_module(mod_path)
            # 优先 TOOL_LIST，否则遍历 __all__
            if hasattr(mod, "TOOL_LIST"):
                tools.extend(mod.TOOL_LIST)
            elif hasattr(mod, "__all__"):
                for name in mod.__all__:
                    obj = getattr(mod, name, None)
                    if callable(obj):
                        tools.append(obj)
        except Exception:
            # 模块不存在时静默跳过（Phase 0 兼容）
            pass
    return tools


def _get_prompt_template(pipeline: PipelineConfig) -> str:
    """读 yaml 的 orchestrator prompt。"""
    tpl = pipeline.prompts.get("orchestrator")
    if tpl:
        return tpl
    # 兜底（Phase 0 阶段或 yaml 没填）
    return "你是 orchestrator。{state_json} {tool_sigs} {max_iter} {goal}"


def _resolve_call_llm():
    """获取 LLM 调用函数（直接从 harness.llm_client 拿）。"""
    from harness.llm_client import call_llm_json
    return call_llm_json


def make_plan_node(pipeline: PipelineConfig) -> Callable:
    """返回一个闭包，签名 plan_node(state) -> dict。

    闭包捕获：pipeline 对象本身（每次调用重新读 nodes / prompts / rules）
    这样允许 plan_node 在 pipeline.nodes 尚未填充时构造（lazy binding）。
    """
    # 预编译 hard rules + 工具列表 + prompt template
    rules: list = [build_rule(r) for r in pipeline.hard_rules]
    tools = _collect_tools(pipeline)
    prompt_template = _get_prompt_template(pipeline)
    call_llm_json = _resolve_call_llm()
    max_iter = pipeline.max_iter

    @observe_node("plan_node")
    def plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """plan 节点：LLM 决策 + 硬规则覆盖 + 防循环 + max_iter 兜底。"""
        # 每次调用重新查 node_names（避免工厂时序 bug）
        node_names = set(pipeline.nodes.keys())

        system = prompt_template.format(
            state_json=_truncate_state(state, max_iter=max_iter),
            tool_sigs=_tool_signatures_for(tools),
            max_iter=max_iter,
            goal=state.get("goal", "(no goal)"),
        )
        user = "请根据当前状态决定下一步。严格输出 JSON。"

        # 1. LLM 决策
        try:
            decision = call_llm_json(system, user)
        except Exception as e:
            decision = _fallback_decision(state, max_iter=max_iter, node_names=list(node_names))
            decision["llm_error"] = str(e)

        # 2. 应用硬规则（yaml）
        decision = apply_rules(rules, state, decision)

        # 3. 防重复：visited.count(node) >= 2 → respond
        visited = list(state.get("visited_nodes", []))
        next_node = decision.get("next_node", "respond")
        if visited.count(next_node) >= 2:
            next_node = "respond"
            decision["next_node"] = "respond"
            decision["reason"] = "node visited too many times, forced respond"

        # 4. 校验 next_node 在 pipeline.nodes
        if next_node not in node_names:
            next_node = "respond"
            decision["next_node"] = "respond"
            decision["reason"] = "unknown next_node, forced respond"

        visited.append(next_node)

        new_plan = list(state.get("plan", []))
        new_plan.append({
            "node": next_node,
            "args": decision.get("tool_calls", []),
            "reason": decision.get("reason", ""),
        })

        return {
            "plan": new_plan,
            "current_step_idx": len(new_plan),
            "iteration_count": state.get("iteration_count", 0) + 1,
            "visited_nodes": visited,
            "errors": state.get("errors", []) + (
                [decision["llm_error"]] if "llm_error" in decision else []
            ),
        }

    return plan_node
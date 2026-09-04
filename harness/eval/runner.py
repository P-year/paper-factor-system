"""
harness/eval/runner.py - eval case 运行器

用法：
    from harness.eval import run_case, run_cases, run_cases_from_file

    # 单个 case
    result = await run_case(case_dict)

    # 多个 case
    results = await run_cases(case_list)

    # 从文件
    results = await run_cases_from_file("eval/test_cases.json")

类型分发：
    full           全流程（默认），interrupt_before 从 yaml 默认
    inject_papers  注入 papers 到 state，build_graph 用 skip_collect_mode
    skip_approve   全流程但 interrupt_before=[]
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.eval.case_schema import CaseType, normalize_case
from harness.eval.judge import judge_output
from harness.eval.paper_loader import (
    normalize_papers,
    write_papers_to_collection,
)
from harness.registry import get as _registry_get
from harness.graph_builder import build_graph as _build_graph
from pipelines.paper_factor.state import PaperFactorState


# === state 构造 ===

def _build_initial_state(case: Dict[str, Any]) -> Dict[str, Any]:
    """构造 pipeline state 初始 dict。

    包含 BasePipelineState + PaperFactorState 所有字段的默认值。
    """
    state: Dict[str, Any] = {
        # === BasePipelineState ===
        "plan": [],
        "current_step_idx": 0,
        "iteration_count": 0,
        "visited_nodes": [],
        "errors": [],
        "messages": [],
        "session_id": f"eval-{case['id']}",
        "run_mode": "eval",
        "pipeline_name": case.get("pipeline", "paper_factor"),
        "human_action": None,
        # === PaperFactorState ===
        "collected_paper_ids": [],
        "extracted_factors": [],
        "persisted_count": 0,
        "persist_attempted": False,
        "quality_results": [],
        "backtest_results": [],
        "pending_decisions": [],
    }

    ctype = case.get("type", CaseType.FULL.value)

    # inject_papers：先把 papers 注入 state
    if ctype == CaseType.INJECT_PAPERS.value:
        papers = normalize_papers(case["input_papers"])
        write_papers_to_collection(papers)
        state["collected_paper_ids"] = [p["arxiv_id"] for p in papers]

    # goal 总是写入
    state["goal"] = case["goal"]

    return state


# === interrupt_before 计算 ===

def _interrupt_before_for(case: Dict[str, Any]) -> Optional[List[str]]:
    """根据 case type 计算 interrupt_before。

    full         → yaml 默认（["decide"]）
    inject_papers → [] （不挂起）
    skip_approve → [] （不挂起）
    """
    ctype = case.get("type", CaseType.FULL.value)
    if ctype in (CaseType.SKIP_APPROVE.value, CaseType.INJECT_PAPERS.value):
        return []
    return None  # yaml 默认


# === 单 case 执行 ===

async def run_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """跑单个 eval case。返回 result dict。

    Returns:
        {
          "test_case_id": str,
          "type": str,
          "goal": str,
          "trace": [...],                  # 节点执行轨迹
          "final_output": str,             # respond 节点产出
          "state": {...},                  # 最终 state
          "judge_result": {...},           # LLM-as-Judge 评分
          "metrics": {step_count, avg_score, pass},
          "duration_s": float,
          "error": str | None,
        }
    """
    case = normalize_case(case)
    started = datetime.now()

    pipeline = _registry_get(case["pipeline"])
    if pipeline is None:
        return {
            "test_case_id": case["id"],
            "type": case["type"],
            "goal": case["goal"],
            "trace": [],
            "final_output": "",
            "state": {},
            "judge_result": None,
            "metrics": {"step_count": 0, "avg_score": 0, "pass": False},
            "duration_s": 0,
            "error": f"pipeline '{case['pipeline']}' not registered",
        }

    initial_state = _build_initial_state(case)
    interrupt_before = _interrupt_before_for(case)

    graph, _ = _build_graph(pipeline, interrupt_before=interrupt_before or [])
    config = {"configurable": {"thread_id": f"eval-{case['id']}"}}

    trace: List[Dict[str, Any]] = []
    final_state: Dict[str, Any] = {}

    try:
        async for event in graph.astream(initial_state, config=config):
            for node_name, node_output in event.items():
                if node_name == "__end__":
                    continue
                if isinstance(node_output, dict) and "plan" in node_output and node_output["plan"]:
                    last = node_output["plan"][-1]
                    trace.append({
                        "node": last.get("node", node_name),
                        "reason": last.get("reason", ""),
                    })
                else:
                    trace.append({
                        "node": node_name,
                        "reason": str(node_output)[:200] if node_output else "",
                    })
                final_state = node_output
    except Exception as e:
        return {
            "test_case_id": case["id"],
            "type": case["type"],
            "goal": case["goal"],
            "trace": trace,
            "final_output": "",
            "state": final_state,
            "judge_result": None,
            "metrics": {"step_count": len(trace), "avg_score": 0, "pass": False},
            "duration_s": (datetime.now() - started).total_seconds(),
            "error": str(e),
        }

    # 获取最终 state（含 messages / extracted_factors / backtest_results）
    snapshot = graph.get_state(config)
    if snapshot and snapshot.values:
        final_state = snapshot.values

    # 提取 respond 输出
    final_output = ""
    msgs = final_state.get("messages", []) if final_state else []
    if msgs:
        last_msg = msgs[-1]
        if isinstance(last_msg, dict):
            final_output = last_msg.get("content", "")
        else:
            final_output = str(last_msg)

    # 调用 judge
    judge_result = judge_output(
        goal=case["goal"],
        criteria=case["judge_criteria"],
        trace=trace,
        final_output=final_output,
    )

    avg_score = (
        sum(s.get("score", 0) for s in judge_result.get("scores", []))
        / max(len(judge_result.get("scores", [])), 1)
    )

    return {
        "test_case_id": case["id"],
        "type": case["type"],
        "goal": case["goal"],
        "trace": trace,
        "final_output": final_output,
        "state": final_state,
        "judge_result": judge_result,
        "metrics": {
            "step_count": len(trace),
            "avg_score": round(avg_score, 2),
            "pass": judge_result.get("pass", False),
        },
        "duration_s": round((datetime.now() - started).total_seconds(), 2),
        "error": None,
    }


def run_case_sync(case: Dict[str, Any]) -> Dict[str, Any]:
    """同步版本的 run_case。"""
    return asyncio.run(run_case(case))


# === 多 case ===

async def run_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """跑多个 case。串行（LLM 调用顺序避免 rate limit）。"""
    results = []
    for c in cases:
        print(f"\n{'='*60}")
        print(f"[{c.get('id', '?')}] {c.get('goal', '?')}")
        print(f"{'='*60}")
        result = await run_case(c)
        results.append(result)
    return results


async def run_cases_from_file(cases_file: str) -> List[Dict[str, Any]]:
    """从 JSON 文件加载 cases 并跑。"""
    with open(cases_file, encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("test_cases", [])
    return await run_cases(cases)


# === report 生成 ===

def generate_report(results: List[Dict[str, Any]], output_path: str) -> None:
    """生成 markdown 报告。"""
    lines = [
        "# Paper Factor System 评估报告",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 用例数：{len(results)}",
        "",
        "## 总览",
        "",
        "| 用例 | 类型 | 平均分 | 通过 | 步数 | 耗时(s) |",
        "|---|---|---|---|---|---|",
    ]

    pass_count = 0
    total_score = 0.0
    for r in results:
        m = r.get("metrics", {})
        avg = m.get("avg_score", 0)
        passed = "[OK]" if m.get("pass") else "[X]"
        duration = r.get("duration_s", 0)
        lines.append(
            f"| {r['test_case_id']} | {r.get('type', '?')} | {avg:.2f} | {passed} | "
            f"{m.get('step_count', 0)} | {duration} |"
        )
        if m.get("pass"):
            pass_count += 1
        total_score += avg

    pass_rate = pass_count / max(len(results), 1) * 100
    avg_overall = total_score / max(len(results), 1)

    lines.extend([
        "",
        f"**通过率：{pass_rate:.1f}% ({pass_count}/{len(results)})**",
        f"**平均分：{avg_overall:.2f} / 5.00**",
        "",
        "---",
        "",
    ])

    for r in results:
        lines.extend([
            f"## {r['test_case_id']} ({r.get('type', '?')})",
            "",
            f"**目标：** {r['goal']}",
            "",
        ])

        if r.get("error"):
            lines.extend([f"**错误：** {r['error']}", ""])
            lines.append("---")
            lines.append("")
            continue

        jr = r.get("judge_result") or {}
        scores = jr.get("scores", [])
        if scores:
            lines.append("| # | 标准 | 分数 | 评语 |")
            lines.append("|---|---|---|---|")
            for i, s in enumerate(scores, 1):
                lines.append(
                    f"| {i} | {s.get('criterion', '')} | {s.get('score', 0)} | {s.get('comment', '')} |"
                )
            lines.append("")

        if jr.get("overall_comment"):
            lines.extend([f"**总体评价：** {jr['overall_comment']}", ""])

        trace = r.get("trace", [])
        if trace:
            lines.append("**执行轨迹：**")
            for i, step in enumerate(trace, 1):
                lines.append(f"{i}. `[{step.get('node', '?')}]` {step.get('reason', '')[:120]}")
            lines.append("")

        # 关键指标
        st = r.get("state", {})
        if st:
            lines.append("**关键指标：**")
            lines.append(f"- 采集论文：{len(st.get('collected_paper_ids', []))}")
            lines.append(f"- 提取因子：{len(st.get('extracted_factors', []))}")
            lines.append(f"- 回测结果：{len(st.get('backtest_results', []))}")
            lines.append(f"- 访问节点：{st.get('visited_nodes', [])}")
            lines.append("")

        lines.append("---")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n报告已生成：{output_path}")
    print(f"通过率：{pass_rate:.1f}%，平均分：{avg_overall:.2f}")
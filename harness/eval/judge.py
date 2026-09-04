"""
harness/eval/judge.py - LLM-as-Judge 评估器

从 eval/judge.py 搬运而来。改用 harness.llm_client.call_llm_json 替代 langchain_openai。

打分维度：
    scores: List[{criterion, score, comment}]，每个标准 1-5 分
    overall_comment: 总体评价
    pass: 所有 score >= 3
"""
import json
import re
from typing import Any, Dict, List

from harness.llm_client import call_llm_json


JUDGE_SYSTEM = """你是 Paper Factor System 的质量评估专家。

你的任务：根据预设的评估标准，给 Agent 的执行结果打分（1-5 分）。
- 5 分：完全满足
- 4 分：大部分满足
- 3 分：部分满足
- 2 分：基本不满足
- 1 分：完全不满足或失败

要求：
1. 严格基于事实判断，不放过明显的缺陷
2. 给出 1-2 句具体改进建议
3. 严格返回 JSON，不要包含 markdown 代码块标记
"""

JUDGE_USER_TEMPLATE = """## 测试目标
{goal}

## 评估标准
{criteria}

## Agent 执行的轨迹摘要
{trace_summary}

## Agent 最终输出
{output}

---

请按 JSON 格式输出：
{{"scores": [{{"criterion": "标准1", "score": N, "comment": "..."}}, ...], "overall_comment": "总体评价", "pass": true/false}}

要求：
- scores 长度 = criteria 长度
- pass = 所有 score >= 3
- overall_comment 控制在 100 字内"""


def _build_trace_summary(trace: List[Dict[str, Any]]) -> str:
    """从执行轨迹构建摘要。"""
    if not trace:
        return "(no trace)"

    lines = []
    for i, step in enumerate(trace, 1):
        node = step.get("node", "?")
        reason = step.get("reason", "")[:80]
        lines.append(f"{i}. [{node}] {reason}")
    return "\n".join(lines)


def judge_output(
    goal: str,
    criteria: List[str],
    trace: List[Dict[str, Any]],
    final_output: str,
) -> Dict[str, Any]:
    """调用 LLM 评估 Agent 输出。

    Args:
        goal: 测试目标
        criteria: 评估标准列表
        trace: Agent 执行轨迹
        final_output: Agent 最终输出

    Returns:
        {"scores": [...], "overall_comment": "...", "pass": bool}
    """
    criteria_str = "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))
    trace_str = _build_trace_summary(trace)
    output_str = final_output[:1500] if final_output else "(empty)"

    user = JUDGE_USER_TEMPLATE.format(
        goal=goal,
        criteria=criteria_str,
        trace_summary=trace_str,
        output=output_str,
    )

    try:
        result = call_llm_json(JUDGE_SYSTEM, user)
        return _normalize_judge_result(result, criteria)
    except Exception as e:
        return {
            "scores": [{"criterion": c, "score": 1, "comment": f"judge failed: {e}"}
                       for c in criteria],
            "overall_comment": f"Judge LLM error: {e}",
            "pass": False,
            "error": str(e),
        }


def _normalize_judge_result(result: Dict, criteria: List[str]) -> Dict[str, Any]:
    """兜底字段：scores / overall_comment / pass。"""
    if "scores" not in result:
        result["scores"] = []
    if "overall_comment" not in result:
        result["overall_comment"] = ""
    if "pass" not in result:
        result["pass"] = all(s.get("score", 0) >= 3 for s in result["scores"])
    return result
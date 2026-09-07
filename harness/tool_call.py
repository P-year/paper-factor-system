"""
harness/tool_call.py - Closed-loop tool call wrapper

设计目标：把"工具调用"变成"verify → retry → fallback"的闭环，
保证每次调工具的结果都经过 sanity check + 自动恢复，
不让幻觉/数据错误静默传到下游。

核心函数：
- safe_tool_call() — 闭环工具调用（自动 verify / retry / fallback）
- backtest_recovery() — backtest 失败的恢复策略（扩大日期范围等）
- factor_recovery() — factor 提取失败的恢复策略（重试 LLM）

闭环流程：
    1. 调用 tool_fn(**args)
    2. 用 verifier(result) 验证
    3. 通过 → 返回（带 verified 标签）
    4. 不通过 → 调 recovery_fn(args, verify_result) 生成新参数
    5. 重试（最多 max_retries 次）
    6. 仍失败 → 返回带 low-confidence 标签的结果

所有经过此 wrapper 的结果都包含：
    _verified: bool
    _verify_issues: [str]
    _verify_confidence: "high" | "medium" | "low"
    _tool_attempts: int (调用了几次)
    _tool_name: str
"""
import logging
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def safe_tool_call(
    tool_name: str,
    args: Dict[str, Any],
    tool_fn: Callable[..., Dict[str, Any]],
    verifier: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    *,
    max_retries: int = 1,
    recovery_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    fallback_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Closed-loop tool call wrapper.

    Args:
        tool_name: 工具名（用于日志 / trace）
        args: 传给 tool_fn 的参数
        tool_fn: 可调用对象，接受 **args 返回 dict
        verifier: 验证函数，接受 result 返回 {verified, issues, confidence}
        max_retries: 验证失败时最多重试几次（不含首次）
        recovery_fn: 失败时调 (args, verify_result, result) -> new_args 或 None
                     返回 None 表示不再重试
        fallback_result: 所有重试失败后返回的结果（默认带 low-confidence 标签）

    Returns:
        一定包含 _verified / _verify_issues / _verify_confidence / _tool_attempts / _tool_name
    """
    current_args = dict(args)
    last_result: Optional[Dict[str, Any]] = None
    last_verify: Optional[Dict[str, Any]] = None
    total_attempts = 0

    for attempt in range(max_retries + 1):
        total_attempts = attempt + 1
        raised_exception = False
        raw_exception_msg = ""

        # 1. 调用工具
        try:
            result = tool_fn(**current_args)
            if not isinstance(result, dict):
                # 非 dict 直接判失败，不调 verifier（避免错误诊断）
                result = {
                    "_raw_output": str(result)[:500],
                    "_verified": False,
                    "_verify_issues": [f"结果不是 dict（type={type(result).__name__}）"],
                    "_verify_confidence": "low",
                    "_tool_attempts": total_attempts,
                    "_tool_name": tool_name,
                }
                return result
        except Exception as e:
            raised_exception = True
            raw_exception_msg = str(e)
            result = {"_error": raw_exception_msg, "status": "error"}
            logger.warning(f"[{tool_name}] attempt {total_attempts} raised: {e}")

        # 2. 验证
        if verifier is None:
            v = {"verified": True, "issues": [], "confidence": "high"}
        else:
            try:
                v = verifier(result)
            except Exception as e:
                v = {"verified": False, "issues": [f"verifier 自身异常: {e}"], "confidence": "low"}

        # 如果是 exception 触发的失败，把异常信息加进 issues
        if raised_exception and v.get("issues"):
            v["issues"] = list(v["issues"]) + [f"工具异常: {raw_exception_msg[:100]}"]
            v["verified"] = False
            v["confidence"] = "low"

        last_result = result
        last_verify = v

        # 3. 打标签（保证返回值都有这些字段）
        if isinstance(result, dict):
            result["_verified"] = v["verified"]
            result["_verify_issues"] = v.get("issues", [])
            result["_verify_confidence"] = v.get("confidence", "low")
            result["_tool_attempts"] = total_attempts
            result["_tool_name"] = tool_name

        # 4. 通过 → 立即返回
        if v["verified"]:
            if total_attempts > 1:
                logger.info(f"[{tool_name}] 重试后通过（attempt {total_attempts}）")
            return result

        # 5. 失败 → 决定是否重试
        if attempt >= max_retries:
            break  # 重试次数用完

        if recovery_fn is None:
            break  # 没有恢复策略

        # 调恢复策略生成新参数
        try:
            new_args = recovery_fn(current_args, v, result)
        except Exception as e:
            logger.warning(f"[{tool_name}] recovery_fn 异常: {e}")
            break

        if new_args is None:
            break  # 恢复策略说"放弃重试"

        if new_args == current_args:
            break  # 没变化，避免死循环

        logger.info(
            f"[{tool_name}] verify 失败，进入第 {total_attempts + 1} 次重试。"
            f"issues={v.get('issues', [])[:2]}"
        )
        current_args = new_args

    # 6. 全部失败
    if last_result is None:
        # 工具从未成功调用
        last_result = fallback_result or {
            "_error": "tool never returned a result",
            "status": "error",
        }
        last_verify = {"verified": False, "issues": ["tool call failed completely"], "confidence": "low"}

    if isinstance(last_result, dict):
        last_result["_verified"] = False
        last_result["_verify_issues"] = last_verify.get("issues", [])
        last_result["_verify_confidence"] = "low"
        last_result["_tool_attempts"] = total_attempts
        last_result["_tool_name"] = tool_name

    logger.warning(
        f"[{tool_name}] 重试 {total_attempts - 1} 次后仍失败，"
        f"issues={last_verify.get('issues', [])[:3]}"
    )
    return last_result


# ============================================================
# 业务级 recovery 策略
# ============================================================

def backtest_recovery(
    args: Dict[str, Any],
    verify_result: Dict[str, Any],
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """backtest 失败的恢复策略。

    策略：
    1. n_dates 太少 → 扩大日期范围到 5 年
    2. status=error → 网络问题重试（不改参数）
    3. IC 越界 / 方向不一致 → 不重试（属于数据/逻辑问题，重试无意义）
    """
    issues = " ".join(verify_result.get("issues", []))
    new_args = dict(args)

    # 1. 数据不足 → 拉长日期
    if "交易日数太少" in issues or "数据不足" in issues:
        new_args["start_date"] = "20180101"  # 5 年数据
        new_args["end_date"] = "20241231"
        return new_args

    # 2. status=error → 简单重试（不改参数）
    if "报错" in issues or "error" in result.get("status", ""):
        # 返回 None 等于"下次调用不改参数"
        return dict(args)  # 同参数重试

    # 3. 其他（IC越界 / 方向不一致 / 缺失字段）→ 不重试
    return None


def factor_recovery(
    args: Dict[str, Any],
    verify_result: Dict[str, Any],
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """factor 提取失败的恢复策略。

    对 LLM 调用的 retry 通常不靠改参数（prompt 已在 tool 里）。
    这里返回 None 表示"LLM 类调用不重试"，让 verify 标签自己说话。
    """
    return None


# ============================================================
# 闭环节点 wrapper（高阶）
# ============================================================

def verified_extracted_factor(
    paper: Dict[str, Any],
    *,
    max_retries: int = 0,
) -> Dict[str, Any]:
    """对单篇论文做带 verify 的因子提取。

    这是业务节点的便捷 wrapper，集成了 extract_factor_tool + verify_extracted_factor。
    """
    from agent.tools import extract_factor_tool
    from harness.verify import verify_extracted_factor

    return safe_tool_call(
        tool_name="extract_factor",
        args={"paper": paper},
        tool_fn=lambda paper: extract_factor_tool(paper),
        verifier=verify_extracted_factor,
        max_retries=max_retries,
        recovery_fn=factor_recovery,
    )


def verified_backtest(
    factor_name: str,
    *,
    universe: str = "csi500",
    start_date: str = "20230101",
    end_date: str = "20241231",
    n_stocks: int = 50,
    max_retries: int = 1,
) -> Dict[str, Any]:
    """对单个因子做带 verify 的回测。

    闭环：会自动扩大日期范围 / 重试一次。
    """
    from agent.tools import run_backtest
    from harness.verify import verify_backtest_result

    return safe_tool_call(
        tool_name="run_backtest",
        args={
            "factor_name": factor_name,
            "universe": universe,
            "start_date": start_date,
            "end_date": end_date,
            "n_stocks": n_stocks,
        },
        tool_fn=lambda **kw: run_backtest(**kw),
        verifier=verify_backtest_result,
        max_retries=max_retries,
        recovery_fn=backtest_recovery,
    )
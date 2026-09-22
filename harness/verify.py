"""
harness/verify.py - 工具输出验证器

设计目标：把"工具输出是否可信"的检查从业务节点抽出来
让所有工具调用都能共享同一套规则，避免幻觉/数据错误被静默传递到下游。

主要函数：
- verify_backtest_result() — 回测结果的 sanity check
- verify_extracted_factor() — LLM 提取的因子做基础校验
- verify_paper_collect() — 论文采集结果的合理性检查

使用示例：
    from harness.verify import verify_backtest_result

    result = run_backtest(...)
    v = verify_backtest_result(result)
    result["_verified"] = v["verified"]
    result["_verify_issues"] = v["issues"]
    result["_verify_confidence"] = v["confidence"]
"""
import math
from typing import Any, Dict, List, Optional


# ============================================================
# Backtest 结果验证
# ============================================================

# 阈值常量（统一管理）
MIN_N_DATES = 20         # 最少交易日数
MIN_N_STOCKS = 30        # 最少股票数（用于分组）
MAX_ABS_IC = 1.0         # IC 最大绝对值（理论上不会超过 1）
MAX_ABS_ICIR = 10.0      # ICIR 最大绝对值（异常大说明有问题）


def verify_backtest_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    对单次回测结果做 6 类 sanity check。

    Args:
        result: run_backtest 工具返回的 dict

    Returns:
        {
          "verified": bool,
          "issues": [str, ...],
          "confidence": "high" | "medium" | "low",
        }

    6 类检查：
    1. status 检查（ok / insufficient_data / error）
    2. 必填字段完整性
    3. IC 数值合法性（NaN / Inf / 越界）
    4. 样本量（n_dates / n_stocks）
    5. ICIR 合理性
    6. 方向一致性（IC 和 long_short 应同号）
    """
    if not isinstance(result, dict):
        return {
            "verified": False,
            "issues": [f"结果不是 dict: {type(result).__name__}"],
            "confidence": "low",
        }

    issues: List[str] = []

    # ----- 1. status 检查 -----
    status = result.get("status", "")
    if status == "error":
        issues.append(f"回测报错: {result.get('error', 'unknown')[:80]}")
    elif status == "insufficient_data":
        issues.append("数据不足，回测未完成")
    elif status and status != "ok":
        issues.append(f"未知状态: {status}")

    # ----- 2. 必填字段完整性 -----
    required = ["IC", "ICIR", "n_dates"]
    missing = [f for f in required if f not in result]
    if missing and status != "error":
        issues.append(f"缺失字段: {missing}")

    # ----- 3. IC 数值合法性 -----
    ic = result.get("IC")
    if ic is not None:
        if isinstance(ic, (int, float)):
            if math.isnan(ic) or math.isinf(ic):
                issues.append(f"IC={ic} 不是有限数（NaN/Inf）")
            elif abs(ic) > MAX_ABS_IC:
                issues.append(f"IC={ic:.4f} 超出 [-1, 1] 范围")
        else:
            issues.append(f"IC 类型异常: {type(ic).__name__}")

    # ----- 4. 样本量 -----
    n_dates = result.get("n_dates")
    if isinstance(n_dates, (int, float)):
        if n_dates < MIN_N_DATES:
            issues.append(f"交易日数太少: {n_dates} < {MIN_N_DATES}")

    # ICIR 合理性（仅在 IC 正常时检查）
    icir = result.get("ICIR")
    if icir is not None and isinstance(icir, (int, float)):
        if math.isnan(icir) or math.isinf(icir):
            issues.append(f"ICIR={icir} 不是有限数")
        elif abs(icir) > MAX_ABS_ICIR:
            issues.append(f"ICIR={icir:.2f} 异常大（>{MAX_ABS_ICIR}）")

    # ----- 5. 方向一致性（IC 和 long_short 应同号）-----
    long_short = result.get("long_short")
    if (ic is not None and long_short is not None
            and isinstance(ic, (int, float)) and isinstance(long_short, (int, float))
            and not math.isnan(ic) and not math.isnan(long_short)
            and not math.isinf(ic) and not math.isinf(long_short)):
        ic_sign = 1 if ic > 0 else (-1 if ic < 0 else 0)
        ls_sign = 1 if long_short > 0 else (-1 if long_short < 0 else 0)
        if ic_sign != 0 and ls_sign != 0 and ic_sign != ls_sign:
            issues.append(
                f"方向不一致: IC={ic:.4f} ({'+' if ic > 0 else '-'}), "
                f"long_short={long_short:.4f} ({'+' if long_short > 0 else '-'})"
            )

    # ----- 6. 综合判定 -----
    if not issues:
        return {"verified": True, "issues": [], "confidence": "high"}

    # 有 issues 时根据严重程度给 confidence
    has_critical = any(
        kw in " ".join(issues)
        for kw in ["报错", "NaN", "Inf", "超出", "方向不一致", "缺失字段", "不是有限数"]
    )
    confidence = "low" if has_critical else "medium"
    return {"verified": False, "issues": issues, "confidence": confidence}


# ============================================================
# LLM 提取的因子验证
# ============================================================

GENERIC_FACTOR_NAMES = {
    "alpha", "factor", "signal", "策略", "方法", "模型", "指标",
    "score", "ratio", "value", "factor1", "test", "demo", "xxx",
}

INCOMPLETE_FORMULA_TOKENS = ["...", "etc", "类似", "等", "tbd", "todo"]


def verify_extracted_factor(factor: Dict[str, Any]) -> Dict[str, Any]:
    """
    对 LLM 提取的因子做基础校验。

    注意：这是和 harness/badcase/analyzer.py 互补的——badcase_analyzer
    只跑事后分析；这里在节点执行时就实时打 verified 标志。

    Returns:
        {
          "verified": bool,
          "issues": [str, ...],
          "confidence": "high" | "medium" | "low",
        }
    """
    if not isinstance(factor, dict):
        return {
            "verified": False,
            "issues": [f"factor 不是 dict: {type(factor).__name__}"],
            "confidence": "low",
        }

    # 非因子论文直接通过
    if not factor.get("is_factor", True):
        return {"verified": True, "issues": [], "confidence": "high"}

    issues: List[str] = []

    # 必填字段
    for field in ["factor_name", "definition", "data_needed"]:
        val = factor.get(field, "")
        if not val or (isinstance(val, str) and not val.strip()):
            issues.append(f"缺失必填字段: {field}")

    # 因子名不能太宽泛
    name = (factor.get("factor_name") or "").strip()
    if name and name.lower() in GENERIC_FACTOR_NAMES:
        issues.append(f"因子名 {name!r} 太宽泛")
    if name and len(name) <= 1:
        issues.append(f"因子名 {name!r} 长度过短（≤1 字符）")

    # 公式不能空泛
    formula = (factor.get("calculation_formula") or "").strip()
    if formula:
        for tok in INCOMPLETE_FORMULA_TOKENS:
            if tok in formula:
                issues.append(f"公式包含偷懒写法 {tok!r}")
                break
    elif factor.get("definition"):
        issues.append("有定义但公式为空")

    # 过度自信检查
    confidence = (factor.get("confidence") or "").strip()
    conclusion = (factor.get("paper_conclusion") or "").strip()
    if confidence == "高" and len(conclusion) < 20:
        issues.append(
            f"置信度=高但 paper_conclusion 仅 {len(conclusion)} 字符"
        )

    if not issues:
        return {"verified": True, "issues": [], "confidence": "high"}

    has_critical = any(
        kw in " ".join(issues)
        for kw in ["缺失必填字段", "太宽泛", "长度过短", "置信度=高"]
    )
    return {
        "verified": False,
        "issues": issues,
        "confidence": "low" if has_critical else "medium",
    }


# ============================================================
# 论文采集验证
# ============================================================

def verify_paper_collect(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    对 collect_node 返回的论文列表做合理性检查。

    Returns:
        {
          "verified": bool,
          "issues": [str, ...],
          "n_papers": int,
        }
    """
    issues: List[str] = []
    if not papers:
        return {"verified": False, "issues": ["未采集到任何论文"], "n_papers": 0}

    for i, p in enumerate(papers):
        if not isinstance(p, dict):
            issues.append(f"第 {i} 个 paper 不是 dict")
            continue
        if not p.get("title"):
            issues.append(f"第 {i} 个 paper 缺标题")
        if not p.get("summary") and not p.get("abstract"):
            issues.append(f"第 {i} 个 paper 缺摘要")

    return {
        "verified": not issues,
        "issues": issues[:5],  # 只返回前 5 个，避免太长
        "n_papers": len(papers),
    }
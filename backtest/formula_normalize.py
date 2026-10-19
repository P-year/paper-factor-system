"""
backtest/formula_normalize.py - 自然语言公式 → DSL 归一化

启发式优先 + LLM 兜底，把论文中的自然语言公式转换为受限 DSL 表达式。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FormulaSpec:
    """归一化后的公式规格。"""
    expression: str = ""
    inputs: List[str] = field(default_factory=list)
    higher_is_better: bool = True
    notes: str = ""
    source_text: str = ""
    method: str = ""       # heuristic | llm | builtin | manual
    error: str = ""


# 关键词映射（中文金融术语 → DSL）
_KEYWORD_MAP = {
    # 收益/动量
    "近1月收益": "pct_change(close, 21)",
    "近3月收益": "pct_change(close, 63)",
    "近6月收益": "pct_change(close, 126)",
    "近1年收益": "pct_change(close, 252)",
    "1月动量": "pct_change(close, 21)",
    "3月动量": "pct_change(close, 63)",
    "6月动量": "pct_change(close, 126)",
    "12月动量": "pct_change(close, 252)",
    "短期反转": "-pct_change(close, 5)",
    "1月反转": "-pct_change(close, 21)",
    "动量因子": "pct_change(close, 21)",
    "反转因子": "-pct_change(close, 21)",
    # 风险/波动
    "波动率": "-rolling_std(returns, 21)",
    "波动因子": "-rolling_std(returns, 21)",
    "风险因子": "-rolling_std(returns, 63)",
    "下行风险": "-rolling_std(returns[returns < 0], 21)",
    "21日波动率": "-rolling_std(returns, 21)",
    # 流动性/交易
    "换手率": "volume",
    "成交量": "volume",
    "流动性": "volume",
    "成交额": "volume",
    # 规模（具体在前，通用在后）
    "对数市值": "log(market_cap)",
    "小市值": "-log(market_cap)",
    "市值": "market_cap",
    "规模": "market_cap",
    # 质量
    "收益率": "returns",
    "价格": "close",
    "收盘价": "close",
    "相对价格": "close / rolling_mean(close, 252)",
    # 截面
    "截面排名": "cs_rank(returns)",
    "截面动量": "cs_rank(pct_change(close, 21))",
    "相对强弱": "cs_rank(pct_change(close, 21))",
    "市场中性": "cs_zscore(returns)",
    "标准化": "cs_zscore(returns)",
    # 组合/加权
    "等权": "cs_rank(returns)",
    "市值加权": "market_cap * returns",
    "风险加权": "returns / rolling_std(returns, 21)",
    # 时序
    "时序动量": "ts_rank(pct_change(close, 21), 252)",
    "时序均值": "rolling_mean(close, 21)",
    "时序标准差": "rolling_std(returns, 21)",
    "均值回归": "-ts_zscore(close, 21)",
    # 收益率计算
    "超额收益": "returns - rolling_mean(returns, 252)",
    "累计收益": "rolling_sum(returns, 21)",
    "对数收益": "log(close / delay(close, 1))",
    # 其他
    "偏度": "rolling_mean(returns ** 3, 21)",
    "峰度": "rolling_mean(returns ** 4, 21)",
    "信息比": "rolling_mean(returns, 21) / rolling_std(returns, 21)",
    "夏普": "rolling_mean(returns, 21) / (rolling_std(returns, 21) + 0.001)",
}


def normalize_formula(factor: Dict[str, Any], *, use_llm: bool = False) -> FormulaSpec:
    """归一化因子公式为 DSL 表达式。

    优先级: 已有 DSL → 启发式关键词 → LLM（可选）
    """
    spec = FormulaSpec(
        source_text=factor.get("calculation_formula", ""),
        higher_is_better=True,
    )

    formula_text = spec.source_text.strip()
    if not formula_text:
        spec.notes = "empty formula"
        return spec

    # 1. 尝试直接当 DSL 解析（必须含有效函数调用或输入面板名）
    from backtest.formula_compiler import FormulaCompiler, FormulaError
    compiler = FormulaCompiler()
    try:
        compiler.parse(formula_text)
        inputs = compiler.extract_inputs(formula_text)
        # 必须至少有一个有效输入才认为是 DSL
        if inputs:
            spec.expression = formula_text
            spec.inputs = inputs
            spec.method = "manual"
            spec.notes = "already valid DSL"
            return spec
    except FormulaError:
        pass

    # 2. 启发式关键词匹配
    expr = _heuristic_normalize(formula_text, factor)
    if expr:
        try:
            compiler.parse(expr)
            spec.expression = expr
            spec.inputs = compiler.extract_inputs(expr)
            spec.method = "heuristic"
            spec.notes = "keyword mapping"
            return spec
        except FormulaError:
            pass

    # 3. LLM 兜底
    if use_llm:
        expr = _llm_normalize(formula_text, factor)
        if expr:
            try:
                compiler.parse(expr)
                spec.expression = expr
                spec.inputs = compiler.extract_inputs(expr)
                spec.method = "llm"
                spec.notes = "LLM normalized"
                return spec
            except FormulaError as e:
                spec.error = str(e)

    spec.error = "unable to normalize"
    spec.notes = "no valid DSL expression"
    return spec


def _heuristic_normalize(text: str, factor: Dict) -> Optional[str]:
    """启发式关键词匹配。"""
    # 直接关键词匹配
    for key, expr in _KEYWORD_MAP.items():
        if key in text:
            return expr

    # 组合模式: "X 的 Y 日收益率"
    m = re.search(r"(\d+)\s*(?:日|天|期)\s*收益", text)
    if m:
        n = int(m.group(1))
        return f"pct_change(close, {n})"

    # "N 日动量"
    m = re.search(r"(\d+)\s*(?:日|月|天)\s*动量", text)
    if m:
        n = int(m.group(1))
        if n <= 30:
            days = n
        else:
            days = n * 21  # 月 → 日
        return f"pct_change(close, {days})"

    # "N 日波动率" / "N 日标准差" / "N 日风险"
    m = re.search(r"(\d+)\s*(?:日|天|周)\s*(?:波动率|标准差|波动|风险)", text)
    if m:
        n = int(m.group(1))
        return f"rolling_std(returns, {n})"

    # "N 日均值" / "N 日均线"
    m = re.search(r"(\d+)\s*(?:日|天)\s*(?:均值|均线|平均)", text)
    if m:
        n = int(m.group(1))
        return f"rolling_mean(close, {n})"

    # "N 日收益率" 简写
    m = re.search(r"(\d+)\s*(?:日|天|周)\s*收益率?", text)
    if m:
        n = int(m.group(1))
        return f"pct_change(close, {n})"

    # "排名" / "排序" / "截面"
    if "排名" in text or "排序" in text or "截面" in text:
        return "cs_rank(returns)"

    # "对数"
    if "对数" in text or "log" in text.lower():
        return "log(close)"

    # "超额" / "基准"
    if "超额" in text or "基准" in text:
        return "returns - rolling_mean(returns, 252)"

    # "均值回归" / "回归"
    if "均值回归" in text or "回归" in text:
        return "-ts_zscore(close, 21)"

    # "流动性" / "活跃度"
    if "流动" in text or "活跃" in text:
        return "volume"

    # "规模" / "大小"
    if "规模" in text or "市值" in text or "大小" in text:
        return "log(market_cap)"

    return None


# LLM 归一化结果缓存（避免重复调用）
_normalize_cache: Dict[str, str] = {}


def _llm_normalize(text: str, factor: Dict) -> Optional[str]:
    """LLM 归一化（需要 API key）。带缓存。"""
    # 缓存查找
    cache_key = text.strip()[:200]
    if cache_key in _normalize_cache:
        return _normalize_cache[cache_key]

    try:
        from config import FORMULA_NORMALIZE_PROMPT
    except ImportError:
        return None

    try:
        from harness.llm_client import call_llm_json
    except ImportError:
        return None

    prompt = FORMULA_NORMALIZE_PROMPT.format(
        formula=text,
        definition=factor.get("definition", ""),
        data_needed=factor.get("data_needed", ""),
    )
    try:
        result = call_llm_json(system="你是量化公式转换专家", user=prompt, timeout=20)
        if result and isinstance(result, dict):
            expr = result.get("expression", "")
            if expr:
                _normalize_cache[cache_key] = expr
                return expr
    except Exception:
        pass

    _normalize_cache[cache_key] = ""
    return None

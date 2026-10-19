"""
backtest/formula_compiler.py - 安全的公式 DSL 编译器

将受限 DSL 表达式编译为 Callable[[MarketData], pd.DataFrame]。
AST 白名单验证，不 exec 自由文本。

DSL 示例:
    pct_change(close, 21)
    - rolling_std(returns, 21)
    cs_rank(returns) - cs_rank(volume)
"""
from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.types import FactorPanel, MarketData


class FormulaError(Exception):
    """公式编译/执行错误。"""


# === DSL 函数实现（DataFrame 级操作）===

def _pct_change(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return df.pct_change(int(n))

def _rolling_mean(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return df.rolling(window=int(n), min_periods=max(1, int(n) // 2)).mean()

def _rolling_std(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return df.rolling(window=int(n), min_periods=max(1, int(n) // 2)).std()

def _rolling_sum(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return df.rolling(window=int(n), min_periods=max(1, int(n) // 2)).sum()

def _delay(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return df.shift(int(n))

def _ts_rank(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """时序排名（当前值在过去 N 期中的分位数）。"""
    return df.rolling(window=int(n), min_periods=max(1, int(n) // 2)).rank(pct=True)

def _ts_zscore(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    m = df.rolling(window=int(n), min_periods=max(1, int(n) // 2)).mean()
    s = df.rolling(window=int(n), min_periods=max(1, int(n) // 2)).std()
    return (df - m) / (s + 1e-12)

def _ts_delta(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return df - df.shift(int(n))

def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面排名（每行独立 rank，归一化到 [0,1]）。"""
    return df.rank(axis=1, pct=True)

def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """截面 z-score（每行标准化）。"""
    m = df.mean(axis=1)
    s = df.std(axis=1)
    return df.sub(m, axis=0).div(s + 1e-12, axis=0)

def _winsorize(df: pd.DataFrame, q: float = 0.01) -> pd.DataFrame:
    """截面 winsorize（上下分位数截断）。"""
    lo = df.quantile(q, axis=1)
    hi = df.quantile(1 - q, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)

def _log(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df.abs() + 1e-12)

def _abs(df: pd.DataFrame) -> pd.DataFrame:
    return df.abs()

def _sign(df: pd.DataFrame) -> pd.DataFrame:
    return np.sign(df)

def _clip(df: pd.DataFrame, lo: float = -3.0, hi: float = 3.0) -> pd.DataFrame:
    return df.clip(lower=float(lo), upper=float(hi))

def _where(cond: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.where(cond.values, a.values, b.values),
                        index=cond.index, columns=cond.columns)


# === 允许的函数名 → Python callable ===
ALLOWED_FUNCS: Dict[str, Callable] = {
    "pct_change": _pct_change,
    "rolling_mean": _rolling_mean,
    "rolling_std": _rolling_std,
    "rolling_sum": _rolling_sum,
    "delay": _delay,
    "ts_rank": _ts_rank,
    "ts_zscore": _ts_zscore,
    "ts_delta": _ts_delta,
    "cs_rank": _cs_rank,
    "cs_zscore": _cs_zscore,
    "winsorize": _winsorize,
    "log": _log,
    "abs": _abs,
    "sign": _sign,
    "clip": _clip,
    "where": _where,
}

# 允许的输入面板名
ALLOWED_INPUTS = {"close", "prices", "open", "high", "low", "volume", "returns", "market_cap"}

# 允许的 AST 节点
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name,
    ast.Load, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Pow, ast.USub, ast.UAdd, ast.Tuple, ast.keyword,
)


class FormulaCompiler:
    """安全的公式编译器。"""

    def parse(self, expression: str) -> ast.AST:
        """解析并验证表达式（AST 白名单）。"""
        if not expression or not expression.strip():
            raise FormulaError("empty expression")
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as e:
            raise FormulaError(f"syntax error: {e}") from e

        # 验证所有节点类型
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                raise FormulaError(
                    f"forbidden node: {type(node).__name__}"
                )
            # 检查函数名
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in ALLOWED_FUNCS:
                        raise FormulaError(f"unknown function: {node.func.id}")
                else:
                    raise FormulaError("only direct function calls allowed")

        return tree

    def compile(self, expression: str) -> Callable[[MarketData], pd.DataFrame]:
        """编译表达式为 panel 函数。"""
        tree = self.parse(expression)

        def evaluator(data: MarketData) -> pd.DataFrame:
            namespace = self._build_namespace(data)
            try:
                result = eval(compile(tree, "<formula>", "eval"),
                             {"__builtins__": {}}, namespace)
            except Exception as e:
                raise FormulaError(f"evaluation error: {e}") from e

            if not isinstance(result, pd.DataFrame):
                raise FormulaError(f"expected DataFrame, got {type(result).__name__}")
            return result

        return evaluator

    def compute(self, expression: str, data: MarketData, *,
                name: str = "", **meta) -> FactorPanel:
        """编译并计算 FactorPanel。"""
        fn = self.compile(expression)
        values = fn(data)
        return FactorPanel(
            name=name or expression[:50],
            values=values,
            higher_is_better=meta.get("higher_is_better", True),
            category=meta.get("category", "custom"),
            formula=expression,
            meta=meta,
        )

    def extract_inputs(self, expression: str) -> List[str]:
        """提取表达式中引用的输入面板名。"""
        tree = self.parse(expression)
        inputs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in ALLOWED_INPUTS:
                inputs.add(node.id)
        return sorted(inputs)

    def validate_inputs(self, expression: str, data: MarketData) -> List[str]:
        """检查表达式需要的输入面板是否可用。返回缺失列表。"""
        inputs = self.extract_inputs(expression)
        missing = []
        for inp in inputs:
            attr = self._input_attr(inp)
            val = getattr(data, attr, None)
            if val is None or (isinstance(val, pd.DataFrame) and val.empty):
                missing.append(inp)
        return missing

    def _build_namespace(self, data: MarketData) -> Dict[str, Any]:
        ns = dict(ALLOWED_FUNCS)
        ns["close"] = data.prices
        ns["prices"] = data.prices
        ns["returns"] = data.returns
        ns["volume"] = data.volume if data.volume is not None else pd.DataFrame(0.0, index=data.prices.index, columns=data.prices.columns)
        ns["market_cap"] = data.market_cap if data.market_cap is not None else pd.DataFrame(1.0, index=data.prices.index, columns=data.prices.columns)
        ns["open"] = data.prices
        ns["high"] = data.prices
        ns["low"] = data.prices
        return ns

    @staticmethod
    def _input_attr(name: str) -> str:
        return {"close": "prices", "prices": "prices", "open": "prices",
                "high": "prices", "low": "prices"}.get(name, name)

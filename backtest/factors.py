"""
backtest/factors.py - 因子计算 + registry

复用 core/data_fetcher.FACTOR_CONFIGS 的因子元信息。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from backtest.types import FactorPanel, MarketData

# 默认 registry（来自 core/data_fetcher.FACTOR_CONFIGS 的名称/分类）
_DEFAULT_FACTOR_META: Dict[str, Dict[str, Any]] = {
    "return_1m": {"category": "momentum", "period": 21, "higher_is_better": True, "formula": "pct_change(21)"},
    "return_3m": {"category": "momentum", "period": 63, "higher_is_better": True, "formula": "pct_change(63)"},
    "return_6m": {"category": "momentum", "period": 126, "higher_is_better": True, "formula": "pct_change(126)"},
    "return_1y": {"category": "momentum", "period": 252, "higher_is_better": True, "formula": "pct_change(252)"},
    "turnover_1m": {"category": "momentum", "period": 21, "higher_is_better": False, "formula": "mean(volume)"},
}


class FactorCalculator:
    """因子计算器。"""

    # 全局 registry
    _registry: Dict[str, Callable] = {}

    def __init__(self, data: MarketData):
        self.data = data

    def compute(self, factor_name: str, **overrides) -> FactorPanel:
        """按 registry 分发计算。"""
        if factor_name in self._registry:
            fn = self._registry[factor_name]
            return fn(self.data, overrides)

        # 内置价格因子
        meta = _DEFAULT_FACTOR_META.get(factor_name, {})
        period = overrides.get("period", meta.get("period", 21))

        if factor_name.startswith("return_"):
            return self._compute_price_momentum(factor_name, period)
        elif factor_name.startswith("volatility_"):
            return self._compute_volatility(factor_name, period)
        elif factor_name.startswith("turnover_"):
            return self._compute_turnover(factor_name, period)

        # 默认：返回全零因子（测试用）
        return FactorPanel(
            name=factor_name,
            values=pd.DataFrame(0.0, index=self.data.prices.index, columns=self.data.prices.columns),
            category="custom",
        )

    def _compute_price_momentum(self, name: str, period: int) -> FactorPanel:
        values = self.data.prices.pct_change(period)
        return FactorPanel(
            name=name,
            values=values,
            higher_is_better=True,
            category="momentum",
            formula=f"pct_change({period})",
        )

    def _compute_volatility(self, name: str, period: int) -> FactorPanel:
        values = self.data.returns.rolling(window=period).std()
        return FactorPanel(
            name=name,
            values=values,
            higher_is_better=False,
            category="risk",
            formula=f"rolling({period}).std()",
        )

    def _compute_turnover(self, name: str, period: int) -> FactorPanel:
        if self.data.volume is not None:
            values = self.data.volume.rolling(window=period).mean()
        else:
            values = pd.DataFrame(0.0, index=self.data.prices.index, columns=self.data.prices.columns)
        return FactorPanel(
            name=name,
            values=values,
            higher_is_better=False,
            category="liquidity",
            formula=f"volume.rolling({period}).mean()",
        )

    def compute_from_series(
        self,
        name: str,
        values: pd.DataFrame,
        **meta,
    ) -> FactorPanel:
        """从预构建的 DataFrame 创建 FactorPanel（自定义/LLM 提取因子）。"""
        return FactorPanel(
            name=name,
            values=values,
            higher_is_better=meta.get("higher_is_better", True),
            category=meta.get("category", "custom"),
            formula=meta.get("formula", ""),
            meta=meta,
        )

    def compute_from_formula(self, name: str, expression: str, **meta) -> FactorPanel:
        """从 DSL 表达式编译并计算 FactorPanel。"""
        from backtest.formula_compiler import FormulaCompiler
        return FormulaCompiler().compute(expression, self.data, name=name, **meta)

    @classmethod
    def register(cls, name: str, fn: Callable[[MarketData, Dict], FactorPanel], config: Optional[Dict] = None):
        """注册自定义因子计算函数。"""
        cls._registry[name] = fn
        if config:
            _DEFAULT_FACTOR_META[name] = config

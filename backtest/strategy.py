"""
backtest/strategy.py - 策略引擎

StrategyConfig + WeightingScheme(3种) + StrategyEngine
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.types import FactorPanel, MarketData, StrategyResult
from backtest.metrics import compute_performance


@dataclass
class StrategyConfig:
    """策略配置。"""
    strategy_name: str = "factor_top_quantile"
    factor_name: str = ""
    weighting: str = "equal"           # equal | market_cap | risk_parity
    rebalance_freq: str = "monthly"    # daily | weekly | monthly | quarterly
    n_groups: int = 5
    top_group: int = 5                 # 做多最高分位
    long_short: bool = False           # True=多空；False=纯多头
    bottom_group: int = 1              # long_short 时做空组
    max_weight: float = 0.10           # 单票权重上限
    transaction_cost_bps: float = 10.0 # 单边 10bp
    slippage_bps: float = 5.0
    initial_capital: float = 1_000_000.0
    periods_per_year: int = 252

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "factor_name": self.factor_name,
            "weighting": self.weighting,
            "rebalance_freq": self.rebalance_freq,
            "n_groups": self.n_groups,
            "top_group": self.top_group,
            "long_short": self.long_short,
            "bottom_group": self.bottom_group,
            "max_weight": self.max_weight,
            "transaction_cost_bps": self.transaction_cost_bps,
            "slippage_bps": self.slippage_bps,
            "initial_capital": self.initial_capital,
        }


# === WeightingScheme ===

class WeightingScheme(ABC):
    """权重分配方案。"""

    @abstractmethod
    def weight(self, scores: pd.Series, caps: Optional[pd.Series] = None,
               vols: Optional[pd.Series] = None) -> pd.Series:
        """输入选中股票的因子分，输出权重（和为 1）。"""


class EqualWeight(WeightingScheme):
    """等权。"""
    def weight(self, scores, caps=None, vols=None):
        n = len(scores)
        return pd.Series(1.0 / n, index=scores.index)


class MarketCapWeight(WeightingScheme):
    """市值加权。"""
    def weight(self, scores, caps=None, vols=None):
        if caps is None or caps.sum() == 0:
            # 降级等权
            n = len(scores)
            return pd.Series(1.0 / n, index=scores.index)
        w = caps / caps.sum()
        return w


class RiskParity(WeightingScheme):
    """逆波动率近似风险平价。"""
    def weight(self, scores, caps=None, vols=None):
        if vols is None or (vols == 0).all():
            n = len(scores)
            return pd.Series(1.0 / n, index=scores.index)
        inv_vol = 1.0 / (vols + 1e-8)
        return inv_vol / inv_vol.sum()


def get_weighting(name: str) -> WeightingScheme:
    mapping = {"equal": EqualWeight, "market_cap": MarketCapWeight, "risk_parity": RiskParity}
    if name not in mapping:
        raise ValueError(f"unknown weighting: {name!r}")
    return mapping[name]()


# === StrategyEngine ===

class StrategyEngine:
    """策略回测引擎。"""

    def __init__(self, data: MarketData):
        self.data = data

    @staticmethod
    def _apply_max_weight(weights: pd.Series, max_weight: float) -> pd.Series:
        """应用单票权重上限。允许 sum < 1（剩余为现金仓位）。"""
        if max_weight <= 0:
            return weights / weights.sum() if weights.sum() > 0 else weights
        w = weights.clip(upper=max_weight)
        # 不再强制归一化到 1.0（可能有现金仓位）
        return w

    def _get_rebalance_dates(self, freq: str) -> List[int]:
        """返回再平衡日的索引位置。"""
        dates = self.data.prices.index
        if freq == "daily":
            return list(range(len(dates)))
        if freq == "weekly":
            return [i for i in range(len(dates)) if i == 0 or dates[i].week != dates[i-1].week]
        if freq == "monthly":
            return [i for i in range(len(dates)) if i == 0 or dates[i].month != dates[i-1].month]
        if freq == "quarterly":
            return [i for i in range(len(dates)) if i == 0 or dates[i].quarter != dates[i-1].quarter]
        # fallback: monthly
        return [i for i in range(len(dates)) if i == 0 or dates[i].month != dates[i-1].month]

    def run(self, factor: FactorPanel, config: StrategyConfig) -> StrategyResult:
        """执行策略回测。"""
        prices = self.data.prices
        returns = self.data.returns
        codes = prices.columns
        dates = prices.index
        n = len(dates)

        # 对齐因子
        aligned = factor.align(self.data)
        factor_values = aligned.values

        # 权重方案
        scheme = get_weighting(config.weighting)

        # 再平衡日期
        rebalance_idx = self._get_rebalance_dates(config.rebalance_freq)

        # 成本
        cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 1e4

        # 持仓权重（date × code）
        weights = pd.DataFrame(0.0, index=dates, columns=codes)
        current_weights = pd.Series(0.0, index=codes)

        equity = np.ones(n)
        daily_returns = np.zeros(n)
        turnover_series = np.zeros(n)
        cost_cumulative = 0.0

        for t in range(n):
            cost = 0.0
            if t in rebalance_idx and t < n - 1:
                # 当日因子值
                f_t = factor_values.iloc[t].dropna()
                if len(f_t) < config.n_groups * 2:
                    # 样本不足，保持上期权重
                    pass
                else:
                    try:
                        groups = pd.qcut(f_t, config.n_groups, labels=False, duplicates="drop")
                    except ValueError:
                        groups = pd.Series(0, index=f_t.index)

                    # 选 top 组
                    top_g = config.n_groups - 1 if config.top_group <= 0 else config.top_group - 1
                    top_mask = groups == top_g
                    top_stocks = f_t[top_mask].index.tolist()

                    # 选 bottom 组（多空时做空）
                    bottom_mask = groups == 0
                    bottom_stocks = f_t[bottom_mask].index.tolist()

                    # 权重分配
                    new_weights = pd.Series(0.0, index=codes)

                    if top_stocks:
                        caps = None
                        if config.weighting == "market_cap" and self.data.market_cap is not None:
                            caps = self.data.market_cap.iloc[t, [list(codes).index(s) for s in top_stocks]]
                        vols = None
                        if config.weighting == "risk_parity":
                            vols = self.data.returns[top_stocks].iloc[:t+1].std() if t > 0 else pd.Series(1.0, index=top_stocks)
                        w_top = scheme.weight(f_t[top_stocks], caps, vols)
                        # max_weight 截断（迭代归一化确保约束满足）
                        w_top = self._apply_max_weight(w_top, config.max_weight)
                        new_weights[top_stocks] = w_top.values

                    if config.long_short and bottom_stocks:
                        caps = None
                        if config.weighting == "market_cap" and self.data.market_cap is not None:
                            caps = self.data.market_cap.iloc[t, [list(codes).index(s) for s in bottom_stocks]]
                        w_bot = scheme.weight(f_t[bottom_stocks], caps)
                        w_bot = self._apply_max_weight(w_bot, config.max_weight)
                        new_weights[bottom_stocks] = -w_bot.values

                    # 调仓成本
                    delta = (new_weights - current_weights).abs().sum()
                    cost = delta * cost_rate
                    cost_cumulative += cost
                    turnover_series[t] = delta / 2  # 单边换手

                    current_weights = new_weights

            # 记录权重
            weights.iloc[t] = current_weights.values

            # 计算当日收益
            if t > 0:
                r_t = (current_weights * returns.iloc[t]).sum()
                if t in rebalance_idx:
                    r_t -= cost  # 调仓日扣成本
                daily_returns[t] = r_t
                equity[t] = equity[t-1] * (1 + r_t)

        # 净值曲线
        equity_series = pd.Series(equity, index=dates)
        returns_series = pd.Series(daily_returns, index=dates)
        drawdown = equity_series / equity_series.cummax() - 1.0

        # 指标
        turnover_pd = pd.Series(turnover_series, index=dates)
        metrics = compute_performance(
            returns_series, equity_series,
            turnover_series=turnover_pd,
            cost_total=cost_cumulative,
            periods_per_year=config.periods_per_year,
        )

        # 权重历史（稀疏，只保存非零）
        weights_history = {}
        for t in rebalance_idx:
            w = weights.iloc[t]
            nonzero = w[w != 0].to_dict()
            if nonzero:
                weights_history[str(dates[t].date())] = {k: round(v, 6) for k, v in nonzero.items()}

        return StrategyResult(
            strategy_name=config.strategy_name,
            factor_name=config.factor_name or factor.name,
            config=config.to_dict(),
            dates=[str(d.date()) for d in dates],
            equity=equity_series.tolist(),
            returns=returns_series.tolist(),
            drawdown=drawdown.tolist(),
            turnover_series=turnover_pd.tolist(),
            weights_history=weights_history,
            metrics=metrics,
        )

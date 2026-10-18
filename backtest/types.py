"""
backtest/types.py - 回测系统核心数据容器

与 core/backtester.BacktestResult 字段对齐，扩展量化分析字段。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    """统一市场数据面板。

    prices:  index=date, columns=stock_code, values=close price
    returns: 日收益率 = prices.pct_change()
    """
    prices: pd.DataFrame
    returns: pd.DataFrame
    market_cap: Optional[pd.DataFrame] = None
    volume: Optional[pd.DataFrame] = None
    universe: List[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    provider: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def codes(self) -> List[str]:
        return list(self.prices.columns)

    def dates(self) -> pd.DatetimeIndex:
        return self.prices.index

    def forward_returns(self, periods: int = 21) -> pd.DataFrame:
        """未来 N 日收益 = prices.shift(-N)/prices - 1。"""
        return self.prices.shift(-periods) / self.prices - 1.0

    def to_meta_dict(self) -> Dict[str, Any]:
        """只序列化元信息，不 dump 整个 panel。"""
        return {
            "universe": self.universe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "provider": self.provider,
            "n_stocks": len(self.prices.columns),
            "n_days": len(self.prices),
            "meta": self.meta,
        }


@dataclass
class FactorPanel:
    """因子值面板 (date × stock_code)。"""
    name: str
    values: pd.DataFrame
    higher_is_better: bool = True
    category: str = ""
    formula: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def align(self, data: MarketData) -> "FactorPanel":
        """对齐到 MarketData 的日期/股票轴。"""
        dates = data.prices.index
        codes = data.prices.columns
        aligned = self.values.reindex(index=dates, columns=codes)
        return FactorPanel(
            name=self.name,
            values=aligned,
            higher_is_better=self.higher_is_better,
            category=self.category,
            formula=self.formula,
            meta=self.meta,
        )


@dataclass
class FactorEvalResult:
    """因子评价结果（扩展 core/backtester.BacktestResult 字段）。"""
    # 与 core/backtester.BacktestResult 对齐
    factor_name: str
    ic: float = 0.0
    icir: float = 0.0
    rank_ic: float = 0.0
    top_return: float = 0.0
    bottom_return: float = 0.0
    top_minus_bottom: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    t_stat: float = 0.0
    p_value: float = 0.0
    data_points: int = 0
    modality: str = "single"
    news_contribution: str = ""
    # 新增
    quantile_returns: Dict[str, float] = field(default_factory=dict)
    mean_turnover: float = 0.0
    decay: Dict[str, float] = field(default_factory=dict)
    ic_series: List[float] = field(default_factory=list)
    rank_ic_series: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    status: str = "ok"
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "ic": round(self.ic, 4),
            "icir": round(self.icir, 4),
            "rank_ic": round(self.rank_ic, 4),
            "top_return": round(self.top_return, 4),
            "bottom_return": round(self.bottom_return, 4),
            "top_minus_bottom": round(self.top_minus_bottom, 4),
            "annual_return": round(self.annual_return, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "sharpe": round(self.sharpe, 4),
            "t_stat": round(self.t_stat, 4),
            "p_value": round(self.p_value, 4),
            "data_points": self.data_points,
            "modality": self.modality,
            "news_contribution": self.news_contribution,
            "quantile_returns": {k: round(v, 4) for k, v in self.quantile_returns.items()},
            "mean_turnover": round(self.mean_turnover, 4),
            "decay": {k: round(v, 4) for k, v in self.decay.items()},
            "ic_series": [round(x, 4) for x in self.ic_series],
            "rank_ic_series": [round(x, 4) for x in self.rank_ic_series],
            "dates": self.dates,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_legacy(cls, r: Any) -> "FactorEvalResult":
        """从 core/backtester.BacktestResult 转换。"""
        return cls(
            factor_name=getattr(r, "factor_name", ""),
            ic=getattr(r, "ic", 0.0),
            icir=getattr(r, "icir", 0.0),
            rank_ic=getattr(r, "rank_ic", 0.0),
            top_return=getattr(r, "top_return", 0.0),
            bottom_return=getattr(r, "bottom_return", 0.0),
            top_minus_bottom=getattr(r, "top_minus_bottom", 0.0),
            annual_return=getattr(r, "annual_return", 0.0),
            max_drawdown=getattr(r, "max_drawdown", 0.0),
            win_rate=getattr(r, "win_rate", 0.0),
            sharpe=getattr(r, "sharpe", 0.0),
            t_stat=getattr(r, "t_stat", 0.0),
            p_value=getattr(r, "p_value", 0.0),
            data_points=getattr(r, "data_points", 0),
            modality=getattr(r, "modality", "single"),
            news_contribution=getattr(r, "news_contribution", ""),
        )

    def to_legacy(self) -> Any:
        """转回 core/backtester.BacktestResult。"""
        from core.backtester import BacktestResult
        return BacktestResult(
            factor_name=self.factor_name,
            ic=self.ic, icir=self.icir, rank_ic=self.rank_ic,
            top_return=self.top_return, bottom_return=self.bottom_return,
            top_minus_bottom=self.top_minus_bottom,
            annual_return=self.annual_return, max_drawdown=self.max_drawdown,
            win_rate=self.win_rate, sharpe=self.sharpe,
            t_stat=self.t_stat, p_value=self.p_value,
            data_points=self.data_points,
            modality=self.modality, news_contribution=self.news_contribution,
        )


@dataclass
class PerformanceMetrics:
    """风险绩效指标。"""
    annual_return: float = 0.0
    annual_vol: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    turnover_mean: float = 0.0
    cost_total: float = 0.0
    monthly_returns: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annual_return": round(self.annual_return, 4),
            "annual_vol": round(self.annual_vol, 4),
            "sharpe": round(self.sharpe, 4),
            "sortino": round(self.sortino, 4),
            "calmar": round(self.calmar, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "max_drawdown_duration": self.max_drawdown_duration,
            "win_rate": round(self.win_rate, 4),
            "total_return": round(self.total_return, 4),
            "turnover_mean": round(self.turnover_mean, 4),
            "cost_total": round(self.cost_total, 4),
            "monthly_returns": {k: round(v, 4) for k, v in self.monthly_returns.items()},
        }


@dataclass
class StrategyResult:
    """策略回测结果。"""
    strategy_name: str
    factor_name: str
    config: Dict[str, Any] = field(default_factory=dict)
    dates: List[str] = field(default_factory=list)
    equity: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    drawdown: List[float] = field(default_factory=list)
    turnover_series: List[float] = field(default_factory=list)
    weights_history: Optional[Dict[str, Dict[str, float]]] = None
    metrics: Optional[PerformanceMetrics] = None
    benchmark_equity: Optional[List[float]] = None
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "factor_name": self.factor_name,
            "config": self.config,
            "dates": self.dates,
            "equity": [round(x, 6) for x in self.equity],
            "returns": [round(x, 6) for x in self.returns],
            "drawdown": [round(x, 6) for x in self.drawdown],
            "turnover_series": [round(x, 4) for x in self.turnover_series],
            "metrics": self.metrics.to_dict() if self.metrics else {},
            "benchmark_equity": [round(x, 6) for x in self.benchmark_equity] if self.benchmark_equity else None,
            "status": self.status,
        }

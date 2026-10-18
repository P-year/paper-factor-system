"""
backtest - 因子 + 策略回测系统

模块：
- types:     数据容器 (MarketData, FactorPanel, FactorEvalResult, ...)
- providers: 可插拔数据层 (Synthetic, AKShare, DB)
- factors:   因子计算 (FactorCalculator + registry)
- ic:        IC/ICIR/分组收益/衰减/换手率
- strategy:  策略引擎 (组合构建/再平衡/成本)
- metrics:   风险绩效指标 (Sharpe/Sortino/Calmar)
- report:    JSON + HTML 报告
- pipeline:  一键编排
"""
from backtest.types import MarketData, FactorPanel, FactorEvalResult
from backtest.providers import DataProvider, SyntheticProvider, get_provider
from backtest.ic import ic_series, ic_summary, quantile_returns, factor_decay, factor_turnover

__all__ = [
    "MarketData", "FactorPanel", "FactorEvalResult",
    "DataProvider", "SyntheticProvider", "get_provider",
    "ic_series", "ic_summary", "quantile_returns", "factor_decay", "factor_turnover",
]

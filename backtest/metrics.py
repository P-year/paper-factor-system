"""
backtest/metrics.py - 风险绩效指标（纯函数）

252 交易日，百分比用小数表示（报告层格式化为 %）。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def annual_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """年化收益率。"""
    if len(returns) == 0:
        return 0.0
    total = (1 + returns).prod()
    n_periods = len(returns)
    if n_periods == 0 or total <= 0:
        return 0.0
    return float(total ** (periods_per_year / n_periods) - 1)


def annual_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """年化波动率。"""
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> Tuple[float, int]:
    """最大回撤和持续时间。

    Returns:
        (max_drawdown < 0, duration in trading days)
    """
    if len(equity) == 0:
        return 0.0, 0
    cummax = equity.cummax()
    drawdown = equity / cummax - 1.0
    mdd = float(drawdown.min())
    # 持续时间：从峰值到恢复的天数
    dd_series = drawdown
    peak_idx = dd_series.idxmin()
    if isinstance(peak_idx, int):
        peak_pos = peak_idx
    else:
        peak_pos = dd_series.index.get_loc(peak_idx)
    # 找到从 peak_pos 往后第一次恢复到 0 的位置
    after_peak = dd_series.iloc[peak_pos:]
    recovery = after_peak[after_peak >= 0]
    if len(recovery) > 0:
        duration = len(after_peak.iloc[:recovery.index[0] if isinstance(recovery.index[0], int) else after_peak.index.get_loc(recovery.index[0])]) if False else len(after_peak) - len(after_peak[after_peak >= 0]) + 1
    else:
        duration = len(after_peak)
    duration = max(0, int(duration))
    return mdd, duration


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """夏普比率。"""
    excess = returns - rf / periods_per_year
    vol = excess.std()
    if vol == 0 or np.isnan(vol) or abs(vol) < 1e-12:
        return 0.0
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """索提诺比率（下行波动）。"""
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float(excess.mean() * np.sqrt(periods_per_year) / 1e-8) if excess.mean() > 0 else 0.0
    downside_vol = downside.std()
    if downside_vol == 0:
        return 0.0
    return float(excess.mean() / downside_vol * np.sqrt(periods_per_year))


def calmar_ratio(returns: pd.Series, equity: pd.Series, periods_per_year: int = 252) -> float:
    """卡玛比率 = 年化收益 / |最大回撤|。"""
    ann = annual_return(returns, periods_per_year)
    mdd, _ = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(ann / abs(mdd))


def win_rate(returns: pd.Series) -> float:
    """日度正收益占比。"""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def monthly_returns(returns: pd.Series) -> Dict[str, float]:
    """月度收益。key="YYYY-MM"。"""
    if len(returns) == 0:
        return {}
    result = {}
    for period, group in returns.groupby(returns.index.to_period("M")):
        cum = (1 + group).prod() - 1
        result[str(period)] = float(cum)
    return result


def monthly_return_matrix(returns: pd.Series) -> pd.DataFrame:
    """月度收益矩阵 (year × month)，用于热力图。"""
    if len(returns) == 0:
        return pd.DataFrame()
    mr = monthly_returns(returns)
    data = {}
    for key, val in mr.items():
        year, month = key.split("-")
        data.setdefault(year, {})[int(month)] = val
    df = pd.DataFrame(data).T.fillna(0.0)
    df.columns = [f"{int(c):02d}" for c in df.columns]
    return df


def rolling_performance(
    returns: pd.Series,
    window: int = 63,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """滚动窗口绩效。"""
    if len(returns) < window:
        return pd.DataFrame()
    ann_ret = returns.rolling(window).apply(lambda x: annual_return(x, periods_per_year), raw=False)
    rolling_vol = returns.rolling(window).std() * np.sqrt(periods_per_year)
    rolling_sharpe = returns.rolling(window).mean() / (returns.rolling(window).std() + 1e-8) * np.sqrt(periods_per_year)
    return pd.DataFrame({
        "rolling_ann_return": ann_ret,
        "rolling_vol": rolling_vol,
        "rolling_sharpe": rolling_sharpe,
    })


def compute_performance(
    returns: pd.Series,
    equity: pd.Series,
    turnover_series: Optional[pd.Series] = None,
    cost_total: float = 0.0,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> "PerformanceMetrics":
    """计算全套绩效指标。"""
    from backtest.types import PerformanceMetrics

    ann = annual_return(returns, periods_per_year)
    vol = annual_volatility(returns, periods_per_year)
    sharpe = sharpe_ratio(returns, rf, periods_per_year)
    sortino = sortino_ratio(returns, rf, periods_per_year)
    mdd, duration = max_drawdown(equity)
    calmar = calmar_ratio(returns, equity, periods_per_year)
    wr = win_rate(returns)
    total = float((1 + returns).prod() - 1) if len(returns) > 0 else 0.0
    mr = monthly_returns(returns)
    turnover_mean = float(turnover_series.mean()) if turnover_series is not None and len(turnover_series) > 0 else 0.0

    return PerformanceMetrics(
        annual_return=ann,
        annual_vol=vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=mdd,
        max_drawdown_duration=duration,
        win_rate=wr,
        total_return=total,
        turnover_mean=turnover_mean,
        cost_total=cost_total,
        monthly_returns=mr,
    )

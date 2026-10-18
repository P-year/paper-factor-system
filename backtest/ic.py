"""
backtest/ic.py - IC/ICIR/RankIC/分组收益/衰减/换手率

纯函数，输入 pandas DataFrame，输出 Series/Dict。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def ic_series(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    method: str = "pearson",
) -> pd.Series:
    """逐日截面 IC。

    Args:
        factor: date × code 因子值
        fwd_ret: date × code 未来收益
        method: "pearson" (IC) 或 "spearman" (RankIC)

    Returns:
        pd.Series, index=dates, values=IC
    """
    aligned_factor, aligned_ret = factor.align(fwd_ret, join="inner")
    if len(aligned_factor) == 0:
        return pd.Series(dtype=float)

    ics = []
    for date in aligned_factor.index:
        f = aligned_factor.loc[date].dropna()
        r = aligned_ret.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 3:
            ics.append(np.nan)
            continue
        f_vals = f[common].values
        r_vals = r[common].values
        if method == "spearman":
            f_rank = pd.Series(f_vals).rank().values
            r_rank = pd.Series(r_vals).rank().values
            corr = np.corrcoef(f_rank, r_rank)[0, 1]
        else:
            if np.std(f_vals) == 0 or np.std(r_vals) == 0:
                corr = np.nan
            else:
                corr = np.corrcoef(f_vals, r_vals)[0, 1]
        ics.append(corr)

    return pd.Series(ics, index=aligned_factor.index, name="ic")


def rank_ic_series(factor: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    """Rank IC（Spearman）序列。"""
    s = ic_series(factor, fwd_ret, method="spearman")
    s.name = "rank_ic"
    return s


def ic_summary(ic: pd.Series) -> Dict[str, float]:
    """IC 统计摘要。

    Returns:
        {"mean_ic", "icir", "t_stat", "p_value", "positive_ratio", "data_points"}
    """
    valid = ic.dropna()
    n = len(valid)
    if n == 0:
        return {"mean_ic": 0.0, "icir": 0.0, "t_stat": 0.0, "p_value": 1.0,
                "positive_ratio": 0.0, "data_points": 0}

    mean_ic = float(valid.mean())
    std_ic = float(valid.std())
    icir = mean_ic / std_ic if std_ic > 0 else 0.0

    # t 统计量
    if n > 1 and std_ic > 0:
        t_stat = mean_ic / (std_ic / np.sqrt(n))
    else:
        t_stat = 0.0

    # 近似 p 值（双尾）
    from math import erfc, sqrt
    if n > 1:
        p_value = erfc(abs(t_stat) / sqrt(2))
    else:
        p_value = 1.0

    positive_ratio = float((valid > 0).sum() / n) if n > 0 else 0.0

    return {
        "mean_ic": mean_ic,
        "icir": icir,
        "t_stat": t_stat,
        "p_value": p_value,
        "positive_ratio": positive_ratio,
        "data_points": n,
    }


def quantile_returns(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_groups: int = 5,
) -> pd.DataFrame:
    """分组未来收益。

    Args:
        factor: date × code 因子值
        fwd_ret: date × code 未来收益
        n_groups: 分位数数量

    Returns:
        pd.DataFrame, index=dates, columns=Q1..Qn, values=组平均未来收益
    """
    aligned_factor, aligned_ret = factor.align(fwd_ret, join="inner")
    if len(aligned_factor) == 0:
        return pd.DataFrame()

    groups_data = {f"Q{i+1}": [] for i in range(n_groups)}
    dates = []

    for date in aligned_factor.index:
        f = aligned_factor.loc[date].dropna()
        r = aligned_ret.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < n_groups * 2:
            continue
        f_common = f[common]
        r_common = r[common]
        # 按因子值分组
        try:
            groups = pd.qcut(f_common, n_groups, labels=False, duplicates="drop")
        except ValueError:
            continue
        dates.append(date)
        for g in range(n_groups):
            mask = groups == g
            if mask.any():
                groups_data[f"Q{g+1}"].append(float(r_common[mask].mean()))
            else:
                groups_data[f"Q{g+1}"].append(0.0)

    return pd.DataFrame(groups_data, index=pd.DatetimeIndex(dates))


def quantile_summary(qret: pd.DataFrame) -> Dict[str, float]:
    """分组收益摘要。"""
    if qret.empty:
        return {}
    out = {}
    for col in qret.columns:
        out[col] = float(qret[col].mean())
    if len(qret.columns) >= 2:
        top = qret.iloc[:, -1]
        bottom = qret.iloc[:, 0]
        out["top_minus_bottom"] = float((top - bottom).mean())
    return out


def long_short_curve(
    qret: pd.DataFrame,
    top: int = -1,
    bottom: int = 0,
) -> pd.Series:
    """多空日收益序列。

    Args:
        qret: 分组收益 DataFrame
        top: 做多组（默认最后一列 = 最高分位）
        bottom: 做空组（默认第一列 = 最低分位）
    """
    if qret.empty or len(qret.columns) < 2:
        return pd.Series(dtype=float)
    top_col = qret.columns[top] if top < 0 else qret.columns[min(top, len(qret.columns) - 1)]
    bottom_col = qret.columns[bottom]
    return qret[top_col] - qret[bottom_col]


def factor_decay(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 10, 21, 63),
) -> Dict[str, float]:
    """因子衰减：不同持有期的 mean IC。"""
    decay = {}
    for h in horizons:
        fwd = prices.shift(-h) / prices - 1.0
        ic = ic_series(factor, fwd)
        valid = ic.dropna()
        decay[str(h)] = float(valid.mean()) if len(valid) > 0 else 0.0
    return decay


def factor_turnover(
    factor: pd.DataFrame,
    n_groups: int = 5,
    top_group: int = -1,
) -> pd.Series:
    """Top 组合换手率时间序列。

    换手率 = 1 - (当期 top 组 ∩ 上期 top 组) / top 组大小
    """
    if factor.empty:
        return pd.Series(dtype=float)

    turnovers = [0.0]
    prev_top: set = set()

    for i in range(len(factor)):
        f = factor.iloc[i].dropna()
        if len(f) < n_groups * 2:
            turnovers.append(0.0)
            continue
        try:
            groups = pd.qcut(f, n_groups, labels=False, duplicates="drop")
        except ValueError:
            turnovers.append(0.0)
            continue

        top_g = n_groups - 1 if top_group < 0 else top_group
        top_stocks = set(f[groups == top_g].index)

        if prev_top:
            overlap = len(top_stocks & prev_top)
            turnover = 1.0 - overlap / max(len(top_stocks), 1)
        else:
            turnover = 1.0

        turnovers.append(turnover)
        prev_top = top_stocks

    return pd.Series(turnovers[1:], index=factor.index, name="turnover")


def rolling_ic(ic: pd.Series, window: int = 63) -> pd.Series:
    """滚动 IC 均值。"""
    return ic.rolling(window=window, min_periods=max(5, window // 3)).mean()

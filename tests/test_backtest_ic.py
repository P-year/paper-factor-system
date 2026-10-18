"""
test_backtest_ic.py - Phase A: IC/分组/衰减/换手率测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from backtest.ic import (
    ic_series, ic_summary, rank_ic_series,
    quantile_returns, quantile_summary, long_short_curve,
    factor_decay, factor_turnover, rolling_ic,
)


def _make_toy_data(n_days=20, n_stocks=10, seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    codes = [f"S{i}" for i in range(n_stocks)]
    factor = pd.DataFrame(np.random.randn(n_days, n_stocks), index=dates, columns=codes)
    fwd_ret = pd.DataFrame(np.random.randn(n_days, n_stocks) * 0.01, index=dates, columns=codes)
    return factor, fwd_ret, dates, codes


# === ic_series ===

def test_ic_perfect_correlation():
    """因子与未来收益完全正相关 → IC ≈ 1。"""
    dates = pd.bdate_range("2023-01-02", periods=5)
    codes = ["A", "B", "C", "D", "E"]
    factor = pd.DataFrame(np.arange(25).reshape(5, 5), index=dates, columns=codes, dtype=float)
    fwd_ret = factor * 0.01  # 完全线性
    ic = ic_series(factor, fwd_ret)
    valid = ic.dropna()
    assert (valid > 0.99).all()


def test_ic_zero_correlation():
    """因子与未来收益不相关 → IC ≈ 0。"""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", periods=50)
    codes = [f"S{i}" for i in range(10)]
    factor = pd.DataFrame(np.random.randn(50, 10), index=dates, columns=codes)
    fwd_ret = pd.DataFrame(np.random.randn(50, 10), index=dates, columns=codes)
    ic = ic_series(factor, fwd_ret)
    valid = ic.dropna()
    assert abs(valid.mean()) < 0.3  # 随机噪声 IC 接近 0


def test_ic_summary():
    factor, fwd_ret, dates, codes = _make_toy_data()
    ic = ic_series(factor, fwd_ret)
    summary = ic_summary(ic)
    assert "mean_ic" in summary
    assert "icir" in summary
    assert "t_stat" in summary
    assert "data_points" in summary
    assert summary["data_points"] > 0


def test_ic_summary_empty():
    empty_ic = pd.Series(dtype=float)
    summary = ic_summary(empty_ic)
    assert summary["mean_ic"] == 0.0
    assert summary["data_points"] == 0


# === rank_ic_series ===

def test_rank_ic_invariant_to_monotone_transform():
    """单调变换不改变 RankIC。"""
    factor, fwd_ret, dates, codes = _make_toy_data()
    ic1 = rank_ic_series(factor, fwd_ret)
    # 单调变换（log 只对正数，用平方）
    factor_transformed = factor ** 3
    ic2 = rank_ic_series(factor_transformed, fwd_ret)
    # Rank 应该相同（或接近）
    common = ic1.index.intersection(ic2.index)
    if len(common) > 0:
        assert abs(ic1[common].mean() - ic2[common].mean()) < 0.2


# === quantile_returns ===

def test_quantile_returns_shape():
    factor, fwd_ret, dates, codes = _make_toy_data(n_days=30, n_stocks=10)
    qret = quantile_returns(factor, fwd_ret, n_groups=5)
    assert len(qret.columns) == 5
    assert list(qret.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"]


def test_quantile_summary():
    factor, fwd_ret, dates, codes = _make_toy_data(n_days=30, n_stocks=10)
    qret = quantile_returns(factor, fwd_ret, n_groups=5)
    summary = quantile_summary(qret)
    assert "Q1" in summary
    assert "Q5" in summary
    assert "top_minus_bottom" in summary


def test_long_short_curve():
    factor, fwd_ret, dates, codes = _make_toy_data(n_days=30, n_stocks=10)
    qret = quantile_returns(factor, fwd_ret, n_groups=5)
    ls = long_short_curve(qret)
    assert len(ls) == len(qret)
    # 多空 = Q5 - Q1
    expected = qret["Q5"] - qret["Q1"]
    pd.testing.assert_series_equal(ls, expected, check_names=False)


# === factor_decay ===

def test_factor_decay():
    factor, fwd_ret, dates, codes = _make_toy_data(n_days=30, n_stocks=10)
    prices = (fwd_ret + 1).cumprod() * 100  # 构造价格
    decay = factor_decay(factor, prices, horizons=(1, 5, 10))
    assert set(decay.keys()) == {"1", "5", "10"}
    for v in decay.values():
        assert isinstance(v, float)


# === factor_turnover ===

def test_factor_turnover_range():
    factor, fwd_ret, dates, codes = _make_toy_data(n_days=30, n_stocks=10)
    turnover = factor_turnover(factor, n_groups=5)
    valid = turnover.dropna()
    assert len(valid) > 0
    assert (valid >= 0).all()
    assert (valid <= 2).all()


def test_factor_turnover_empty():
    empty_factor = pd.DataFrame()
    turnover = factor_turnover(empty_factor)
    assert len(turnover) == 0


# === rolling_ic ===

def test_rolling_ic():
    factor, fwd_ret, dates, codes = _make_toy_data(n_days=30, n_stocks=10)
    ic = ic_series(factor, fwd_ret)
    rolling = rolling_ic(ic, window=5)
    assert len(rolling) == len(ic)
    assert rolling.dropna().notna().any()

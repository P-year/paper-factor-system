"""
test_backtest_metrics.py - Phase B: 风险绩效指标测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    annual_return, annual_volatility, max_drawdown, sharpe_ratio,
    sortino_ratio, calmar_ratio, win_rate, monthly_returns,
    monthly_return_matrix, rolling_performance, compute_performance,
)


def _make_returns(n=252, mean=0.001, std=0.01, seed=42):
    np.random.seed(seed)
    return pd.Series(np.random.randn(n) * std + mean,
                     index=pd.bdate_range("2023-01-02", periods=n))


# === annual_return ===

def test_annual_return_positive():
    r = _make_returns(mean=0.001)
    result = annual_return(r)
    assert result > 0


def test_annual_return_zero():
    r = pd.Series([0.0] * 252)
    assert annual_return(r) == pytest.approx(0.0)


def test_annual_return_single_period():
    r = pd.Series([0.1])
    result = annual_return(r)
    assert result > 0


# === annual_volatility ===

def test_annual_volatility():
    r = _make_returns()
    vol = annual_volatility(r)
    assert vol > 0


def test_annual_volatility_constant():
    r = pd.Series([0.001] * 100)
    assert annual_volatility(r) == pytest.approx(0.0)


# === max_drawdown ===

def test_max_drawdown_positive_curve():
    equity = pd.Series([100, 110, 120, 130, 140])
    mdd, _ = max_drawdown(equity)
    assert mdd == 0.0  # 单调上升，无回撤


def test_max_drawdown_negative():
    equity = pd.Series([100, 120, 80, 140, 60])
    mdd, duration = max_drawdown(equity)
    assert mdd < 0
    assert mdd == pytest.approx(60/140 - 1, rel=1e-2)


def test_max_drawdown_empty():
    mdd, duration = max_drawdown(pd.Series(dtype=float))
    assert mdd == 0.0
    assert duration == 0


# === sharpe_ratio ===

def test_sharpe_positive():
    r = _make_returns(mean=0.001, std=0.005)
    sharpe = sharpe_ratio(r)
    assert sharpe > 0


def test_sharpe_zero_vol():
    r = pd.Series([0.001] * 100)
    assert sharpe_ratio(r) == 0.0


# === sortino_ratio ===

def test_sortino_positive():
    r = _make_returns(mean=0.001, std=0.005)
    sortino = sortino_ratio(r)
    assert sortino > 0


# === calmar_ratio ===

def test_calmar():
    r = _make_returns()
    equity = (1 + r).cumprod()
    calmar = calmar_ratio(r, equity)
    assert isinstance(calmar, float)


# === win_rate ===

def test_win_rate():
    r = pd.Series([0.01, -0.005, 0.02, -0.01, 0.005])
    assert win_rate(r) == pytest.approx(3/5)


# === monthly_returns ===

def test_monthly_returns_keys():
    r = _make_returns(n=252)
    mr = monthly_returns(r)
    assert len(mr) > 0
    for key in mr.keys():
        assert len(key) == 7  # "YYYY-MM"
        assert "-" in key


def test_monthly_returns_empty():
    assert monthly_returns(pd.Series(dtype=float)) == {}


# === monthly_return_matrix ===

def test_monthly_return_matrix():
    r = _make_returns(n=252)
    matrix = monthly_return_matrix(r)
    assert not matrix.empty
    assert matrix.shape[1] >= 1  # 至少一个月


# === rolling_performance ===

def test_rolling_performance():
    r = _make_returns(n=252)
    rp = rolling_performance(r, window=63)
    assert not rp.empty
    assert "rolling_sharpe" in rp.columns
    assert "rolling_ann_return" in rp.columns


# === compute_performance ===

def test_compute_performance():
    r = _make_returns(n=252)
    equity = (1 + r).cumprod()
    pm = compute_performance(r, equity)
    assert pm.annual_return != 0
    assert pm.annual_vol > 0
    assert isinstance(pm.monthly_returns, dict)
    assert pm.max_drawdown <= 0

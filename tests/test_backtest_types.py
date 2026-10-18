"""
test_backtest_types.py - Phase A: 回测数据容器测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from backtest.types import MarketData, FactorPanel, FactorEvalResult, PerformanceMetrics, StrategyResult


# === MarketData ===

def test_marketdata_basic():
    dates = pd.bdate_range("2023-01-02", periods=10)
    codes = ["A", "B", "C"]
    prices = pd.DataFrame(np.random.randn(10, 3).cumsum(axis=0) + 100, index=dates, columns=codes)
    returns = prices.pct_change().fillna(0)
    md = MarketData(prices=prices, returns=returns, universe=codes, provider="synthetic")
    assert md.codes() == codes
    assert len(md.dates()) == 10
    assert md.provider == "synthetic"


def test_marketdata_forward_returns():
    dates = pd.bdate_range("2023-01-02", periods=10)
    prices = pd.DataFrame({"A": range(100, 110)}, index=dates)
    returns = prices.pct_change().fillna(0)
    md = MarketData(prices=prices, returns=returns)
    fwd = md.forward_returns(periods=2)
    assert fwd.iloc[0, 0] == pytest.approx(2/100, rel=1e-6)
    assert fwd.iloc[-1, 0] == 0 or np.isnan(fwd.iloc[-1, 0])


def test_marketdata_meta_dict():
    dates = pd.bdate_range("2023-01-02", periods=5)
    prices = pd.DataFrame({"A": [1,2,3,4,5]}, index=dates)
    md = MarketData(prices=prices, returns=prices.pct_change().fillna(0), universe=["A"], provider="synthetic")
    meta = md.to_meta_dict()
    assert meta["n_stocks"] == 1
    assert meta["n_days"] == 5
    assert meta["provider"] == "synthetic"


# === FactorPanel ===

def test_factorpanel_align():
    dates = pd.bdate_range("2023-01-02", periods=5)
    prices = pd.DataFrame({"A": [1,2,3,4,5], "B": [5,4,3,2,1]}, index=dates)
    md = MarketData(prices=prices, returns=prices.pct_change().fillna(0))
    # 因子只覆盖部分日期/股票
    partial_dates = dates[:3]
    fp = FactorPanel(name="test", values=pd.DataFrame({"A": [0.1,0.2,0.3]}, index=partial_dates))
    aligned = fp.align(md)
    assert aligned.values.shape == (5, 2)
    # reindex 后缺失值为 NaN（不是 0）
    assert pd.isna(aligned.values.iloc[3, 0])


# === FactorEvalResult ===

def test_factorevalresult_to_dict():
    r = FactorEvalResult(
        factor_name="test", ic=0.05, icir=0.8, rank_ic=0.04,
        top_return=12.5, bottom_return=3.2, top_minus_bottom=9.3,
        annual_return=10.0, max_drawdown=-5.0, win_rate=0.6,
        sharpe=1.2, t_stat=2.5, p_value=0.01, data_points=252,
    )
    d = r.to_dict()
    assert d["factor_name"] == "test"
    assert d["ic"] == 0.05
    assert d["icir"] == 0.8
    assert d["data_points"] == 252
    assert isinstance(d["quantile_returns"], dict)
    assert isinstance(d["ic_series"], list)


def test_factorevalresult_legacy_roundtrip():
    """from_legacy / to_legacy 保持字段。"""
    r = FactorEvalResult(
        factor_name="momentum", ic=0.06, icir=1.0, rank_ic=0.05,
        top_return=15.0, bottom_return=5.0, top_minus_bottom=10.0,
        annual_return=12.0, max_drawdown=-8.0, win_rate=0.55,
        sharpe=1.5, t_stat=3.0, p_value=0.002, data_points=252,
    )
    legacy = r.to_legacy()
    restored = FactorEvalResult.from_legacy(legacy)
    assert restored.factor_name == "momentum"
    assert restored.ic == pytest.approx(0.06)
    assert restored.icir == pytest.approx(1.0)
    assert restored.data_points == 252


def test_factorevalresult_status():
    r = FactorEvalResult(factor_name="x", status="insufficient_data", error="too few samples")
    assert r.status == "insufficient_data"
    assert r.error == "too few samples"


# === PerformanceMetrics ===

def test_performance_metrics_to_dict():
    pm = PerformanceMetrics(
        annual_return=0.12, annual_vol=0.18, sharpe=0.67,
        sortino=0.85, calmar=1.5, max_drawdown=-0.08,
        max_drawdown_duration=45, win_rate=0.55, total_return=0.25,
        turnover_mean=0.3, cost_total=0.02,
        monthly_returns={"2023-01": 0.01, "2023-02": -0.02},
    )
    d = pm.to_dict()
    assert d["sharpe"] == 0.67
    assert "2023-01" in d["monthly_returns"]


# === StrategyResult ===

def test_strategy_result_to_dict():
    sr = StrategyResult(
        strategy_name="factor_top", factor_name="momentum",
        dates=["2023-01-02", "2023-01-03"],
        equity=[1.0, 1.01],
        returns=[0.0, 0.01],
        drawdown=[0.0, -0.01],
        turnover_series=[0.0, 0.1],
    )
    d = sr.to_dict()
    assert d["strategy_name"] == "factor_top"
    assert d["equity"] == [1.0, 1.01]
    assert d["status"] == "ok"

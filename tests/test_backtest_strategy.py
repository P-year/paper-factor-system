"""
test_backtest_strategy.py - Phase B: 策略引擎测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from backtest.types import FactorPanel, MarketData
from backtest.strategy import (
    StrategyConfig, EqualWeight, MarketCapWeight, RiskParity,
    get_weighting, StrategyEngine,
)


def _make_toy_data(n_days=60, n_stocks=10, seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    codes = [f"S{i}" for i in range(n_stocks)]
    prices = pd.DataFrame(np.random.randn(n_days, n_stocks).cumsum(axis=0) + 100, index=dates, columns=codes)
    returns = prices.pct_change().fillna(0)
    md = MarketData(prices=prices, returns=returns, universe=codes, provider="synthetic")
    factor_values = pd.DataFrame(np.random.randn(n_days, n_stocks), index=dates, columns=codes)
    factor = FactorPanel(name="test_factor", values=factor_values)
    return md, factor


# === WeightingScheme ===

def test_equal_weight():
    scores = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    w = EqualWeight().weight(scores)
    assert w.sum() == pytest.approx(1.0)
    assert all(abs(v - 1/3) < 1e-8 for v in w.values)


def test_market_cap_weight():
    scores = pd.Series({"A": 0.5, "B": 0.3})
    caps = pd.Series({"A": 300, "B": 700})
    w = MarketCapWeight().weight(scores, caps)
    assert w.sum() == pytest.approx(1.0)
    assert w["A"] == pytest.approx(0.3)
    assert w["B"] == pytest.approx(0.7)


def test_market_cap_weight_fallback():
    """无市值数据时降级等权。"""
    scores = pd.Series({"A": 0.5, "B": 0.3})
    w = MarketCapWeight().weight(scores, caps=None)
    assert w.sum() == pytest.approx(1.0)
    assert abs(w["A"] - 0.5) < 1e-8


def test_risk_parity():
    scores = pd.Series({"A": 0.5, "B": 0.3})
    vols = pd.Series({"A": 0.02, "B": 0.01})  # B 波动小 → 权重大
    w = RiskParity().weight(scores, vols=vols)
    assert w.sum() == pytest.approx(1.0)
    assert w["B"] > w["A"]  # 低波动权重高


def test_get_weighting():
    assert isinstance(get_weighting("equal"), EqualWeight)
    assert isinstance(get_weighting("market_cap"), MarketCapWeight)
    assert isinstance(get_weighting("risk_parity"), RiskParity)
    with pytest.raises(ValueError):
        get_weighting("unknown")


# === StrategyEngine ===

def test_strategy_engine_basic():
    md, factor = _make_toy_data()
    config = StrategyConfig(factor_name="test_factor", rebalance_freq="monthly")
    engine = StrategyEngine(md)
    result = engine.run(factor, config)
    assert result.status == "ok"
    assert len(result.equity) == len(md.prices)
    assert len(result.returns) == len(md.prices)
    assert result.metrics is not None


def test_strategy_weights_sum_to_one():
    md, factor = _make_toy_data()
    config = StrategyConfig(factor_name="test_factor", rebalance_freq="monthly", max_weight=0.5)
    engine = StrategyEngine(md)
    result = engine.run(factor, config)
    # 检查权重历史中的权重和
    for date, w in (result.weights_history or {}).items():
        total = sum(abs(v) for v in w.values())
        if total > 0:
            assert total <= 1.0 + 0.01  # 允许微小误差


def test_strategy_max_weight_constraint():
    md, factor = _make_toy_data()
    config = StrategyConfig(
        factor_name="test_factor", rebalance_freq="monthly",
        max_weight=0.1, n_groups=5, top_group=5,
    )
    engine = StrategyEngine(md)
    result = engine.run(factor, config)
    for date, w in (result.weights_history or {}).items():
        for code, weight in w.items():
            assert abs(weight) <= 0.1 + 0.01  # max_weight 约束


def test_strategy_cost_reduces_return():
    """有成本时收益应低于无成本。"""
    md, factor = _make_toy_data()
    config_no_cost = StrategyConfig(factor_name="f", rebalance_freq="monthly",
                                     transaction_cost_bps=0, slippage_bps=0)
    config_with_cost = StrategyConfig(factor_name="f", rebalance_freq="monthly",
                                       transaction_cost_bps=50, slippage_bps=50)
    engine = StrategyEngine(md)
    r1 = engine.run(factor, config_no_cost)
    r2 = engine.run(factor, config_with_cost)
    assert r1.metrics.total_return >= r2.metrics.total_return


def test_strategy_equity_starts_at_one():
    md, factor = _make_toy_data()
    config = StrategyConfig(factor_name="f", rebalance_freq="monthly")
    engine = StrategyEngine(md)
    result = engine.run(factor, config)
    assert result.equity[0] == pytest.approx(1.0)


def test_strategy_long_short():
    md, factor = _make_toy_data()
    config = StrategyConfig(factor_name="f", rebalance_freq="monthly", long_short=True)
    engine = StrategyEngine(md)
    result = engine.run(factor, config)
    assert result.status == "ok"
    # 多空权重有正有负
    all_weights = []
    for w in (result.weights_history or {}).values():
        all_weights.extend(w.values())
    assert any(v < 0 for v in all_weights)  # 有空头


def test_strategy_to_dict():
    md, factor = _make_toy_data()
    config = StrategyConfig(factor_name="f", rebalance_freq="monthly")
    engine = StrategyEngine(md)
    result = engine.run(factor, config)
    d = result.to_dict()
    assert d["strategy_name"] == "factor_top_quantile"
    assert isinstance(d["equity"], list)
    assert isinstance(d["metrics"], dict)

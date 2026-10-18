"""
test_backtest_providers.py - Phase A: 数据提供者测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from backtest.providers import SyntheticProvider, get_provider, DataProvider


# === SyntheticProvider ===

def test_synthetic_deterministic():
    """同 seed 生成相同数据。"""
    p1 = SyntheticProvider(n_stocks=10, n_days=30, seed=42)
    p2 = SyntheticProvider(n_stocks=10, n_days=30, seed=42)
    prices1 = p1.get_prices(p1.get_universe())
    prices2 = p2.get_prices(p2.get_universe())
    pd.testing.assert_frame_equal(prices1, prices2)


def test_synthetic_different_seed():
    """不同 seed 生成不同数据。"""
    p1 = SyntheticProvider(n_stocks=10, n_days=30, seed=42)
    p2 = SyntheticProvider(n_stocks=10, n_days=30, seed=99)
    prices1 = p1.get_prices(p1.get_universe())
    prices2 = p2.get_prices(p2.get_universe())
    assert not prices1.equals(prices2)


def test_synthetic_shape():
    p = SyntheticProvider(n_stocks=5, n_days=100)
    md = p.load()
    assert md.prices.shape == (100, 5)
    assert md.returns.shape == (100, 5)
    assert len(md.universe) == 5


def test_synthetic_returns_computed():
    p = SyntheticProvider(n_stocks=3, n_days=20)
    md = p.load()
    # returns = prices.pct_change()
    assert md.returns.iloc[1, 0] == pytest.approx(md.prices.iloc[1, 0] / md.prices.iloc[0, 0] - 1)


def test_synthetic_market_cap():
    p = SyntheticProvider(n_stocks=5, n_days=10)
    caps = p.get_market_cap(p.get_universe())
    assert caps.shape == (10, 5)
    assert (caps > 0).all().all()


# === get_provider factory ===

def test_get_provider_synthetic():
    p = get_provider("synthetic", n_stocks=5, n_days=10, seed=42)
    assert p.name == "synthetic"
    assert isinstance(p, SyntheticProvider)


def test_get_provider_unknown():
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("unknown")


# === DataProvider abstract ===

def test_provider_is_abstract():
    with pytest.raises(TypeError):
        DataProvider()

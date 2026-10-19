"""
test_formula_compiler.py - Phase 1: 公式编译器测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from backtest.formula_compiler import FormulaCompiler, FormulaError
from backtest.formula_normalize import normalize_formula, FormulaSpec
from backtest.providers import SyntheticProvider


@pytest.fixture
def data():
    provider = SyntheticProvider(n_stocks=10, n_days=60, seed=42)
    return provider.load()


@pytest.fixture
def compiler():
    return FormulaCompiler()


# === parse ===

def test_parse_valid(compiler):
    tree = compiler.parse("pct_change(close, 21)")
    assert tree is not None


def test_parse_compound(compiler):
    compiler.parse("cs_rank(returns) - cs_rank(volume)")


def test_parse_arithmetic(compiler):
    compiler.parse("(close / rolling_sum(close, 21)) * cs_zscore(returns)")


def test_parse_empty(compiler):
    with pytest.raises(FormulaError, match="empty"):
        compiler.parse("")


def test_parse_syntax_error(compiler):
    with pytest.raises(FormulaError, match="syntax"):
        compiler.parse("pct_change(")


def test_parse_unknown_function(compiler):
    with pytest.raises(FormulaError, match="unknown function"):
        compiler.parse("eval('import os')")


def test_parse_injection_rejected(compiler):
    for evil in ["__import__('os')", "open('x')", "exec('x')", "lambda x: x", "[x for x in []"]:
        with pytest.raises(FormulaError):
            compiler.parse(evil)


def test_parse_attribute_access_rejected(compiler):
    with pytest.raises(FormulaError):
        compiler.parse("close.__class__")


# === compile ===

def test_compile_and_run(compiler, data):
    fn = compiler.compile("pct_change(close, 21)")
    result = fn(data)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == data.prices.shape


def test_compile_rolling(compiler, data):
    fn = compiler.compile("rolling_std(returns, 21)")
    result = fn(data)
    assert result.shape == data.prices.shape


def test_compile_cs_rank(compiler, data):
    fn = compiler.compile("cs_rank(returns)")
    result = fn(data)
    assert result.shape == data.prices.shape


def test_compile_arithmetic(compiler, data):
    fn = compiler.compile("returns * 2 + 1")
    result = fn(data)
    assert result.shape == data.prices.shape


# === compute ===

def test_compute_factor_panel(compiler, data):
    panel = compiler.compute("pct_change(close, 21)", data, name="mom_21")
    assert panel.name == "mom_21"
    assert panel.values.shape == data.prices.shape
    assert panel.formula == "pct_change(close, 21)"


# === extract_inputs ===

def test_extract_inputs_single(compiler):
    assert compiler.extract_inputs("pct_change(close, 21)") == ["close"]

def test_extract_inputs_multiple(compiler):
    inputs = compiler.extract_inputs("cs_rank(returns) - volume")
    assert set(inputs) == {"returns", "volume"}

def test_extract_inputs_none(compiler):
    assert compiler.extract_inputs("rolling_mean(close, 5)") == ["close"]


# === validate_inputs ===

def test_validate_inputs_ok(compiler, data):
    missing = compiler.validate_inputs("pct_change(close, 21)", data)
    assert missing == []


# === Formula Normalize ===

def test_normalize_already_dsl():
    spec = normalize_formula({"calculation_formula": "pct_change(close, 21)"})
    assert spec.expression == "pct_change(close, 21)"
    assert spec.method == "manual"


def test_normalize_chinese_momentum():
    spec = normalize_formula({"calculation_formula": "近1月收益"})
    assert "pct_change" in spec.expression
    assert spec.method == "heuristic"


def test_normalize_chinese_volatility():
    spec = normalize_formula({"calculation_formula": "21日波动率"})
    assert "rolling_std" in spec.expression


def test_normalize_empty_formula():
    spec = normalize_formula({"calculation_formula": ""})
    assert spec.expression == ""


def test_normalize_unknown_returns_empty():
    spec = normalize_formula({"calculation_formula": "这是一个无法理解的公式xyz123"})
    assert spec.expression == ""


def test_formula_spec_defaults():
    spec = FormulaSpec()
    assert spec.expression == ""
    assert spec.higher_is_better is True

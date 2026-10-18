"""
test_backtest_report.py - Phase C: 报告测试
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from backtest.types import FactorEvalResult, StrategyResult, PerformanceMetrics
from backtest.report import BacktestReport, chart_specs


def _make_factor_result(name="test_factor"):
    return FactorEvalResult(
        factor_name=name, ic=0.05, icir=0.8, rank_ic=0.04,
        top_return=12.5, bottom_return=3.2, top_minus_bottom=9.3,
        annual_return=10.0, max_drawdown=-5.0, win_rate=0.6,
        sharpe=1.2, t_stat=2.5, p_value=0.01, data_points=252,
        quantile_returns={"Q1": 0.01, "Q5": 0.05},
        decay={"1": 0.05, "5": 0.03},
        ic_series=[0.05, 0.03, 0.06],
        dates=["2023-01-02", "2023-01-03", "2023-01-04"],
    )


def _make_strategy_result():
    return StrategyResult(
        strategy_name="factor_top", factor_name="test_factor",
        dates=["2023-01-02", "2023-01-03"],
        equity=[1.0, 1.01],
        returns=[0.0, 0.01],
        drawdown=[0.0, -0.01],
        metrics=PerformanceMetrics(annual_return=0.12, sharpe=1.5, max_drawdown=-0.05),
    )


# === BacktestReport ===

def test_report_to_dict():
    fr = _make_factor_result()
    sr = _make_strategy_result()
    report = BacktestReport(
        title="测试报告",
        config={"provider": "synthetic"},
        factor_results=[fr],
        strategy_result=sr,
        data_meta={"provider": "synthetic", "n_stocks": 10},
    )
    d = report.to_dict()
    assert d["title"] == "测试报告"
    assert len(d["factor_results"]) == 1
    assert d["strategy_result"] is not None


def test_report_json_roundtrip(tmp_path):
    fr = _make_factor_result()
    sr = _make_strategy_result()
    report = BacktestReport(
        title="roundtrip",
        factor_results=[fr],
        strategy_result=sr,
        data_meta={"provider": "synthetic"},
    )
    path = tmp_path / "result.json"
    report.save_json(path)

    loaded = BacktestReport.load_json(path)
    assert loaded.title == "roundtrip"
    assert len(loaded.factor_results) == 1
    assert loaded.factor_results[0].factor_name == "test_factor"
    assert loaded.strategy_result is not None


def test_report_html_contains_keywords():
    fr = _make_factor_result()
    sr = _make_strategy_result()
    report = BacktestReport(title="HTML报告", factor_results=[fr], strategy_result=sr)
    html = report.to_html()
    assert "<!DOCTYPE html>" in html
    assert "策略净值" in html
    assert "echarts" in html.lower()
    assert "IC" in html
    assert "夏普" in html


def test_report_save_html(tmp_path):
    fr = _make_factor_result()
    report = BacktestReport(title="HTML", factor_results=[fr])
    path = tmp_path / "report.html"
    report.save_html(path)
    assert path.exists()
    assert len(path.read_text(encoding="utf-8")) > 100


def test_report_summary():
    fr = _make_factor_result()
    sr = _make_strategy_result()
    report = BacktestReport(title="s", factor_results=[fr], strategy_result=sr)
    summary = report.summary()
    assert summary["n_factors"] == 1
    assert "mean_ic" in summary
    assert "strategy_metrics" in summary


# === chart_specs ===

def test_chart_specs_has_charts():
    fr = _make_factor_result()
    sr = _make_strategy_result()
    report = BacktestReport(title="c", factor_results=[fr], strategy_result=sr)
    specs = chart_specs(report)
    assert len(specs) >= 3  # 净值 + 回撤 + IC
    titles = [s["title"] for s in specs]
    assert any("净值" in t for t in titles)
    assert any("IC" in t for t in titles)


def test_chart_specs_empty():
    report = BacktestReport(title="empty")
    specs = chart_specs(report)
    assert specs == []

"""
test_backtest_pipeline.py - Phase C: pipeline 端到端测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from backtest.pipeline import BacktestPipeline, PipelineConfig
from backtest.strategy import StrategyConfig


# === Pipeline e2e (synthetic) ===

def test_pipeline_basic(tmp_path):
    config = PipelineConfig(
        provider="synthetic",
        factor_names=["return_1m"],
        n_stocks=10,
        output_dir=str(tmp_path),
        save_json=True,
        save_html=True,
        seed=42,
    )
    pipeline = BacktestPipeline(config)
    report = pipeline.run()

    assert report is not None
    assert len(report.factor_results) == 1
    assert report.factor_results[0].factor_name == "return_1m"
    assert report.strategy_result is not None

    # 文件存在
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "report.html").exists()


def test_pipeline_multiple_factors(tmp_path):
    config = PipelineConfig(
        provider="synthetic",
        factor_names=["return_1m", "return_3m"],
        n_stocks=10,
        output_dir=str(tmp_path),
        seed=42,
    )
    pipeline = BacktestPipeline(config)
    report = pipeline.run()

    assert len(report.factor_results) == 2
    names = [r.factor_name for r in report.factor_results]
    assert "return_1m" in names
    assert "return_3m" in names


def test_pipeline_deterministic(tmp_path):
    """同 seed 两次运行结果一致。"""
    config1 = PipelineConfig(
        provider="synthetic", factor_names=["return_1m"],
        n_stocks=10, output_dir=str(tmp_path / "a"), seed=42,
    )
    config2 = PipelineConfig(
        provider="synthetic", factor_names=["return_1m"],
        n_stocks=10, output_dir=str(tmp_path / "b"), seed=42,
    )
    report1 = BacktestPipeline(config1).run()
    report2 = BacktestPipeline(config2).run()

    assert report1.factor_results[0].ic == pytest.approx(report2.factor_results[0].ic, rel=1e-6)
    assert report1.factor_results[0].icir == pytest.approx(report2.factor_results[0].icir, rel=1e-6)


def test_pipeline_custom_strategy(tmp_path):
    config = PipelineConfig(
        provider="synthetic",
        factor_names=["return_1m"],
        n_stocks=10,
        output_dir=str(tmp_path),
        seed=42,
        strategy=StrategyConfig(
            weighting="equal",
            rebalance_freq="weekly",
            long_short=True,
            transaction_cost_bps=20,
        ),
    )
    pipeline = BacktestPipeline(config)
    report = pipeline.run()
    assert report.strategy_result is not None
    assert report.strategy_result.config["rebalance_freq"] == "weekly"


def test_pipeline_run_factor():
    config = PipelineConfig(provider="synthetic", n_stocks=10, seed=42)
    pipeline = BacktestPipeline(config)
    fr = pipeline.run_factor("return_1m")
    assert fr.factor_name == "return_1m"
    assert fr.data_points > 0
    assert fr.status in ("ok", "insufficient_data")


def test_pipeline_run_strategy():
    config = PipelineConfig(provider="synthetic", n_stocks=10, seed=42)
    pipeline = BacktestPipeline(config)
    sr = pipeline.run_strategy("return_1m")
    assert sr.status == "ok"
    assert len(sr.equity) > 0

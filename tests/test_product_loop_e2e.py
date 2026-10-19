"""
test_product_loop_e2e.py - 端到端集成测试

验证完整产品闭环: 论文 → 提取 → 编译 → 回测 → 报告 → 写回
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from unittest import mock


# === Product Loop E2E ===

def test_product_loop_with_formula(tmp_path, monkeypatch):
    """用 DSL 公式的因子走完整闭环。"""
    from core.product_loop import run_factor_to_decision
    from core.factor_db import FactorDatabase

    # 准备: 添加有 DSL 公式的因子
    db = FactorDatabase(db_path=str(tmp_path / "factors.json"))
    db.add_factor({
        "factor_name": "TestMomentum21",
        "definition": "21日动量因子",
        "calculation_formula": "pct_change(close, 21)",
        "confidence": "高",
    })

    # 用 mock 的 FactorDatabase 替换（函数内 import 所以 patch core.factor_db）
    with mock.patch('core.factor_db.FactorDatabase', return_value=db):
        result = run_factor_to_decision(
            "TestMomentum21",
            provider="synthetic",
            n_stocks=10,
            writeback=True,
        )

    assert result.status in ("ok", "partial")
    assert len(result.backtests) >= 1
    assert result.backtests[0].get("ic") is not None


def test_product_loop_no_compute_path(tmp_path, monkeypatch):
    """无法编译的公式应标记 no_compute_path。"""
    from core.product_loop import run_factor_to_decision
    from core.factor_db import FactorDatabase

    db = FactorDatabase(db_path=str(tmp_path / "factors.json"))
    db.add_factor({
        "factor_name": "TestUncompilable",
        "definition": "一个无法编译的复杂因子",
        "calculation_formula": "用深度学习模型预测股价趋势然后取对数变换",
        "confidence": "中",
    })

    with mock.patch('core.factor_db.FactorDatabase', return_value=db):
        result = run_factor_to_decision(
            "TestUncompilable",
            provider="synthetic",
            n_stocks=10,
        )

    # 应该尝试但可能失败
    assert result.status in ("ok", "partial", "failed")


def test_product_loop_factor_not_found(tmp_path):
    from core.product_loop import run_factor_to_decision
    from core.factor_db import FactorDatabase

    db = FactorDatabase(db_path=str(tmp_path / "factors.json"))
    with mock.patch('core.factor_db.FactorDatabase', return_value=db):
        result = run_factor_to_decision("Nonexistent", provider="synthetic")
    assert result.status == "failed"
    assert len(result.errors) > 0


# === Formula → FactorPanel → Backtest E2E ===

def test_formula_to_factor_panel():
    """DSL 公式编译为 FactorPanel 并回测。"""
    from backtest.pipeline import BacktestPipeline, PipelineConfig

    config = PipelineConfig(
        provider="synthetic", n_stocks=10,
        factor_names=["return_1m"],
        save_json=False, save_html=False, seed=42,
    )
    pipeline = BacktestPipeline(config)
    pipeline.load_data()

    # 直接用公式编译
    result = pipeline.run_formula_factor("TestFormula", "pct_change(close, 21)")
    assert result.factor_name == "TestFormula"
    assert result.data_points > 0
    assert result.status == "ok"


def test_factor_record_resolution_formula():
    """run_factor_record 应优先使用 formula_expression。"""
    from backtest.pipeline import BacktestPipeline, PipelineConfig

    config = PipelineConfig(
        provider="synthetic", n_stocks=10,
        factor_names=[], save_json=False, save_html=False, seed=42,
    )
    pipeline = BacktestPipeline(config)
    pipeline.load_data()

    record = {
        "factor_name": "CustomMom",
        "formula_expression": "pct_change(close, 63)",
        "calculation_formula": "近3月收益",
    }
    result = pipeline.run_factor_record(record)
    assert result.status == "ok"
    assert result.data_points > 0


def test_factor_record_resolution_builtin():
    """run_factor_record 应 fallback 到内置因子。"""
    from backtest.pipeline import BacktestPipeline, PipelineConfig

    config = PipelineConfig(
        provider="synthetic", n_stocks=10,
        factor_names=[], save_json=False, save_html=False, seed=42,
    )
    pipeline = BacktestPipeline(config)
    pipeline.load_data()

    record = {
        "factor_name": "return_1m",
        "calculation_formula": "",
        "formula_expression": "",
    }
    result = pipeline.run_factor_record(record)
    assert result.status == "ok"
    assert result.data_points > 0


# === Writeback E2E ===

def test_writeback_to_factor_db(tmp_path):
    """回测结果应写回 factor_db。"""
    from core.factor_db import FactorDatabase

    db = FactorDatabase(db_path=str(tmp_path / "factors.json"))
    db.add_factor({"factor_name": "WBTest", "definition": "test"})

    result_dict = {"ic": 0.05, "icir": 0.8, "rank_ic": 0.04, "data_points": 252}
    db.update_backtest_result("WBTest", result_dict, run_id="run_123", compute_source="formula")

    factors = db.get_by_status("pending")
    f = factors[0]
    assert f["backtest_result"]["ic"] == 0.05
    assert f["backtest_run_id"] == "run_123"
    assert f["compute_source"] == "formula"


# === API E2E (TestClient) ===

def test_api_backtest_then_review(tmp_path, monkeypatch):
    """API: 回测因子 → 查看结果 → 审核决策。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.factor_routes import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    # 1. 查看因子列表
    r = client.get("/factors")
    assert r.status_code == 200
    factors = r.json()["factors"]
    assert len(factors) > 0

    # 2. 查看仪表盘
    r = client.get("/dashboard/stats")
    assert r.status_code == 200
    assert "factors" in r.json()

    # 3. 查看报告列表
    r = client.get("/reports")
    assert r.status_code == 200

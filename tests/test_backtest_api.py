"""
test_backtest_api.py - Phase D: 回测 API 测试
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient


def _get_client():
    """构造 FastAPI TestClient（跳过 startup 事件）。"""
    from fastapi import FastAPI
    from api.backtest_routes import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# === /backtest/providers ===

def test_list_providers():
    client = _get_client()
    r = client.get("/backtest/providers")
    assert r.status_code == 200
    assert "synthetic" in r.json()["providers"]


# === /backtest/factor ===

def test_factor_backtest():
    client = _get_client()
    r = client.post("/backtest/factor", json={
        "factor_name": "return_1m",
        "provider": "synthetic",
        "n_stocks": 10,
        "seed": 42,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["factor_name"] == "return_1m"
    assert "ic" in data
    assert "icir" in data


# === /backtest/strategy ===

def test_strategy_backtest():
    client = _get_client()
    r = client.post("/backtest/strategy", json={
        "factor_name": "return_1m",
        "provider": "synthetic",
        "n_stocks": 10,
        "seed": 42,
    })
    assert r.status_code == 200
    data = r.json()
    assert "result" in data
    assert "summary" in data
    if data["result"]:
        assert "equity" in data["result"]
        assert "metrics" in data["result"]


# === /backtest/pipeline ===

def test_pipeline_backtest():
    client = _get_client()
    r = client.post("/backtest/pipeline", json={
        "provider": "synthetic",
        "factor_names": ["return_1m"],
        "n_stocks": 10,
        "seed": 42,
    })
    assert r.status_code == 200
    data = r.json()
    assert "run_id" in data
    assert "summary" in data
    assert data["n_factors"] == 1


# === /backtest/results ===

def test_list_results_empty():
    client = _get_client()
    r = client.get("/backtest/results")
    assert r.status_code == 200
    assert "results" in r.json()


def test_get_result_not_found():
    client = _get_client()
    r = client.get("/backtest/results/nonexistent_id")
    assert r.status_code == 404


def test_get_report_not_found():
    client = _get_client()
    r = client.get("/backtest/report/nonexistent_id")
    assert r.status_code == 404


# === Error handling ===

def test_factor_backtest_invalid_provider():
    client = _get_client()
    with pytest.raises(ValueError, match="unknown provider"):
        client.post("/backtest/factor", json={
            "factor_name": "return_1m",
            "provider": "unknown_provider",
            "n_stocks": 10,
        })

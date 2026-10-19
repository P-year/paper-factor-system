"""
test_dashboard_api.py - Phase 4: 仪表盘 + 报告 API 测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient


def _client():
    from fastapi import FastAPI
    from api.factor_routes import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_list_factors():
    c = _client()
    r = c.get("/factors")
    assert r.status_code == 200
    assert "factors" in r.json()
    assert "total" in r.json()


def test_list_factors_with_status_filter():
    c = _client()
    r = c.get("/factors", params={"status": "pending"})
    assert r.status_code == 200


def test_get_factor_not_found():
    c = _client()
    r = c.get("/factors/nonexistent_factor")
    assert r.status_code == 404


def test_dashboard_stats():
    c = _client()
    r = c.get("/dashboard/stats")
    assert r.status_code == 200
    data = r.json()
    assert "factors" in data
    assert "papers" in data
    assert "backtests" in data
    assert "metrics" in data


def test_dashboard_stats_structure():
    c = _client()
    data = c.get("/dashboard/stats").json()
    assert "total" in data["factors"]
    assert "total" in data["papers"]
    assert "total_runs" in data["backtests"]


def test_list_reports():
    c = _client()
    r = c.get("/reports")
    assert r.status_code == 200
    assert "reports" in r.json()


def test_get_report_not_found():
    c = _client()
    r = c.get("/reports/nonexistent_run")
    assert r.status_code == 404


def test_decision_invalid_status():
    c = _client()
    r = c.post("/factors/nonexistent/decision", json={"decision": "bad_status"})
    assert r.status_code == 422  # validation error


def test_decision_factor_not_found():
    c = _client()
    r = c.post("/factors/nonexistent/decision", json={"decision": "approved"})
    assert r.status_code == 404

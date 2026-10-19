"""
test_factor_writeback.py - Phase 0: 因子回写测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from core.factor_db import FactorDatabase


@pytest.fixture
def db(tmp_path):
    return FactorDatabase(db_path=str(tmp_path / "factors.json"))


def _add_test_factor(db):
    return db.add_factor({
        "factor_name": "TestFactor",
        "definition": "测试因子",
        "calculation_formula": "pct_change(close, 21)",
        "confidence": "高",
    })


# === update_backtest_result ===

def test_update_backtest_result(db):
    _add_test_factor(db)
    result = {"ic": 0.06, "icir": 0.8, "rank_ic": 0.05}
    assert db.update_backtest_result("TestFactor", result, run_id="run1", compute_source="formula") is True

    factors = db.get_by_status("pending")
    assert factors[0]["backtest_result"]["ic"] == 0.06
    assert factors[0]["backtest_run_id"] == "run1"
    assert factors[0]["compute_source"] == "formula"


def test_update_backtest_missing(db):
    assert db.update_backtest_result("Nonexistent", {"ic": 0}) is False


# === update_formula ===

def test_update_formula(db):
    _add_test_factor(db)
    assert db.update_formula("TestFactor", expression="pct_change(close, 21)",
                              status="compiled", compute_source="formula") is True
    factors = db.get_by_status("pending")
    assert factors[0]["formula_expression"] == "pct_change(close, 21)"
    assert factors[0]["formula_status"] == "compiled"


def test_update_formula_with_error(db):
    _add_test_factor(db)
    db.update_formula("TestFactor", status="failed", error="cannot parse")
    factors = db.get_by_status("pending")
    assert factors[0]["formula_status"] == "failed"
    assert factors[0]["formula_error"] == "cannot parse"


# === get_by_paper ===

def test_get_by_paper(db):
    db.add_factor({"factor_name": "F1", "paper_id": "p1"})
    db.add_factor({"factor_name": "F2", "paper_id": "p1"})
    db.add_factor({"factor_name": "F3", "paper_id": "p2"})
    result = db.get_by_paper("p1")
    assert len(result) == 2


# === get_product_summary ===

def test_product_summary_empty(db):
    s = db.get_product_summary()
    assert s["total"] == 0
    assert s["with_backtest"] == 0


def test_product_summary(db):
    _add_test_factor(db)
    db.add_factor({"factor_name": "F2", "confidence": "中", "status": "approved"})
    db.update_backtest_result("TestFactor", {"ic": 0.06})

    s = db.get_product_summary()
    assert s["total"] == 2
    assert s["with_backtest"] == 1
    assert s["avg_abs_ic"] > 0
    assert "pending" in s["status"]

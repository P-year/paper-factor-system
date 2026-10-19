"""
api/factor_routes.py - 因子管理 + 仪表盘 + 报告 API

GET  /factors                     — 因子列表
GET  /factors/{name}              — 因子详情
POST /factors/{name}/backtest     — 回测单因子
POST /factors/{name}/decision     — 批准/拒绝/激活
GET  /dashboard/stats             — 仪表盘实时数据
GET  /reports                     — 报告列表
GET  /reports/{run_id}            — 报告详情
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal

router = APIRouter(tags=["factors"])


class DecisionRequest(BaseModel):
    decision: Literal["approved", "rejected", "active", "deprecated"]
    notes: str = ""


class BacktestRequest(BaseModel):
    provider: str = "synthetic"
    universe: str = "csi500"
    start_date: str = "20230101"
    end_date: str = "20241231"
    n_stocks: int = 50
    writeback: bool = True


# === Factors ===

@router.get("/factors")
async def list_factors(status: str = "", q: str = "", limit: int = 50, offset: int = 0):
    from core.factor_db import FactorDatabase
    db = FactorDatabase()
    factors = db.factors
    if status:
        factors = [f for f in factors if f.get("status") == status]
    if q:
        ql = q.lower()
        factors = [f for f in factors if ql in (f.get("factor_name", "") + f.get("definition", "")).lower()]
    return {"factors": factors[offset:offset + limit], "total": len(db.factors)}


@router.get("/factors/{name}")
async def get_factor(name: str):
    from core.factor_db import FactorDatabase
    db = FactorDatabase()
    for f in db.factors:
        if f.get("factor_name") == name:
            return f
    raise HTTPException(status_code=404, detail="factor not found")


@router.post("/factors/{name}/backtest")
async def backtest_factor(name: str, req: BacktestRequest = BacktestRequest()):
    from core.product_loop import run_factor_to_decision
    result = run_factor_to_decision(
        name,
        provider=req.provider,
        universe=req.universe,
        start_date=req.start_date,
        end_date=req.end_date,
        n_stocks=req.n_stocks,
        writeback=req.writeback,
    )
    return {
        "status": result.status,
        "backtests": result.backtests,
        "run_ids": result.run_ids,
        "errors": result.errors,
    }


@router.post("/factors/{name}/decision")
async def decide_factor(name: str, req: DecisionRequest):
    from core.factor_db import FactorDatabase
    db = FactorDatabase()
    ok = db.update_status(name, req.decision, notes=req.notes)
    if not ok:
        raise HTTPException(status_code=404, detail="factor not found")
    return {"ok": True, "factor_name": name, "new_status": req.decision}


# === Dashboard ===

@router.get("/dashboard/stats")
async def dashboard_stats():
    from core.factor_db import FactorDatabase
    from core.paper_library import PaperLibrary

    db = FactorDatabase()
    lib = PaperLibrary()
    output_dir = Path("output/backtests")

    # 扫描回测报告
    backtest_runs = []
    if output_dir.exists():
        for d in sorted(output_dir.iterdir(), reverse=True):
            if d.is_dir():
                result_file = d / "result.json"
                if result_file.exists():
                    try:
                        data = json.loads(result_file.read_text(encoding="utf-8"))
                        sr = data.get("strategy_result") or {}
                        metrics = sr.get("metrics") or {}
                        factor_results = data.get("factor_results", [])
                        avg_ic = sum(r.get("ic", 0) for r in factor_results) / max(len(factor_results), 1)
                        backtest_runs.append({
                            "run_id": d.name,
                            "title": data.get("title", ""),
                            "created_at": data.get("created_at", ""),
                            "mean_ic": round(avg_ic, 4),
                            "annual_return": metrics.get("annual_return", 0),
                            "max_drawdown": metrics.get("max_drawdown", 0),
                        })
                    except Exception:
                        pass

    factors_summary = db.get_product_summary()
    papers_summary = lib.stats()

    # 最佳指标
    best_ic = 0.0
    best_icir = 0.0
    for f in db.factors:
        bt = f.get("backtest_result", {})
        if bt.get("ic"):
            best_ic = max(best_ic, abs(bt["ic"]))
        if bt.get("icir"):
            best_icir = max(best_icir, abs(bt["icir"]))

    return {
        "factors": factors_summary,
        "papers": papers_summary,
        "backtests": {
            "total_runs": len(backtest_runs),
            "recent": backtest_runs[:5],
        },
        "metrics": {
            "best_ic": round(best_ic, 4),
            "best_icir": round(best_icir, 4),
        },
    }


# === Reports ===

@router.get("/reports")
async def list_reports(limit: int = 20):
    output_dir = Path("output/backtests")
    reports = []
    if output_dir.exists():
        for d in sorted(output_dir.iterdir(), reverse=True)[:limit]:
            if d.is_dir():
                result_file = d / "result.json"
                if result_file.exists():
                    try:
                        data = json.loads(result_file.read_text(encoding="utf-8"))
                        sr = data.get("strategy_result") or {}
                        metrics = sr.get("metrics") or {}
                        factor_results = data.get("factor_results", [])
                        avg_ic = sum(r.get("ic", 0) for r in factor_results) / max(len(factor_results), 1)
                        reports.append({
                            "run_id": d.name,
                            "title": data.get("title", ""),
                            "created_at": data.get("created_at", ""),
                            "mean_ic": round(avg_ic, 4),
                            "annual_return": metrics.get("annual_return", 0),
                            "max_drawdown": metrics.get("max_drawdown", 0),
                            "has_html": (d / "report.html").exists(),
                        })
                    except Exception:
                        pass
    return {"reports": reports}


@router.get("/reports/{run_id}")
async def get_report(run_id: str):
    result_file = Path("output/backtests") / run_id / "result.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="report not found")
    return json.loads(result_file.read_text(encoding="utf-8"))

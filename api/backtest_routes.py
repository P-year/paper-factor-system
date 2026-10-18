"""
api/backtest_routes.py - 回测系统 API 路由

POST /backtest/factor     — 因子回测
POST /backtest/strategy   — 策略回测
POST /backtest/pipeline   — 一键全跑
GET  /backtest/results    — 历史结果列表
GET  /backtest/results/{run_id} — 完整结果
GET  /backtest/report/{run_id}  — HTML 报告
GET  /backtest/providers  — 可用数据源
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/backtest", tags=["backtest"])

OUTPUT_DIR = Path("output") / "backtests"


# === Pydantic Models ===

class FactorBacktestRequest(BaseModel):
    factor_name: str = "return_1m"
    provider: str = "synthetic"
    universe: str = "csi500"
    start_date: str = "20230101"
    end_date: str = "20241231"
    n_stocks: int = 50
    forward_period: int = 21
    n_groups: int = 5
    seed: int = 42


class StrategyBacktestRequest(BaseModel):
    factor_name: str = "return_1m"
    provider: str = "synthetic"
    universe: str = "csi500"
    start_date: str = "20230101"
    end_date: str = "20241231"
    n_stocks: int = 50
    weighting: str = "equal"
    rebalance_freq: str = "monthly"
    n_groups: int = 5
    top_group: int = 5
    long_short: bool = False
    max_weight: float = 0.10
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    seed: int = 42


class PipelineRequest(BaseModel):
    provider: str = "synthetic"
    universe: str = "csi500"
    start_date: str = "20230101"
    end_date: str = "20241231"
    n_stocks: int = 50
    factor_names: List[str] = Field(default_factory=lambda: ["return_1m"])
    forward_period: int = 21
    n_groups: int = 5
    weighting: str = "equal"
    rebalance_freq: str = "monthly"
    long_short: bool = False
    transaction_cost_bps: float = 10.0
    seed: int = 42


# === Routes ===

@router.get("/providers")
async def list_providers():
    return {"providers": ["synthetic", "akshare", "db"]}


@router.post("/factor")
async def run_factor_backtest(req: FactorBacktestRequest):
    """单因子回测。"""
    from backtest.pipeline import BacktestPipeline, PipelineConfig
    from backtest.strategy import StrategyConfig

    config = PipelineConfig(
        provider=req.provider,
        universe=req.universe,
        start_date=req.start_date,
        end_date=req.end_date,
        n_stocks=req.n_stocks,
        forward_period=req.forward_period,
        n_groups=req.n_groups,
        factor_names=[req.factor_name],
        seed=req.seed,
        save_json=False,
        save_html=False,
    )
    pipeline = BacktestPipeline(config)
    result = pipeline.run_factor(req.factor_name)
    return result.to_dict()


@router.post("/strategy")
async def run_strategy_backtest(req: StrategyBacktestRequest):
    """策略回测。"""
    from backtest.pipeline import BacktestPipeline, PipelineConfig
    from backtest.strategy import StrategyConfig

    strategy_config = StrategyConfig(
        factor_name=req.factor_name,
        weighting=req.weighting,
        rebalance_freq=req.rebalance_freq,
        n_groups=req.n_groups,
        top_group=req.top_group,
        long_short=req.long_short,
        max_weight=req.max_weight,
        transaction_cost_bps=req.transaction_cost_bps,
        slippage_bps=req.slippage_bps,
    )
    config = PipelineConfig(
        provider=req.provider,
        universe=req.universe,
        start_date=req.start_date,
        end_date=req.end_date,
        n_stocks=req.n_stocks,
        factor_names=[req.factor_name],
        strategy=strategy_config,
        seed=req.seed,
        save_json=True,
        save_html=True,
    )
    pipeline = BacktestPipeline(config)
    report = pipeline.run()

    # 获取 run_id
    run_id = report._run_id() if hasattr(report, "_run_id") else "unknown"
    return {
        "run_id": run_id,
        "result": report.strategy_result.to_dict() if report.strategy_result else None,
        "summary": report.summary(),
    }


@router.post("/pipeline")
async def run_pipeline(req: PipelineRequest):
    """一键全跑（因子评价 + 策略回测 + 报告落盘）。"""
    from backtest.pipeline import BacktestPipeline, PipelineConfig
    from backtest.strategy import StrategyConfig

    strategy_config = StrategyConfig(
        factor_name=req.factor_names[0] if req.factor_names else "",
        weighting=req.weighting,
        rebalance_freq=req.rebalance_freq,
        n_groups=req.n_groups,
        long_short=req.long_short,
        transaction_cost_bps=req.transaction_cost_bps,
    )
    config = PipelineConfig(
        provider=req.provider,
        universe=req.universe,
        start_date=req.start_date,
        end_date=req.end_date,
        n_stocks=req.n_stocks,
        factor_names=req.factor_names,
        strategy=strategy_config,
        seed=req.seed,
        save_json=True,
        save_html=True,
    )
    pipeline = BacktestPipeline(config)
    report = pipeline.run()
    run_id = report._run_id() if hasattr(report, "_run_id") else "unknown"

    return {
        "run_id": run_id,
        "summary": report.summary(),
        "n_factors": len(report.factor_results),
    }


@router.get("/results")
async def list_results(limit: int = 20):
    """历史结果列表。"""
    if not OUTPUT_DIR.exists():
        return {"results": []}
    results = []
    for d in sorted(OUTPUT_DIR.iterdir(), reverse=True)[:limit]:
        if d.is_dir():
            result_file = d / "result.json"
            if result_file.exists():
                try:
                    data = json.loads(result_file.read_text(encoding="utf-8"))
                    results.append({
                        "run_id": d.name,
                        "title": data.get("title", ""),
                        "created_at": data.get("created_at", ""),
                    })
                except Exception:
                    pass
    return {"results": results}


@router.get("/results/{run_id}")
async def get_result(run_id: str):
    """完整回测结果。"""
    result_file = OUTPUT_DIR / run_id / "result.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail=f"run_id {run_id} not found")
    return json.loads(result_file.read_text(encoding="utf-8"))


@router.get("/report/{run_id}", response_class=HTMLResponse)
async def get_report(run_id: str):
    """HTML 回测报告。"""
    report_file = OUTPUT_DIR / run_id / "report.html"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail=f"report for {run_id} not found")
    return report_file.read_text(encoding="utf-8")

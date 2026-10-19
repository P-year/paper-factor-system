"""
core/product_loop.py - 产品闭环编排

论文 → 提取因子 → 公式编译 → 回测 → 报告 → 写回因子库
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LoopResult:
    paper_id: str = ""
    paper_title: str = ""
    factors: List[Dict] = field(default_factory=list)
    backtests: List[Dict] = field(default_factory=list)
    run_ids: List[str] = field(default_factory=list)
    status: str = "ok"           # ok | partial | failed
    errors: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0


def run_paper_to_decision(
    paper_id: str,
    *,
    provider: str = "synthetic",
    universe: str = "csi500",
    start_date: str = "20230101",
    end_date: str = "20241231",
    n_stocks: int = 50,
    writeback: bool = True,
    save_report: bool = True,
) -> LoopResult:
    """完整产品闭环: 提取 → 编译 → 回测 → 写回。"""
    import time
    t0 = time.time()
    result = LoopResult(paper_id=paper_id)

    # 1. 获取论文
    from core.paper_library import PaperLibrary
    lib = PaperLibrary()
    paper = lib.get(paper_id)
    if not paper:
        result.status = "failed"
        result.errors.append(f"paper not found: {paper_id}")
        return result
    result.paper_title = paper.get("title", "")

    # 2. 提取因子（或获取已有）
    from core.factor_db import FactorDatabase
    db = FactorDatabase()
    existing = db.get_by_paper(paper_id)
    if existing:
        factors = existing
    else:
        try:
            from core.analyzer import FactorAnalyzer
            analyzer = FactorAnalyzer()
            extract_result = analyzer.extract_factor(paper)
            if extract_result.is_factor:
                fd = extract_result.factor_data
                fd["paper_id"] = paper_id
                db.add_factor(fd)
                factors = [fd]
            else:
                factors = []
        except Exception as e:
            result.errors.append(f"extraction failed: {e}")
            factors = []
    result.factors = factors

    # 3. 回测每个因子
    if factors:
        try:
            from backtest.pipeline import BacktestPipeline, PipelineConfig
            config = PipelineConfig(
                provider=provider,
                universe=universe,
                start_date=start_date,
                end_date=end_date,
                n_stocks=n_stocks,
                factor_names=[],
                save_json=save_report,
                save_html=save_report,
            )
            pipeline = BacktestPipeline(config)
            pipeline.load_data()

            for factor in factors:
                fname = factor.get("factor_name", "")
                try:
                    bt_result = pipeline.run_factor_record(factor)
                    bt_dict = bt_result.to_dict()
                    result.backtests.append(bt_dict)

                    if save_report and bt_result.status == "ok":
                        from backtest.report import BacktestReport
                        report = BacktestReport(
                            title=f"回测: {fname}",
                            factor_results=[bt_result],
                            data_meta=pipeline.data.to_meta_dict() if pipeline.data else {},
                        )
                        json_path = report.save_json()
                        run_id = json_path.parent.name
                        result.run_ids.append(run_id)

                    if writeback and bt_result.status == "ok":
                        db.update_backtest_result(
                            fname, bt_dict,
                            run_id=result.run_ids[-1] if result.run_ids else "",
                            compute_source=bt_result.compute_source or "formula",
                        )
                except Exception as e:
                    result.errors.append(f"backtest {fname}: {e}")
        except Exception as e:
            result.errors.append(f"pipeline: {e}")
            result.status = "failed"

    # 4. 更新论文状态
    lib.set_factor_count(paper_id, len(factors))

    result.elapsed_s = round(time.time() - t0, 2)
    if result.errors and result.backtests:
        result.status = "partial"
    elif result.errors:
        result.status = "failed"
    return result


def run_factor_to_decision(
    factor_name: str,
    *,
    provider: str = "synthetic",
    universe: str = "csi500",
    start_date: str = "20230101",
    end_date: str = "20241231",
    n_stocks: int = 50,
    writeback: bool = True,
) -> LoopResult:
    """回测已有因子并写回结果。"""
    import time
    t0 = time.time()
    result = LoopResult()

    from core.factor_db import FactorDatabase
    db = FactorDatabase()
    factors = [f for f in db.factors if f.get("factor_name") == factor_name]
    if not factors:
        result.status = "failed"
        result.errors.append(f"factor not found: {factor_name}")
        return result

    factor = factors[0]
    result.factors = [factor]

    try:
        from backtest.pipeline import BacktestPipeline, PipelineConfig
        config = PipelineConfig(
            provider=provider, universe=universe,
            start_date=start_date, end_date=end_date,
            n_stocks=n_stocks,
            save_json=True, save_html=True,
        )
        pipeline = BacktestPipeline(config)
        pipeline.load_data()

        bt_result = pipeline.run_factor_record(factor)
        bt_dict = bt_result.to_dict()
        result.backtests.append(bt_dict)

        from backtest.report import BacktestReport
        report = BacktestReport(
            title=f"回测: {factor_name}",
            factor_results=[bt_result],
            data_meta=pipeline.data.to_meta_dict() if pipeline.data else {},
        )
        json_path = report.save_json()
        run_id = json_path.parent.name
        result.run_ids.append(run_id)

        if writeback and bt_result.status == "ok":
            db.update_backtest_result(
                factor_name, bt_dict,
                run_id=run_id,
                compute_source=bt_result.compute_source or "formula",
            )
    except Exception as e:
        result.errors.append(str(e))
        result.status = "failed"

    result.elapsed_s = round(time.time() - t0, 2)
    return result

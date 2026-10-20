"""
backtest/pipeline.py - 一键回测编排

数据加载 → 因子计算 → 因子评价 → 策略回测 → 指标 → 报告落盘

用法:
    python -m backtest.pipeline --provider synthetic --factor return_1m --html
    python -m backtest.pipeline --provider akshare --universe csi500 --factor ROE
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from backtest.types import MarketData, FactorPanel, FactorEvalResult
from backtest.providers import DataProvider, get_provider
from backtest.factors import FactorCalculator
from backtest.ic import (
    ic_series, ic_summary, rank_ic_series,
    quantile_returns, quantile_summary,
    factor_decay, factor_turnover,
)
from backtest.strategy import StrategyConfig, StrategyEngine
from backtest.metrics import compute_performance
from backtest.report import BacktestReport


@dataclass
class PipelineConfig:
    """Pipeline 配置。"""
    provider: str = "local"
    universe: str = "csi500"
    start_date: str = "20230101"
    end_date: str = "20241231"
    n_stocks: int = 50
    forward_period: int = 21
    n_groups: int = 5
    factor_names: List[str] = field(default_factory=lambda: ["return_1m"])
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    output_dir: Optional[str] = None
    seed: int = 42
    save_json: bool = True
    save_html: bool = True


class BacktestPipeline:
    """一键回测。"""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        provider: Optional[DataProvider] = None,
    ):
        self.config = config or PipelineConfig()
        self.provider = provider or get_provider(self.config.provider, seed=self.config.seed)
        self.data: Optional[MarketData] = None

    def load_data(self) -> MarketData:
        self.data = self.provider.load(
            universe=self.config.universe,
            start=self.config.start_date,
            end=self.config.end_date,
            n_stocks=self.config.n_stocks,
        )
        return self.data

    def run_factor(self, factor_name: str) -> FactorEvalResult:
        """运行单因子评价。"""
        if self.data is None:
            self.load_data()
        calc = FactorCalculator(self.data)
        factor = calc.compute(factor_name)
        return self._evaluate_factor(factor)

    def _evaluate_factor(self, factor: FactorPanel) -> FactorEvalResult:
        """评价单个因子。"""
        fwd = self.data.forward_returns(self.config.forward_period)
        aligned_factor = factor.align(self.data)

        ic = ic_series(aligned_factor.values, fwd)
        rank_ic = rank_ic_series(aligned_factor.values, fwd)
        summary = ic_summary(ic)
        rank_summary = ic_summary(rank_ic)

        qret = quantile_returns(aligned_factor.values, fwd, n_groups=self.config.n_groups)
        qsummary = quantile_summary(qret)

        decay = factor_decay(aligned_factor.values, self.data.prices,
                            horizons=(1, 5, 10, 21, 63))
        turnover = factor_turnover(aligned_factor.values, n_groups=self.config.n_groups)
        mean_turnover = float(turnover.mean()) if len(turnover) > 0 else 0.0

        return FactorEvalResult(
            factor_name=factor.name,
            ic=summary.get("mean_ic", 0.0),
            icir=summary.get("icir", 0.0),
            rank_ic=rank_summary.get("mean_ic", 0.0),
            top_return=qsummary.get("Q5", 0.0) * 100,
            bottom_return=qsummary.get("Q1", 0.0) * 100,
            top_minus_bottom=qsummary.get("top_minus_bottom", 0.0) * 100,
            annual_return=0.0,
            max_drawdown=0.0,
            win_rate=summary.get("positive_ratio", 0.0),
            sharpe=summary.get("icir", 0.0),
            t_stat=summary.get("t_stat", 0.0),
            p_value=summary.get("p_value", 1.0),
            data_points=summary.get("data_points", 0),
            quantile_returns={k: v for k, v in qsummary.items() if k.startswith("Q")},
            mean_turnover=mean_turnover,
            decay=decay,
            ic_series=ic.dropna().round(4).tolist(),
            rank_ic_series=rank_ic.dropna().round(4).tolist(),
            dates=[str(d.date()) for d in ic.dropna().index],
            status="ok" if summary.get("data_points", 0) > 10 else "insufficient_data",
        )

    def run_strategy(self, factor_name: str) -> Any:
        """运行策略回测。"""
        if self.data is None:
            self.load_data()
        calc = FactorCalculator(self.data)
        factor = calc.compute(factor_name)

        config = self.config.strategy
        config.factor_name = factor_name
        engine = StrategyEngine(self.data)
        return engine.run(factor, config)

    def run_formula_factor(self, name: str, expression: str, **meta) -> FactorEvalResult:
        """从 DSL 公式编译因子并回测。"""
        if self.data is None:
            self.load_data()
        calc = FactorCalculator(self.data)
        panel = calc.compute_from_formula(name, expression, **meta)
        return self._evaluate_factor(panel)

    def run_factor_record(self, record: Dict) -> FactorEvalResult:
        """从因子库记录回测。

        解析顺序: formula → builtin → proxy → no_compute_path
        """
        name = record.get("factor_name", "")
        expr = record.get("formula_expression") or record.get("calculation_formula", "")

        if self.data is None:
            self.load_data()
        calc = FactorCalculator(self.data)

        # 1. 尝试公式编译
        if expr:
            try:
                panel = calc.compute_from_formula(name, expr)
                result = self._evaluate_factor(panel)
                result.compute_source = "formula"
                return result
            except Exception:
                pass

        # 2. 尝试内置因子
        try:
            panel = calc.compute(name)
            if panel.values.values.any():  # 非全零
                result = self._evaluate_factor(panel)
                result.compute_source = "builtin"
                return result
        except Exception:
            pass

        # 3. 无可计算路径
        from backtest.types import FactorEvalResult
        return FactorEvalResult(
            factor_name=name,
            status="no_compute_path",
            error=f"cannot compile formula: {expr[:100]}",
        )

    def run(self) -> BacktestReport:
        """一键全跑。"""
        self.load_data()

        # 因子评价
        factor_results = []
        for fname in self.config.factor_names:
            fr = self.run_factor(fname)
            factor_results.append(fr)

        # 策略回测（用第一个因子）
        strategy_result = None
        if self.config.factor_names:
            strategy_result = self.run_strategy(self.config.factor_names[0])

        # 构建报告
        report = BacktestReport(
            title=f"回测报告: {', '.join(self.config.factor_names)}",
            config={
                "provider": self.config.provider,
                "universe": self.config.universe,
                "start_date": self.config.start_date,
                "end_date": self.config.end_date,
                "n_stocks": self.config.n_stocks,
                "forward_period": self.config.forward_period,
                "n_groups": self.config.n_groups,
                "strategy": self.config.strategy.to_dict(),
            },
            factor_results=factor_results,
            strategy_result=strategy_result,
            data_meta=self.data.to_meta_dict(),
        )

        # 保存
        if self.config.save_json:
            out_dir = Path(self.config.output_dir) if self.config.output_dir else None
            report.save_json(out_dir / "result.json" if out_dir else None)
        if self.config.save_html:
            out_dir = Path(self.config.output_dir) if self.config.output_dir else None
            report.save_html(out_dir / "report.html" if out_dir else None)

        return report

    def run_batch(self, factor_names: List[str], parallel: bool = True, max_workers: int = 4) -> List[FactorEvalResult]:
        """批量回测多个因子（可并行）。"""
        if self.data is None:
            self.load_data()

        if not parallel or len(factor_names) <= 1:
            return [self.run_factor(fname) for fname in factor_names]

        # 并行回测（共享 MarketData，只并行计算）
        from concurrent.futures import ThreadPoolExecutor
        calc = FactorCalculator(self.data)

        def _eval_one(fname: str) -> FactorEvalResult:
            try:
                panel = calc.compute(fname)
                return self._evaluate_factor(panel)
            except Exception as e:
                return FactorEvalResult(factor_name=fname, status="error", error=str(e))

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_eval_one, factor_names))
        return results

    def run_batch(self, factor_names: List[str]) -> BacktestReport:
        self.config.factor_names = factor_names
        return self.run()


def main():
    parser = argparse.ArgumentParser(description="回测系统")
    parser.add_argument("--provider", default="local", choices=["synthetic", "akshare", "local"])
    parser.add_argument("--universe", default="csi500")
    parser.add_argument("--start", default="20230101")
    parser.add_argument("--end", default="20241231")
    parser.add_argument("--n-stocks", type=int, default=50)
    parser.add_argument("--factor", nargs="+", default=["return_1m"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--html", action="store_true", help="生成 HTML 报告")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = PipelineConfig(
        provider=args.provider,
        universe=args.universe,
        start_date=args.start,
        end_date=args.end,
        n_stocks=args.n_stocks,
        factor_names=args.factor,
        output_dir=args.output_dir,
        seed=args.seed,
        save_html=args.html,
    )

    pipeline = BacktestPipeline(config)
    report = pipeline.run()

    # 输出摘要
    summary = report.summary()
    print(f"\n{'='*50}")
    print(f"回测完成: {report.title}")
    print(f"{'='*50}")
    if "mean_ic" in summary:
        print(f"  平均 IC: {summary['mean_ic']:.4f}")
    if "strategy_metrics" in summary:
        m = summary["strategy_metrics"]
        print(f"  年化收益: {m['annual_return']:.2%}")
        print(f"  夏普比率: {m['sharpe']:.3f}")
        print(f"  最大回撤: {m['max_drawdown']:.2%}")
    print(f"\n报告已保存到 output/backtests/")


if __name__ == "__main__":
    main()

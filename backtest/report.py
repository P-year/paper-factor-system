"""
backtest/report.py - JSON + HTML 报告 + ECharts 图表

JSON envelope: timestamp + config + results + summary
HTML: 单文件自包含，内嵌 ECharts CDN + JSON 数据 + 指标表格
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backtest.types import FactorEvalResult, StrategyResult


class BacktestReport:
    """回测报告。"""

    def __init__(
        self,
        title: str = "回测报告",
        config: Optional[Dict] = None,
        factor_results: Optional[List[FactorEvalResult]] = None,
        strategy_result: Optional[StrategyResult] = None,
        data_meta: Optional[Dict] = None,
    ):
        self.title = title
        self.config = config or {}
        self.factor_results = factor_results or []
        self.strategy_result = strategy_result
        self.data_meta = data_meta or {}
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "created_at": self.created_at,
            "config": self.config,
            "data_meta": self.data_meta,
            "factor_results": [r.to_dict() for r in self.factor_results],
            "strategy_result": self.strategy_result.to_dict() if self.strategy_result else None,
        }

    def summary(self) -> Dict[str, Any]:
        """摘要（不含时间序列）。"""
        out = {
            "title": self.title,
            "created_at": self.created_at,
            "n_factors": len(self.factor_results),
        }
        if self.factor_results:
            ics = [r.ic for r in self.factor_results if r.status == "ok"]
            if ics:
                out["mean_ic"] = round(sum(ics) / len(ics), 4)
        if self.strategy_result and self.strategy_result.metrics:
            out["strategy_metrics"] = self.strategy_result.metrics.to_dict()
        return out

    def save_json(self, path: Optional[Path] = None) -> Path:
        """保存 JSON。默认 output/backtests/{run_id}/result.json。"""
        if path is None:
            run_id = self._run_id()
            path = Path("output") / "backtests" / run_id / "result.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: Path) -> "BacktestReport":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        report = cls(
            title=data.get("title", ""),
            config=data.get("config", {}),
            data_meta=data.get("data_meta", {}),
        )
        report.created_at = data.get("created_at", "")
        report.factor_results = [
            FactorEvalResult(**{k: v for k, v in r.items() if k in FactorEvalResult.__dataclass_fields__})
            for r in data.get("factor_results", [])
        ]
        sr = data.get("strategy_result")
        if sr:
            from backtest.types import PerformanceMetrics
            sr_copy = dict(sr)
            metrics_dict = sr_copy.pop("metrics", {})
            strategy = StrategyResult(**{k: v for k, v in sr_copy.items() if k in StrategyResult.__dataclass_fields__})
            if metrics_dict:
                strategy.metrics = PerformanceMetrics(**{
                    k: v for k, v in metrics_dict.items() if k in PerformanceMetrics.__dataclass_fields__
                })
            report.strategy_result = strategy
        return report

    def to_html(self) -> str:
        """生成 HTML 报告（单文件自包含）。"""
        return _render_html(self)

    def save_html(self, path: Optional[Path] = None) -> Path:
        if path is None:
            run_id = self._run_id()
            path = Path("output") / "backtests" / run_id / "report.html"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_html(), encoding="utf-8")
        return path

    def _run_id(self) -> str:
        safe_name = (self.strategy_result.factor_name if self.strategy_result else "backtest")
        safe_name = safe_name.replace("/", "_").replace(" ", "_")[:30]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_name}_{ts}"


def chart_specs(report: BacktestReport) -> List[Dict]:
    """生成 ECharts option JSON 列表。"""
    specs = []

    if report.strategy_result:
        sr = report.strategy_result
        # 1. 净值曲线
        specs.append({
            "title": "策略净值曲线",
            "type": "line",
            "x": sr.dates,
            "series": [{"name": "策略净值", "data": sr.equity}],
        })
        # 2. 回撤水下图
        specs.append({
            "title": "回撤水下图",
            "type": "area",
            "x": sr.dates,
            "series": [{"name": "回撤", "data": sr.drawdown}],
        })

    for fr in report.factor_results:
        if fr.ic_series and fr.dates:
            # 3. IC 时间序列
            specs.append({
                "title": f"IC 时间序列 ({fr.factor_name})",
                "type": "bar",
                "x": fr.dates,
                "series": [{"name": "IC", "data": fr.ic_series}],
            })
        if fr.quantile_returns:
            # 4. 分组收益柱状图
            specs.append({
                "title": f"分组收益 ({fr.factor_name})",
                "type": "bar",
                "x": list(fr.quantile_returns.keys()),
                "series": [{"name": "平均收益", "data": list(fr.quantile_returns.values())}],
            })
        if fr.decay:
            # 5. 因子衰减
            specs.append({
                "title": f"因子衰减 ({fr.factor_name})",
                "type": "line",
                "x": list(fr.decay.keys()),
                "series": [{"name": "IC", "data": list(fr.decay.values())}],
            })

    return specs


def _render_html(report: BacktestReport) -> str:
    """渲染 HTML 报告。"""
    data = report.to_dict()
    data_json = json.dumps(data, ensure_ascii=False, default=str)

    # 指标表格
    metrics_rows = ""
    if report.strategy_result and report.strategy_result.metrics:
        m = report.strategy_result.metrics
        metrics_rows = f"""
        <tr><td>年化收益</td><td>{m.annual_return:.2%}</td></tr>
        <tr><td>年化波动</td><td>{m.annual_vol:.2%}</td></tr>
        <tr><td>夏普比率</td><td>{m.sharpe:.3f}</td></tr>
        <tr><td>索提诺比率</td><td>{m.sortino:.3f}</td></tr>
        <tr><td>卡玛比率</td><td>{m.calmar:.3f}</td></tr>
        <tr><td>最大回撤</td><td>{m.max_drawdown:.2%}</td></tr>
        <tr><td>最大回撤持续</td><td>{m.max_drawdown_duration} 天</td></tr>
        <tr><td>胜率</td><td>{m.win_rate:.2%}</td></tr>
        <tr><td>总收益</td><td>{m.total_return:.2%}</td></tr>
        <tr><td>平均换手率</td><td>{m.turnover_mean:.2%}</td></tr>
        <tr><td>累计成本</td><td>{m.cost_total:.4f}</td></tr>
        """

    # 因子结果表格
    factor_rows = ""
    for fr in report.factor_results:
        factor_rows += f"""
        <tr>
            <td>{fr.factor_name}</td>
            <td>{fr.ic:.4f}</td>
            <td>{fr.icir:.4f}</td>
            <td>{fr.rank_ic:.4f}</td>
            <td>{fr.top_minus_bottom:.2%}</td>
            <td>{fr.status}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Microsoft YaHei', sans-serif; background: #f5f7fa; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ text-align: center; margin: 20px 0; color: #1a1a2e; }}
        h2 {{ margin: 30px 0 15px; color: #16213e; border-left: 4px solid #0f3460; padding-left: 12px; }}
        .card {{ background: #fff; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f0f4f8; color: #16213e; font-weight: 600; }}
        tr:hover {{ background: #f8fafc; }}
        .chart-container {{ width: 100%; height: 350px; margin: 10px 0; }}
        .meta {{ color: #666; font-size: 14px; text-align: center; margin-bottom: 20px; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
        .badge-ok {{ background: #d4edda; color: #155724; }}
        .badge-error {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{report.title}</h1>
        <p class="meta">生成时间: {report.created_at} | 数据源: {report.data_meta.get('provider', 'N/A')} |
           股票数: {report.data_meta.get('n_stocks', 'N/A')} | 天数: {report.data_meta.get('n_days', 'N/A')}</p>

        <div class="card">
            <h2>策略绩效指标</h2>
            <table>
                <tr><th>指标</th><th>值</th></tr>
                {metrics_rows}
            </table>
        </div>

        <div class="card">
            <h2>因子评价结果</h2>
            <table>
                <tr><th>因子</th><th>IC</th><th>ICIR</th><th>Rank IC</th><th>多空收益</th><th>状态</th></tr>
                {factor_rows}
            </table>
        </div>

        <div class="card">
            <h2>图表</h2>
            <div id="charts"></div>
        </div>
    </div>

    <script>
        var data = {data_json};

        // 渲染图表
        function renderCharts() {{
            var container = document.getElementById('charts');
            var chartData = data._chart_specs || [];

            // 如果没有 chart_specs，从数据自动生成
            if (chartData.length === 0) {{
                if (data.strategy_result) {{
                    chartData.push({{
                        title: '策略净值曲线', type: 'line',
                        x: data.strategy_result.dates,
                        series: [{{name: '策略净值', data: data.strategy_result.equity}}]
                    }});
                    chartData.push({{
                        title: '回撤水下图', type: 'line',
                        x: data.strategy_result.dates,
                        series: [{{name: '回撤', data: data.strategy_result.drawdown}}]
                    }});
                }}
                for (var i = 0; i < (data.factor_results || []).length; i++) {{
                    var fr = data.factor_results[i];
                    if (fr.ic_series && fr.ic_series.length > 0 && fr.dates && fr.dates.length > 0) {{
                        chartData.push({{
                            title: 'IC 时间序列 (' + fr.factor_name + ')', type: 'bar',
                            x: fr.dates,
                            series: [{{name: 'IC', data: fr.ic_series}}]
                        }});
                    }}
                    if (fr.quantile_returns && Object.keys(fr.quantile_returns).length > 0) {{
                        chartData.push({{
                            title: '分组收益 (' + fr.factor_name + ')', type: 'bar',
                            x: Object.keys(fr.quantile_returns),
                            series: [{{name: '平均收益', data: Object.values(fr.quantile_returns)}}]
                        }});
                    }}
                    if (fr.decay && Object.keys(fr.decay).length > 0) {{
                        chartData.push({{
                            title: '因子衰减 (' + fr.factor_name + ')', type: 'line',
                            x: Object.keys(fr.decay),
                            series: [{{name: 'IC', data: Object.values(fr.decay)}}]
                        }});
                    }}
                }}
            }}

            chartData.forEach(function(spec, idx) {{
                var div = document.createElement('div');
                div.className = 'chart-container';
                div.id = 'chart_' + idx;
                container.appendChild(div);

                var chart = echarts.init(div);
                var seriesOpt = spec.series.map(function(s) {{
                    return {{
                        name: s.name,
                        type: spec.type === 'bar' ? 'bar' : 'line',
                        data: s.data,
                        smooth: true,
                        areaStyle: spec.type === 'area' ? {{opacity: 0.3}} : undefined
                    }};
                }});

                chart.setOption({{
                    title: {{text: spec.title, left: 'center', textStyle: {{fontSize: 14}}}},
                    tooltip: {{trigger: 'axis'}},
                    xAxis: {{type: 'category', data: spec.x, axisLabel: {{rotate: 45, fontSize: 10}}}},
                    yAxis: {{type: 'value'}},
                    series: seriesOpt,
                    grid: {{left: 60, right: 20, top: 50, bottom: 80}}
                }});
            }});
        }}

        renderCharts();
    </script>
</body>
</html>"""

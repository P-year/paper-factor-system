"""
数据获取模块 - AKShare集成 (v2.0)
Phase 2 真实数据回测的核心数据管道

验证可用的API:
- index_stock_cons(symbol="000905")  → 中证500成分股(500,3)
- stock_zh_a_hist(symbol, period="daily", start_date, end_date, adjust="qfq") → 日频行情
- stock_zh_a_spot_em() → 全市场股票列表（慢）
"""

import warnings
warnings.filterwarnings('ignore')
os_env = __import__('os').environ
os_env['AKSHARE_NO_PROGRESS'] = '1'
os_env['NO_PROGRESS_BARS'] = '1'

import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False

from config import DATA_DIR


# ============================================================
# 常量
# ============================================================

# 因子配置：category 决定数据源，data_source 决定如何计算
FACTOR_CONFIGS = {
    # === 动量类（价格驱动） ===
    "return_1m":  {"name": "1月动量",    "category": "momentum",  "data_source": "price",  "period": 21,   "higher_is_better": False, "formula": "近1月收益率"},
    "return_3m":  {"name": "3月动量",    "category": "momentum",  "data_source": "price",  "period": 63,   "higher_is_better": False, "formula": "近3月收益率"},
    "return_6m":  {"name": "6月动量",    "category": "momentum",  "data_source": "price",  "period": 126,  "higher_is_better": False, "formula": "近6月收益率"},
    "return_1y":  {"name": "1年动量",    "category": "momentum",  "data_source": "price",  "period": 252,  "higher_is_better": False, "formula": "近1年收益率"},
    "turnover_1m": {"name": "1月换手率", "category": "momentum",  "data_source": "price",  "period": 21,   "higher_is_better": False, "formula": "近1月均换手率（用价格变化代理）"},

    # === 质量类（财报数据） ===
    "ROE":               {"name": "ROE 净资产收益率",     "category": "quality",  "data_source": "financial", "column": "净资产收益率(%)",   "higher_is_better": True,  "formula": "净利润 / 净资产"},
    "gross_profit_margin": {"name": "销售毛利率",        "category": "quality",  "data_source": "financial", "column": "销售毛利率(%)",     "higher_is_better": True,  "formula": "(营收 - 成本) / 营收"},
    "net_profit_margin":  {"name": "销售净利率",         "category": "quality",  "data_source": "financial", "column": "销售净利率(%)",     "higher_is_better": True,  "formula": "净利润 / 营收"},

    # === 效率类（财报数据） ===
    "asset_turnover":     {"name": "总资产周转率",       "category": "efficiency", "data_source": "financial", "column": "总资产周转率(次)", "higher_is_better": True,  "formula": "营收 / 平均总资产"},

    # === 成长类（财报数据） ===
    "revenue_growth_yoy": {"name": "营收同比增长率",     "category": "growth",   "data_source": "financial", "column": "营业总收入增长率(%)", "higher_is_better": True,  "formula": "(本期营收 - 上年同期) / 上年同期"},
    "profit_growth_yoy":  {"name": "净利润同比增长率",   "category": "growth",   "data_source": "financial", "column": "净利润增长率(%)",   "higher_is_better": True,  "formula": "(本期净利 - 上年同期) / 上年同期"},
}


@dataclass
class BacktestResult:
    """回测结果"""
    factor_name: str
    ic: float
    icir: float
    rank_ic: float
    top_return: float
    bottom_return: float
    top_minus_bottom: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    data_points: int
    status: str = ""

    def to_dict(self) -> Dict:
        return {
            "factor_name": self.factor_name,
            "IC": round(self.ic, 4),
            "ICIR": round(self.icir, 4),
            "RankIC": round(self.rank_ic, 4),
            "TOP_annual": round(self.top_return, 2),
            "BOTTOM_annual": round(self.bottom_return, 2),
            "Long_Short": round(self.top_minus_bottom, 2),
            "Sharpe": round(self.sharpe, 2),
            "MaxDD": round(self.max_drawdown, 2),
            "N": self.data_points,
            "status": self.status,
        }


# ============================================================
# 数据获取器
# ============================================================

class DataFetcher:
    """AKShare数据获取器（v2.0 验证可用版）"""

    def __init__(self, cache_dir: Optional[str] = None):
        if not HAS_AKSHARE:
            raise ImportError("pip install akshare")

        self.cache_dir = Path(cache_dir) if cache_dir else DATA_DIR / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self._cache: Dict[str, pd.DataFrame] = {}
        self.rate_limit = 0.3  # 秒

    # ------------------------------------------------------
    # 股票池
    # ------------------------------------------------------

    def get_universe(self, name: str = "csi500", n: int = 100) -> List[str]:
        """
        获取股票池代码列表

        Args:
            name: csi500 / csi300 / all_a
            n: 返回数量上限（0=不限）

        Returns:
            股票代码列表，如 ["000001", "600000", ...]
        """
        print(f"[INFO] 获取股票池: {name}")

        if name == "csi500":
            try:
                df = ak.index_stock_cons(symbol="000905")
                codes = df['品种代码'].astype(str).str.zfill(6).tolist()
                print(f"  中证500成分股: {len(codes)} 只")
            except Exception as e:
                print(f"  失败: {e}，使用备用列表")
                codes = self._get_fallback_stocks(n or 50)
        elif name == "csi300":
            try:
                df = ak.index_stock_cons(symbol="000300")
                codes = df['品种代码'].astype(str).str.zfill(6).tolist()
            except Exception:
                codes = self._get_fallback_stocks(n or 50)
        else:
            codes = self._get_fallback_stocks(n or 100)

        if n > 0 and len(codes) > n:
            codes = codes[:n]

        print(f"  返回: {len(codes)} 只")
        return codes

    def _get_fallback_stocks(self, n: int) -> List[str]:
        """备用股票列表（蓝筹股）"""
        blue_chips = [
            "000001", "000002", "000063", "000333", "000338",
            "000651", "000661", "000858", "000876", "000895",
            "002001", "002002", "002003", "002004", "002007",
            "002027", "002028", "002036", "002044", "002049",
            "002050", "002142", "002230", "002236", "002236",
            "002252", "002304", "002311", "002312", "002354",
            "600000", "600009", "600016", "600019", "600028",
            "600030", "600036", "600050", "600104", "600111",
            "600150", "600160", "600183", "600309", "600519",
            "600690", "600887", "600893", "600905", "600941",
        ]
        return blue_chips[:n]

    # ------------------------------------------------------
    # 价格数据
    # ------------------------------------------------------

    def _get_price(self, code: str, start: str, end: str) -> Optional[pd.Series]:
        """获取单只股票收盘价"""
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq"
            )
            if df is None or len(df) == 0:
                return None
            df.columns = [c.lower() for c in df.columns]
            date_col = '日期' if '日期' in df.columns else 'date'
            df['date'] = pd.to_datetime(df[date_col])
            series = df.set_index('date')['收盘']
            series.name = code
            return series
        except Exception:
            return None

    def get_batch_prices(
        self,
        codes: List[str],
        start: str,
        end: str,
        max_workers: int = 8,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        并发批量获取价格面板

        Returns:
            DataFrame (date index × stock code columns, values=close)
        """
        prices: Dict[str, pd.Series] = {}
        done = 0

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self._get_price, c, start, end): c for c in codes}
            for future in as_completed(futures):
                code = futures[future]
                done += 1
                if verbose and done % 20 == 0:
                    print(f"  {done}/{len(codes)}")

                try:
                    s = future.result(timeout=10)
                    if s is not None and len(s) > 100:
                        prices[code] = s
                except Exception:
                    pass

        if not prices:
            return pd.DataFrame()

        panel = pd.DataFrame(prices)
        panel.index = pd.to_datetime(panel.index)
        panel = panel.sort_index()

        if verbose:
            print(f"  成功: {panel.shape[1]}只 x {panel.shape[0]}天")

        return panel

    # ------------------------------------------------------
    # IC计算
    # ------------------------------------------------------

    @staticmethod
    def calculate_IC(factor_panel: pd.DataFrame, ret_panel: pd.DataFrame) -> Tuple[float, float, float]:
        """
        计算 IC / ICIR / RankIC

        Args:
            factor_panel: 因子值面板 (date × stock)
            ret_panel: 未来收益面板 (date × stock)

        Returns:
            (mean_ic, icir, mean_rank_ic)
        """
        common = factor_panel.index.intersection(ret_panel.index)
        ic_list, ric_list = [], []

        for date in common:
            fv = factor_panel.loc[date].values.astype(float)
            rv = ret_panel.loc[date].values.astype(float)
            valid = ~(np.isnan(fv) | np.isnan(rv) | np.isinf(fv) | np.isinf(rv))
            if valid.sum() < 10:
                continue

            f_ok, r_ok = fv[valid], rv[valid]
            if np.std(f_ok) < 1e-10 or np.std(r_ok) < 1e-10:
                continue

            ic = np.corrcoef(f_ok, r_ok)[0, 1]
            if not np.isnan(ic):
                ic_list.append(ic)

            # Rank IC
            rank_f = np.argsort(np.argsort(f_ok))
            rank_r = np.argsort(np.argsort(r_ok))
            ric = np.corrcoef(rank_f, rank_r)[0, 1]
            if not np.isnan(ric):
                ric_list.append(ric)

        if not ic_list:
            return 0.0, 0.0, 0.0

        arr = np.array(ic_list)
        mean_ic = float(np.mean(arr))
        ic_std = float(np.std(arr))
        icir = mean_ic / (ic_std + 1e-9)
        mean_ric = float(np.mean(ric_list)) if ric_list else 0.0

        return mean_ic, icir, mean_ric

    # ------------------------------------------------------
    # 一站式回测
    # ------------------------------------------------------

    def run_backtest(
        self,
        factor_name: str,
        universe: str = "csi500",
        start_date: str = "20230101",
        end_date: str = "20241231",
        n_stocks: int = 50,
        forward_period: int = 21,
        lookback: int = 0,
    ) -> Dict:
        """
        一站式因子回测

        Args:
            factor_name: 因子名（见FACTOR_CONFIGS）
            universe: 股票池
            start_date: 回测开始
            end_date: 回测结束
            n_stocks: 股票数量
            forward_period: 持仓周期（交易日）
            lookback: 因子回看天数（0=自动）

        Returns:
            回测结果字典
        """
        config = FACTOR_CONFIGS.get(factor_name, {})
        print(f"\n[回测] {factor_name} | {universe} | {start_date}~{end_date}")
        print(f"  持仓周期: {forward_period}天 | 股票数: {n_stocks}")

        # 1. 获取股票池
        codes = self.get_universe(universe, n=n_stocks)

        # 2. 获取价格
        t0 = time.time()
        prices = self.get_batch_prices(codes, start_date, end_date)
        if prices.empty:
            return {"error": "价格数据为空"}
        print(f"  价格耗时: {time.time()-t0:.1f}s")

        # 3. 计算因子（按 data_source 分发）
        data_source = config.get("data_source", "price")
        if data_source == "price":
            factor = self._compute_price_factor(factor_name, prices, config)
        elif data_source == "financial":
            try:
                factor = self._compute_financial_factor(
                    factor_name, codes, start_date, end_date, config
                )
                # 财务因子panel可能不覆盖全部价格日期，reindex + ffill
                factor = factor.reindex(index=prices.index, method="ffill")
            except Exception as e:
                return {"error": f"financial factor compute failed: {e}", "factor_name": factor_name, "status": "error"}
        else:
            factor = prices.pct_change(21)

        # 4. 对齐截面
        factor = factor.iloc[lookback if lookback else 30:-forward_period]
        prices_sub = prices.reindex(index=factor.index)

        # 5. 计算未来收益
        future = prices_sub.shift(-forward_period)
        ret = (future - prices_sub) / prices_sub
        ret = ret.iloc[:-forward_period]

        # 6. IC计算
        ic, icir, rank_ic = self.calculate_IC(factor, ret)

        # 7. 分组收益
        group_returns = self._calc_group_returns(factor, ret)

        # 8. 合成结果
        top_q = max(group_returns.keys())
        bot_q = min(group_returns.keys())
        top_ret = group_returns[top_q] * (252 / forward_period) * 100
        bot_ret = group_returns[bot_q] * (252 / forward_period) * 100
        ls_ret = top_ret - bot_ret

        status_map = {
            "有效": abs(ic) > 0.03,
            "偏弱": abs(ic) > 0.01,
            "基本无效": True,
        }
        status = "基本无效"
        for s, cond in status_map.items():
            if cond and status == "基本无效":
                status = s

        result = {
            "factor_name": factor_name,
            "IC": float(round(ic, 4)),
            "ICIR": float(round(icir, 4)),
            "RankIC": float(round(rank_ic, 4)),
            "top_return": float(round(top_ret, 2)),
            "bottom_return": float(round(bot_ret, 2)),
            "long_short": float(round(ls_ret, 2)),
            "n_stocks": int(n_stocks),
            "n_dates": int(factor.shape[0]),
            "status": str(status),
        }

        print(f"\n  结果: IC={ic:.4f} ICIR={icir:.4f} RankIC={rank_ic:.4f}")
        print(f"  TOP年化={top_ret:.2f}% BOTTOM={bot_ret:.2f}% 多空={ls_ret:.2f}%")
        print(f"  结论: {status}")

        return result

    # ------------------------------------------------------
    # 因子计算（新增：价格 vs 财务）
    # ------------------------------------------------------

    def _compute_price_factor(
        self,
        factor_name: str,
        prices: pd.DataFrame,
        config: Dict,
    ) -> pd.DataFrame:
        """价格驱动的因子（动量类）"""
        period = config.get("period", 21)
        if factor_name == "turnover_1m":
            # TODO: 用真实换手率，先用价格变化代理
            return prices.pct_change(period)
        return prices.pct_change(period)

    def _compute_financial_factor(
        self,
        factor_name: str,
        codes: List[str],
        start_date: str,
        end_date: str,
        config: Dict,
    ) -> pd.DataFrame:
        """财报驱动的因子（质量/效率/成长类）

        流程：
        1. 对每只股票调用 stock_financial_analysis_indicator 获取季度财报指标
        2. 提取 config["column"] 列
        3. 横向拼接成 panel（index=季度日期，column=股票代码）
        """
        column = config["column"]
        print(f"  财报因子 {factor_name}: 拉取 {column}")

        # 提取起始年份（财务数据按年）
        start_year = start_date[:4]

        rows = {}
        for code in codes:
            try:
                df = ak.stock_financial_analysis_indicator(symbol=code, start_year=str(int(start_year) - 1))
                if df is None or df.empty or column not in df.columns:
                    continue
                # 第一列是日期列
                date_col = df.columns[0]
                s = df.set_index(date_col)[column]
                s.index = pd.to_datetime(s.index)
                s = s.sort_index()
                rows[code] = s
            except Exception as e:
                # 单只股票失败不影响其他
                continue

        if not rows:
            raise ValueError(f"no financial data for column={column}")

        # 拼成 panel
        panel = pd.DataFrame(rows)
        panel = panel.sort_index()
        return panel

    def _calc_group_returns(
        self,
        factor_panel: pd.DataFrame,
        ret_panel: pd.DataFrame,
        quantile: int = 5,
    ) -> Dict[int, float]:
        """计算分组平均收益"""
        common = factor_panel.index.intersection(ret_panel.index)
        fp = factor_panel.loc[common]
        rp = ret_panel.loc[common]
        groups = {i: [] for i in range(1, quantile + 1)}

        for date in common:
            fv = fp.loc[date].values.astype(float)
            rv = rp.loc[date].values.astype(float)
            valid = ~(np.isnan(fv) | np.isnan(rv) | np.isinf(fv) | np.isinf(rv))
            if valid.sum() < quantile * 3:
                continue
            try:
                q = pd.qcut(fv[valid], q=quantile, labels=False, duplicates='drop')
                for qi in range(quantile):
                    mask = q == qi
                    if mask.sum() > 0:
                        groups[qi + 1].append(np.mean(rv[valid][mask]))
            except Exception:
                continue

        return {q: np.mean(r) if r else 0.0 for q, r in groups.items()}


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("DataFetcher v2.0 测试")

    fetcher = DataFetcher()

    # 测试股票池
    codes = fetcher.get_universe("csi500", n=20)
    print(f"\n股票池: {len(codes)} 只")

    # 测试价格获取
    prices = fetcher.get_batch_prices(codes[:5], "20230101", "20230630")
    print(f"价格面板: {prices.shape}")

    # 测试IC
    if not prices.empty:
        factor = prices.pct_change(21).iloc[30:-21]
        future = prices.shift(-21)
        ret = (future - prices) / prices
        ret = ret.iloc[30:-21]
        ic, icir, ric = fetcher.calculate_IC(factor, ret)
        print(f"IC测试: IC={ic:.4f} ICIR={icir:.4f} RankIC={ric:.4f}")

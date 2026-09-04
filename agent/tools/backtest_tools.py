"""
backtest_tools - 因子真实回测
复用：DataFetcher.run_backtest (data_fetcher.py)
"""
from typing import Dict


def run_backtest(
    factor_name: str,
    universe: str = "csi500",
    start_date: str = "20230101",
    end_date: str = "20241231",
    n_stocks: int = 50,
) -> Dict:
    """
    用 AKShare 真实数据对单个因子做 IC/RankIC 回测。

    Args:
        factor_name: 因子名（见 FACTOR_CONFIGS）
        universe: csi500 / csi300 / all_a
        start_date: 起始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        n_stocks: 股票数

    Returns:
        {
          "factor_name": str,
          "IC": float, "ICIR": float, "RankIC": float,
          "top_return": float, "bottom_return": float, "long_short": float,
          "n_dates": int, "status": "ok"|"insufficient_data"|"error",
          "error": str | None
        }
    """
    try:
        from data_fetcher import DataFetcher

        fetcher = DataFetcher()
        result = fetcher.run_backtest(
            factor_name=factor_name,
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            n_stocks=n_stocks,
        )
        return result
    except ImportError:
        return {
            "factor_name": factor_name,
            "error": "akshare not installed. Run: pip install akshare",
            "status": "error",
        }
    except Exception as e:
        return {
            "factor_name": factor_name,
            "error": str(e),
            "status": "error",
        }
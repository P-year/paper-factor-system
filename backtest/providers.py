"""
backtest/providers.py - 可插拔数据层

DataProvider ABC + SyntheticProvider(默认) + AKShareProvider + get_provider()
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.types import MarketData


class DataProvider(ABC):
    """数据提供者抽象基类。"""

    name: str = "base"

    @abstractmethod
    def get_universe(self, name: str = "csi500", n: int = 100) -> List[str]:
        """获取股票池。返回股票代码列表。"""

    @abstractmethod
    def get_prices(self, codes: List[str], start: str, end: str) -> pd.DataFrame:
        """获取收盘价。index=date, columns=codes, values=close。"""

    def get_market_cap(self, codes: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        """获取市值（可选）。"""
        return None

    def get_volume(self, codes: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        """获取成交量（可选）。"""
        return None

    def load(
        self,
        universe: str = "csi500",
        start: str = "20230101",
        end: str = "20241231",
        n_stocks: int = 50,
    ) -> MarketData:
        """模板方法：universe → prices → returns → MarketData。"""
        codes = self.get_universe(universe, n_stocks)
        prices = self.get_prices(codes, start, end)
        returns = prices.pct_change().fillna(0.0)
        market_cap = self.get_market_cap(codes, start, end)
        volume = self.get_volume(codes, start, end)
        return MarketData(
            prices=prices,
            returns=returns,
            market_cap=market_cap,
            volume=volume,
            universe=codes,
            start_date=start,
            end_date=end,
            provider=self.name,
        )


class SyntheticProvider(DataProvider):
    """合成数据提供者（numpy 随机游走 + 因子暴露）。

    特点：
    - seed 可控，同 seed 结果确定
    - 可注入已知 IC 的潜在因子（用于测试验证）
    - 零外部依赖
    """

    name = "synthetic"

    def __init__(
        self,
        n_stocks: int = 50,
        n_days: int = 500,
        annual_drift: float = 0.05,
        annual_vol: float = 0.25,
        seed: int = 42,
        factor_ic: float = 0.05,
    ):
        self.n_stocks = n_stocks
        self.n_days = n_days
        self.annual_drift = annual_drift
        self.annual_vol = annual_vol
        self.seed = seed
        self.factor_ic = factor_ic
        self._prices_cache: Optional[pd.DataFrame] = None
        self._factor_cache: Optional[pd.DataFrame] = None

    def get_universe(self, name: str = "synthetic", n: int = 100) -> List[str]:
        n = min(n, self.n_stocks)
        return [f"STK{i:04d}" for i in range(n)]

    def get_prices(self, codes: List[str], start: str = "", end: str = "") -> pd.DataFrame:
        if self._prices_cache is not None and list(self._prices_cache.columns) == codes:
            return self._prices_cache

        np.random.seed(self.seed)
        n = len(codes)
        T = self.n_days

        # 日参数
        daily_drift = self.annual_drift / 252
        daily_vol = self.annual_vol / np.sqrt(252)

        # 生成因子暴露（用于注入已知 IC）
        factor_exposure = np.random.randn(n) * 0.5  # 截面因子值

        # 生成收益率: r = drift + factor_ic * factor + noise
        noise = np.random.randn(T, n) * daily_vol
        returns_matrix = daily_drift + self.factor_ic * factor_exposure[np.newaxis, :] / np.sqrt(T) + noise

        # 累积为价格
        prices = 100.0 * np.exp(np.cumsum(returns_matrix, axis=0))
        dates = pd.bdate_range(start="2023-01-02", periods=T)
        self._prices_cache = pd.DataFrame(prices, index=dates, columns=codes)
        return self._prices_cache

    def get_market_cap(self, codes: List[str], start: str = "", end: str = "") -> pd.DataFrame:
        prices = self.get_prices(codes, start, end)
        np.random.seed(self.seed + 1)
        base_cap = np.random.lognormal(mean=22, sigma=1, size=len(codes))
        # shape: (n_days, n_stocks) — 每行相同 base_cap
        caps = pd.DataFrame(
            np.tile(base_cap, (len(prices), 1)),
            index=prices.index,
            columns=codes,
        )
        return caps

    def get_factor_values(self, codes: List[str]) -> pd.DataFrame:
        """返回注入的因子值（date × codes），用于测试。"""
        np.random.seed(self.seed)
        n = len(codes)
        T = self.n_days
        factor_exposure = np.random.randn(n) * 0.5
        values = np.tile(factor_exposure, (T, 1))
        dates = pd.bdate_range(start="2023-01-02", periods=T)
        return pd.DataFrame(values, index=dates, columns=codes)


class AKShareProvider(DataProvider):
    """AKShare 数据提供者（包装 core/data_fetcher.DataFetcher）。"""

    name = "akshare"

    def __init__(self):
        try:
            from core.data_fetcher import DataFetcher
            self._fetcher = DataFetcher()
        except ImportError as e:
            raise ImportError(f"AKShare not available: {e}") from e

    def get_universe(self, name: str = "csi500", n: int = 100) -> List[str]:
        return self._fetcher.get_universe(name, n)

    def get_prices(self, codes: List[str], start: str, end: str) -> pd.DataFrame:
        return self._fetcher.get_batch_prices(codes, start, end)


class DBProvider(DataProvider):
    """数据库数据提供者（骨架，预留扩展）。"""

    name = "db"

    def get_universe(self, name: str = "csi500", n: int = 100) -> List[str]:
        raise NotImplementedError("DBProvider 尚未实现")

    def get_prices(self, codes: List[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("DBProvider 尚未实现")


class LocalDataProvider(DataProvider):
    """从本地 parquet 文件读取真实 A 股数据。"""

    name = "local"

    def __init__(self, data_dir: str = "data/market"):
        self.data_dir = Path(data_dir)
        self._prices: Optional[pd.DataFrame] = None
        self._market_cap: Optional[pd.DataFrame] = None
        self._volume: Optional[pd.DataFrame] = None
        self._meta: Dict = {}

    def _load(self):
        if self._prices is not None:
            return
        prices_path = self.data_dir / "prices.parquet"
        if not prices_path.exists():
            raise FileNotFoundError(
                f"本地数据不存在: {prices_path}\n"
                f"请先运行: python -m backtest.download_data"
            )
        self._prices = pd.read_parquet(prices_path)

        meta_path = self.data_dir / "meta.json"
        if meta_path.exists():
            import json
            self._meta = json.loads(meta_path.read_text(encoding="utf-8"))

        cap_path = self.data_dir / "market_cap.parquet"
        if cap_path.exists():
            cap_df = pd.read_parquet(cap_path)
            self._market_cap = cap_df
            # 扩展为与 prices 同形状
            if isinstance(cap_df, pd.DataFrame):
                self._market_cap = cap_df.iloc[:, 0]
                self._market_cap = pd.DataFrame(
                    [self._market_cap.values] * len(self._prices),
                    index=self._prices.index,
                    columns=self._prices.columns,
                )

        vol_path = self.data_dir / "volume.parquet"
        if vol_path.exists():
            self._volume = pd.read_parquet(vol_path)

    def get_universe(self, name: str = "csi500", n: int = 100) -> List[str]:
        self._load()
        codes = list(self._prices.columns)
        return codes[:n]

    def get_prices(self, codes: List[str], start: str = "", end: str = "") -> pd.DataFrame:
        self._load()
        df = self._prices
        if codes:
            cols = [c for c in codes if c in df.columns]
            df = df[cols]
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    def get_market_cap(self, codes: List[str], start: str = "", end: str = "") -> Optional[pd.DataFrame]:
        self._load()
        return self._market_cap

    def get_volume(self, codes: List[str], start: str = "", end: str = "") -> Optional[pd.DataFrame]:
        self._load()
        return self._volume


def get_provider(name: str = "synthetic", **kwargs) -> DataProvider:
    """数据提供者工厂。"""
    if name == "synthetic":
        return SyntheticProvider(**kwargs)
    elif name == "akshare":
        return AKShareProvider()
    elif name == "local":
        return LocalDataProvider(data_dir=kwargs.get("data_dir", "data/market"))
    elif name == "db":
        return DBProvider()
    else:
        raise ValueError(f"unknown provider: {name!r} (expect synthetic|akshare|local|db)")

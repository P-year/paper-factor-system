"""
core - 业务核心模块（论文采集、因子提取、数据获取、回测、因子库）

这些是从根目录搬过来的"业务工具"层，被 agent/tools/ 和 pipelines/ 调用。
搬迁理由：根目录太乱，把核心工具集中放一起更清晰。

用法：
    from core.analyzer import FactorAnalyzer
    from core.backtester import FactorBacktester
    from core.collector import PaperCollector
    from core.data_fetcher import DataFetcher, FACTOR_CONFIGS
    from core.factor_db import FactorDatabase
    from core.local_collector import LocalPaperCollector
"""
from .analyzer import FactorAnalyzer
from .backtester import FactorBacktester, BacktestResult
from .collector import PaperCollector
from .data_fetcher import DataFetcher, FACTOR_CONFIGS
from .factor_db import FactorDatabase
from .local_collector import LocalPaperCollector

__all__ = [
    "FactorAnalyzer",
    "FactorBacktester",
    "BacktestResult",
    "PaperCollector",
    "DataFetcher",
    "FACTOR_CONFIGS",
    "FactorDatabase",
    "LocalPaperCollector",
]
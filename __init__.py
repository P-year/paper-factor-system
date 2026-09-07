"""
论文挖因子系统
Paper → Factor Extraction → Backtest Verification

Author: 刘烨 (Liu Ye)
"""

from core.collector import PaperCollector
from core.analyzer import FactorAnalyzer
from core.factor_db import FactorDatabase
from core.backtester import FactorBacktester

__all__ = [
    "PaperCollector",
    "FactorAnalyzer",
    "FactorDatabase",
    "FactorBacktester",
]
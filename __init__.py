"""
论文挖因子系统 - 第一阶段
Paper → Factor Extraction → Backtest Verification

Author: 理 (Li)
Date: 2026-05-03
"""

from collector import PaperCollector
from analyzer import FactorAnalyzer
from factor_db import FactorDatabase
from backtester import FactorBacktester

__all__ = [
    "PaperCollector",
    "FactorAnalyzer",
    "FactorDatabase",
    "FactorBacktester",
]
"""
tests/test_verify.py - harness/verify.py 单元测试

覆盖：
- 正常 backtest 结果 → verified=True
- IC 越界 / NaN / Inf → verified=False
- n_dates 太少 → verified=False
- 方向不一致 → verified=False
- 缺失字段 → verified=False
- 错误状态 → verified=False
- 因子名太宽泛 → verified=False
- 因子字段缺失 → verified=False
- 假因子（is_factor=false）→ verified=True（不需要检查）
- 论文采集 → verified=True/False
"""
import math
import pytest

from harness.verify import (
    verify_backtest_result,
    verify_extracted_factor,
    verify_paper_collect,
    MIN_N_DATES,
    MAX_ABS_IC,
)


# ============================================================
# verify_backtest_result 测试
# ============================================================

class TestVerifyBacktestResult:
    """backtest 结果验证测试"""

    def test_normal_ok_result(self):
        """正常回测结果：通过"""
        result = {
            "factor_name": "momentum_1m",
            "IC": 0.05,
            "ICIR": 1.2,
            "RankIC": 0.04,
            "top_return": 0.08,
            "bottom_return": -0.02,
            "long_short": 0.10,
            "n_dates": 240,
            "status": "ok",
        }
        v = verify_backtest_result(result)
        assert v["verified"] is True
        assert v["issues"] == []
        assert v["confidence"] == "high"

    def test_status_error(self):
        """status=error：未通过"""
        result = {
            "status": "error",
            "error": "AKShare timeout",
            "IC": 0.0,
            "ICIR": 0.0,
            "n_dates": 0,
        }
        v = verify_backtest_result(result)
        assert v["verified"] is False
        assert any("报错" in i for i in v["issues"])

    def test_status_insufficient_data(self):
        """status=insufficient_data：未通过"""
        result = {
            "status": "insufficient_data",
            "IC": 0.0,
            "ICIR": 0.0,
            "n_dates": 5,
        }
        v = verify_backtest_result(result)
        assert v["verified"] is False
        assert any("数据不足" in i for i in v["issues"])

    def test_ic_out_of_range(self):
        """IC 超出 [-1, 1]：未通过"""
        result = {
            "IC": 5.0,
            "ICIR": 1.0,
            "n_dates": 240,
            "top_return": 0.5,
            "bottom_return": -0.5,
            "long_short": 1.0,
            "status": "ok",
        }
        v = verify_backtest_result(result)
        assert v["verified"] is False
        assert any("超出" in i for i in v["issues"])

    def test_ic_nan(self):
        """IC=NaN：未通过"""
        result = {
            "IC": float("nan"),
            "ICIR": 1.0,
            "n_dates": 240,
            "top_return": 0.5,
            "long_short": 1.0,
            "status": "ok",
        }
        v = verify_backtest_result(result)
        assert v["verified"] is False
        assert any("NaN" in i or "不是有限数" in i for i in v["issues"])

    def test_ic_inf(self):
        """IC=Inf：未通过"""
        result = {
            "IC": float("inf"),
            "ICIR": 1.0,
            "n_dates": 240,
            "long_short": 1.0,
            "status": "ok",
        }
        v = verify_backtest_result(result)
        assert v["verified"] is False

    def test_n_dates_too_small(self):
        """交易日数太少：未通过"""
        result = {
            "IC": 0.05,
            "ICIR": 1.0,
            "n_dates": 5,
            "top_return": 0.05,
            "long_short": 0.10,
            "status": "ok",
        }
        v = verify_backtest_result(result)
        assert v["verified"] is False
        assert any("交易日数" in i for i in v["issues"])

    def test_icir_too_large(self):
        """ICIR 异常大：未通过"""
        result = {
            "IC": 0.05,
            "ICIR": 50.0,
            "n_dates": 240,
            "top_return": 0.05,
            "long_short": 0.10,
            "status": "ok",
        }
        v = verify_backtest_result(result)
        assert v["verified"] is False
        assert any("ICIR" in i for i in v["issues"])

    def test_direction_inconsistent(self):
        """IC 和 long_short 方向不一致：未通过"""
        result = {
            "IC": 0.05,  # 正
            "ICIR": 1.0,
            "n_dates": 240,
            "top_return": 0.05,
            "long_short": -0.10,  # 负
            "status": "ok",
        }
        v = verify_backtest_result(result)
        assert v["verified"] is False
        assert any("方向不一致" in i for i in v["issues"])

    def test_missing_fields(self):
        """缺失字段：未通过"""
        result = {
            "IC": 0.05,
            # 缺 ICIR, n_dates
            "status": "ok",
        }
        v = verify_backtest_result(result)
        assert v["verified"] is False
        assert any("缺失字段" in i for i in v["issues"])

    def test_not_dict(self):
        """结果不是 dict：未通过"""
        v = verify_backtest_result("not a dict")
        assert v["verified"] is False
        assert any("不是 dict" in i for i in v["issues"])

    def test_ic_zero_with_other_zero_values(self):
        """全 0 结果（典型 ICIR=0, top=-100% 场景）：所有字段合法，应该通过"""
        # 这是真实跑出来的 reject 因子情况——backtest 本身是对的，只是因子效果差
        result = {
            "IC": 0.0,
            "ICIR": 0.0,
            "n_dates": 240,
            "top_return": -0.5,
            "bottom_return": -0.5,
            "long_short": 0.0,
            "status": "ok",
        }
        v = verify_backtest_result(result)
        # 所有数值在合法范围内，应该通过——说明 verify 不是在判断"因子好不好"
        # 而是在判断"回测输出可不可信"
        assert v["verified"] is True


# ============================================================
# verify_extracted_factor 测试
# ============================================================

class TestVerifyExtractedFactor:
    """LLM 提取的因子验证测试"""

    def test_good_factor(self):
        """正常提取的因子：通过"""
        factor = {
            "is_factor": True,
            "factor_name": "12月动量",
            "definition": "过去 252 天的对数收益率",
            "data_needed": "日频收盘价",
            "calculation_formula": "log(close_t / close_t-252)",
            "confidence": "高",
            "paper_conclusion": "在 2010-2022 年 A 股回测中表现优异，IC=0.05",
        }
        v = verify_extracted_factor(factor)
        assert v["verified"] is True
        assert v["issues"] == []

    def test_generic_name(self):
        """因子名太宽泛：未通过"""
        factor = {
            "is_factor": True,
            "factor_name": "alpha",
            "definition": "动量类因子",
            "data_needed": "收益率",
            "calculation_formula": "...",
            "confidence": "高",
            "paper_conclusion": "OK",
        }
        v = verify_extracted_factor(factor)
        assert v["verified"] is False
        assert any("太宽泛" in i for i in v["issues"])

    def test_missing_definition(self):
        """definition 缺失：未通过"""
        factor = {
            "is_factor": True,
            "factor_name": "momentum",
            "data_needed": "close",
            "calculation_formula": "...",
        }
        v = verify_extracted_factor(factor)
        assert v["verified"] is False
        assert any("definition" in i for i in v["issues"])

    def test_incomplete_formula(self):
        """公式偷懒：未通过"""
        factor = {
            "is_factor": True,
            "factor_name": "test_factor",
            "definition": "动量类",
            "data_needed": "close",
            "calculation_formula": "(close - close[N]) 等",
            "confidence": "高",
            "paper_conclusion": "paper conclusion text",
        }
        v = verify_extracted_factor(factor)
        assert v["verified"] is False
        assert any("偷懒写法" in i for i in v["issues"])

    def test_overconfident(self):
        """过度自信：未通过"""
        factor = {
            "is_factor": True,
            "factor_name": "momentum",
            "definition": "...",
            "data_needed": "...",
            "calculation_formula": "...",
            "confidence": "高",
            "paper_conclusion": "好",  # 只有 1 个字符
        }
        v = verify_extracted_factor(factor)
        assert v["verified"] is False
        assert any("置信度=高" in i for i in v["issues"])

    def test_not_a_factor_paper(self):
        """非因子论文：直接通过（不需要字段）"""
        factor = {
            "is_factor": False,
            "reason": "综述类论文",
        }
        v = verify_extracted_factor(factor)
        assert v["verified"] is True


# ============================================================
# verify_paper_collect 测试
# ============================================================

class TestVerifyPaperCollect:
    """论文采集验证测试"""

    def test_papers_with_title_and_summary(self):
        """正常论文：通过"""
        papers = [
            {"title": "Paper 1", "summary": "Abstract..."},
            {"title": "Paper 2", "summary": "Abstract..."},
        ]
        v = verify_paper_collect(papers)
        assert v["verified"] is True
        assert v["n_papers"] == 2

    def test_empty_papers(self):
        """空论文列表：未通过"""
        v = verify_paper_collect([])
        assert v["verified"] is False
        assert any("未采集到" in i for i in v["issues"])

    def test_missing_fields(self):
        """论文缺字段：未通过"""
        papers = [
            {"title": "Paper 1", "summary": "Abstract..."},
            {"title": ""},  # 缺 title 和 summary
        ]
        v = verify_paper_collect(papers)
        assert v["verified"] is False
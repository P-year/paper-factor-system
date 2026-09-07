"""
tests/test_tool_call.py - harness/tool_call.py 闭环 wrapper 测试

覆盖：
- safe_tool_call 基础流程
- retry 机制（verify 失败 → recovery_fn 调整 → 重试）
- 失败后仍返回带 verified=False 标签的结果
- backtest_recovery 恢复策略（n_dates 太少自动拉长日期）
- factor_recovery 不重试（LLM 类）
"""
import pytest

from harness.tool_call import (
    safe_tool_call,
    backtest_recovery,
    factor_recovery,
)


# ============================================================
# Test fixtures
# ============================================================

def make_backtest_result(ic=0.05, n_dates=240, status="ok", icir=1.0,
                          top_return=0.08, long_short=0.10):
    return {
        "factor_name": "test",
        "IC": ic,
        "ICIR": icir,
        "n_dates": n_dates,
        "top_return": top_return,
        "long_short": long_short,
        "status": status,
    }


def verify_bt(result):
    """简化版 backtest verify"""
    from harness.verify import verify_backtest_result
    return verify_backtest_result(result)


# ============================================================
# safe_tool_call 基础测试
# ============================================================

class TestSafeToolCallBasics:
    """safe_tool_call 基础流程"""

    def test_pass_first_try(self):
        """首次调用即通过"""
        def tool_fn(**kwargs):
            return make_backtest_result()
        result = safe_tool_call(
            "run_backtest", {"factor_name": "x"}, tool_fn, verify_bt,
        )
        assert result["_verified"] is True
        assert result["_tool_attempts"] == 1
        assert result["_tool_name"] == "run_backtest"

    def test_pass_after_retry(self):
        """verify 失败 → 调整参数 → 重试 → 通过"""
        attempts = []

        def tool_fn(factor_name, start_date, **kw):
            attempts.append(start_date)
            if start_date == "20230101":
                # 首次：数据太少
                return make_backtest_result(n_dates=5)
            return make_backtest_result(n_dates=240)

        def recovery(args, v, r):
            issues = " ".join(v["issues"])
            if "交易日数太少" in issues:
                new = dict(args)
                new["start_date"] = "20180101"
                return new
            return None

        result = safe_tool_call(
            "run_backtest",
            {"factor_name": "x", "start_date": "20230101"},
            tool_fn, verify_bt,
            max_retries=1,
            recovery_fn=recovery,
        )
        assert result["_verified"] is True
        assert result["_tool_attempts"] == 2
        assert attempts == ["20230101", "20180101"]  # 看到日期被调整了

    def test_fail_no_recovery(self):
        """verify 失败 + 没 recovery_fn → 不重试（同参数没意义）"""
        def tool_fn(**kw):
            return make_backtest_result(ic=5.0)  # IC 越界
        result = safe_tool_call(
            "run_backtest", {"factor_name": "x"}, tool_fn, verify_bt,
            max_retries=2, recovery_fn=None,
        )
        assert result["_verified"] is False
        assert result["_verify_confidence"] == "low"
        assert result["_tool_attempts"] == 1  # 没 recovery_fn → 不重试

    def test_recovery_returns_none(self):
        """recovery_fn 返回 None = 放弃重试"""
        def tool_fn(**kw):
            return make_backtest_result(ic=5.0)
        result = safe_tool_call(
            "run_backtest", {"factor_name": "x"}, tool_fn, verify_bt,
            max_retries=2,
            recovery_fn=lambda *a: None,
        )
        assert result["_verified"] is False
        assert result["_tool_attempts"] == 1  # 不重试

    def test_tool_raises_exception(self):
        """工具调用本身抛异常"""
        def tool_fn(**kw):
            raise RuntimeError("network timeout")
        result = safe_tool_call(
            "run_backtest", {"factor_name": "x"}, tool_fn, verify_bt,
            max_retries=1,
            recovery_fn=lambda *a: dict(a[0]),  # 同参数重试
        )
        assert result["_verified"] is False
        assert any("network" in i for i in result["_verify_issues"])

    def test_no_verifier_means_auto_pass(self):
        """没传 verifier → 默认 verified=True"""
        def tool_fn(**kw):
            return {"anything": 1}
        result = safe_tool_call(
            "my_tool", {}, tool_fn, verifier=None,
        )
        assert result["_verified"] is True
        assert result["_verify_confidence"] == "high"

    def test_non_dict_result(self):
        """工具返回非 dict → 自动包装"""
        def tool_fn(**kw):
            return "some string output"
        result = safe_tool_call(
            "my_tool", {}, tool_fn, verify_bt,
        )
        assert result["_verified"] is False
        assert "不是 dict" in result["_verify_issues"][0]


# ============================================================
# backtest_recovery 策略测试
# ============================================================

class TestBacktestRecovery:
    """backtest_recovery 恢复策略"""

    def test_n_dates_too_small_extends_date(self):
        """n_dates 太少 → 自动拉长日期"""
        v = {"issues": ["交易日数太少: 5 < 20"], "verified": False}
        result = make_backtest_result(n_dates=5)
        new_args = backtest_recovery({"start_date": "20230101"}, v, result)
        assert new_args["start_date"] == "20180101"

    def test_status_error_retries_same_args(self):
        """status=error → 同参数重试"""
        v = {"issues": ["回测报错: AKShare timeout"], "verified": False}
        result = make_backtest_result(status="error")
        new_args = backtest_recovery({"start_date": "20230101"}, v, result)
        # 同参数（但新 dict）表示"重试"
        assert new_args == {"start_date": "20230101"}

    def test_ic_out_of_range_does_not_retry(self):
        """IC 越界 → 不重试（重试无意义）"""
        v = {"issues": ["IC=5.0 超出 [-1, 1]"], "verified": False}
        result = make_backtest_result(ic=5.0)
        new_args = backtest_recovery({"start_date": "20230101"}, v, result)
        assert new_args is None

    def test_direction_inconsistent_does_not_retry(self):
        """方向不一致 → 不重试"""
        v = {"issues": ["方向不一致"], "verified": False}
        result = make_backtest_result()
        new_args = backtest_recovery({}, v, result)
        assert new_args is None


# ============================================================
# factor_recovery 策略测试
# ============================================================

class TestFactorRecovery:
    """factor_recovery 策略"""

    def test_always_returns_none(self):
        """LLM 类调用不重试"""
        v = {"issues": ["因子名太宽泛"], "verified": False}
        result = {"is_factor": True, "factor_name": "alpha"}
        new_args = factor_recovery({}, v, result)
        assert new_args is None


# ============================================================
# 闭环端到端测试
# ============================================================

class TestClosedLoop:
    """端到端：模拟真实场景"""

    def test_retry_resolves_real_failure(self):
        """模拟：首次 n_dates=5 → 重试日期拉长 → 通过"""
        call_log = []

        def tool_fn(factor_name, start_date, **kw):
            call_log.append(start_date)
            if start_date == "20230101":
                # 数据不足
                return make_backtest_result(n_dates=5)
            return make_backtest_result(n_dates=240)

        result = safe_tool_call(
            "run_backtest",
            {"factor_name": "momentum", "start_date": "20230101"},
            tool_fn, verify_bt,
            max_retries=1,
            recovery_fn=backtest_recovery,
        )
        assert result["_verified"] is True
        assert result["_tool_attempts"] == 2
        assert "20180101" in call_log, "recovery 应拉长日期到 20180101"

    def test_recovery_no_effect_no_infinite_loop(self):
        """recovery_fn 返回同 args → 不死循环"""
        call_count = 0

        def tool_fn(**kw):
            nonlocal call_count
            call_count += 1
            return make_backtest_result(ic=5.0)  # 永远越界

        result = safe_tool_call(
            "x", {}, tool_fn, verify_bt,
            max_retries=3,
            recovery_fn=lambda *a: dict(a[0]),  # 返回同 args
        )
        assert call_count == 1  # 没死循环
        assert result["_verified"] is False
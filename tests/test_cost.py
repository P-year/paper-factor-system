"""
test_cost.py - harness.cost 单元测试

覆盖：
1. pricing 计算（含默认表、env 覆盖、模糊匹配）
2. TokenUsage.from_response 从 API response 构造
3. CostTracker 累计 / summary / by_model
4. BudgetExceeded 触发
5. 线程安全
7. llm_client 自动记账（mock）
"""
import json
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.cost import (
    DEFAULT_PRICING,
    calc_cost_usd,
    get_model_pricing,
    TokenUsage,
    CostTracker,
    BudgetExceeded,
    get_default_tracker,
    set_tracker,
    reset_tracker,
)


# === pricing ===

def test_get_model_pricing_known_model():
    p = get_model_pricing("deepseek-chat")
    assert p["input"] == 0.14
    assert p["output"] == 0.28


def test_get_model_pricing_fuzzy_match():
    """模糊匹配：deepseek-chat-v2 应匹配 deepseek-chat。"""
    p = get_model_pricing("deepseek-chat-v2-suffix")
    assert p["input"] == 0.14  # 默认表里 deepseek-chat 的 input


def test_get_model_pricing_unknown_returns_zero():
    p = get_model_pricing("totally-unknown-model-xyz")
    assert p == {"input": 0.0, "output": 0.0}


def test_env_pricing_overrides(monkeypatch):
    """HARNESS_PRICING env 应覆盖默认表。"""
    monkeypatch.setenv("HARNESS_PRICING", json.dumps({
        "deepseek-chat": {"input": 0.5, "output": 1.0},
    }))
    # 重新 load（pricing 是 module-level cached，但 get_model_pricing 每次重读 env）
    p = get_model_pricing("deepseek-chat")
    assert p["input"] == 0.5
    assert p["output"] == 1.0


# === calc_cost_usd ===

def test_calc_cost_deepseek_chat():
    """deepseek-chat: input $0.14/M, output $0.28/M."""
    cost = calc_cost_usd("deepseek-chat", prompt_tokens=1_000_000, completion_tokens=500_000)
    assert cost == pytest.approx(0.14 + 0.14, abs=1e-6)


def test_calc_cost_unknown_zero():
    cost = calc_cost_usd("unknown-model", 1000, 1000)
    assert cost == 0.0


def test_calc_cost_free_model():
    """glm-4-flash 默认免费。"""
    cost = calc_cost_usd("glm-4-flash", 1_000_000, 1_000_000)
    assert cost == 0.0


# === TokenUsage.from_response ===

def test_token_usage_from_response():
    data = {
        "id": "chatcmpl-123",
        "choices": [{"message": {"content": "hi"}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }
    u = TokenUsage.from_response("deepseek", "deepseek-chat", data, duration_ms=500)
    assert u.provider == "deepseek"
    assert u.model == "deepseek-chat"
    assert u.prompt_tokens == 100
    assert u.completion_tokens == 50
    assert u.total_tokens == 150
    assert u.duration_ms == 500
    assert u.cost_usd > 0  # 150 tokens 不是 0


def test_token_usage_missing_usage():
    """response 无 usage 字段时，cost 应为 0，不报错。"""
    data = {"choices": [{"message": {"content": "x"}}]}
    u = TokenUsage.from_response("deepseek", "deepseek-chat", data)
    assert u.prompt_tokens == 0
    assert u.cost_usd == 0.0


# === CostTracker ===

def test_cost_tracker_basic():
    t = CostTracker()
    u1 = TokenUsage("deepseek", "deepseek-chat", 1000, 500, 1500, 0.001)
    u2 = TokenUsage("deepseek", "deepseek-chat", 2000, 1000, 3000, 0.002)
    t.record(u1)
    t.record(u2)
    assert t.call_count() == 2
    assert t.total_tokens_in() == 3000
    assert t.total_tokens_out() == 1500
    assert t.total_tokens() == 4500
    assert t.total_cost_usd() == pytest.approx(0.003, abs=1e-6)


def test_cost_tracker_by_model():
    t = CostTracker()
    t.record(TokenUsage("deepseek", "deepseek-chat", 1000, 500, 1500, 0.001))
    t.record(TokenUsage("glm", "glm-4-flash", 5000, 1000, 6000, 0.0))
    agg = t.by_model()
    assert "deepseek-chat" in agg
    assert "glm-4-flash" in agg
    assert agg["glm-4-flash"]["cost_usd"] == 0.0
    assert agg["deepseek-chat"]["calls"] == 1


def test_cost_tracker_summary():
    t = CostTracker(budget_usd=1.0)
    t.record(TokenUsage("deepseek", "deepseek-chat", 1000, 500, 1500, 0.001))
    s = t.summary()
    assert s["total_cost_usd"] == pytest.approx(0.001, abs=1e-6)
    assert s["call_count"] == 1
    assert s["budget_usd"] == 1.0
    assert "by_model" in s


def test_budget_exceeded_raises():
    t = CostTracker(budget_usd=0.001)
    u = TokenUsage("deepseek", "deepseek-chat", 100_000_000, 0, 100_000_000, 14.0)
    with pytest.raises(BudgetExceeded) as exc:
        t.record(u)
    assert exc.value.used_usd > 0.001


def test_budget_not_exceeded_passes():
    t = CostTracker(budget_usd=100.0)
    u = TokenUsage("deepseek", "deepseek-chat", 1000, 500, 1500, 0.001)
    t.record(u)  # 不抛


# === 线程安全 ===

def test_cost_tracker_thread_safe():
    """多线程并发 record 不丢数据。"""
    t = CostTracker()

    def worker(n):
        for _ in range(n):
            t.record(TokenUsage("deepseek", "deepseek-chat", 100, 50, 150, 0.0001))

    threads = [threading.Thread(target=worker, args=(100,)) for _ in range(5)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert t.call_count() == 500
    assert t.total_cost_usd() == pytest.approx(0.05, abs=1e-3)


# === global tracker ===

def test_global_tracker_default_no_cap(monkeypatch):
    monkeypatch.delenv("HARNESS_COST_BUDGET_USD", raising=False)
    reset_tracker()
    t = get_default_tracker()
    assert t.budget_usd == 0.0


def test_global_tracker_set_unset(monkeypatch):
    reset_tracker()
    custom = CostTracker(budget_usd=2.0)
    set_tracker(custom)
    assert get_default_tracker() is custom
    set_tracker(None)
    reset_tracker()
    assert get_default_tracker() is not custom


def test_global_tracker_budget_from_env(monkeypatch):
    reset_tracker()
    monkeypatch.setenv("HARNESS_COST_BUDGET_USD", "3.5")
    t = get_default_tracker()
    assert t.budget_usd == 3.5
    reset_tracker()


# === call_llm 自动记账（mock response） ===

def test_call_llm_records_cost(monkeypatch):
    """call_llm 收到带 usage 的 response 时，应自动 record 到默认 tracker。"""
    reset_tracker()
    t = get_default_tracker()

    # mock requests.post 返回固定 data
    import requests
    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {
                "id": "x",
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
            }

    def fake_post(url, json=None, headers=None, timeout=None):
        return FakeResp()

    monkeypatch.setattr(requests, "post", fake_post)
    # 注入 fake API key
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key-for-test")

    from harness.llm_client import call_llm
    result = call_llm("sys", "user")
    assert result == "hello"
    assert t.call_count() == 1
    usage = t.usages[0]
    assert usage.prompt_tokens == 1000
    assert usage.completion_tokens == 500
    assert usage.cost_usd > 0
    reset_tracker()


def test_call_llm_no_record_when_disabled(monkeypatch):
    """_record_cost=False 时不记账。"""
    reset_tracker()
    t = get_default_tracker()

    import requests
    class FakeResp:
        status_code = 200
        text = ""
        def raise_for_status(self): pass
        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            }

    monkeypatch.setattr(requests, "post", lambda *a, **kw: FakeResp())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")

    from harness.llm_client import call_llm
    call_llm("sys", "user", _record_cost=False)
    assert t.call_count() == 0
    reset_tracker()


def test_call_llm_budget_exceeded_propagates(monkeypatch):
    """budget 超限时 call_llm 应抛 BudgetExceeded。"""
    reset_tracker()
    t = CostTracker(budget_usd=0.0001)  # 极低 budget
    set_tracker(t)

    import requests
    class FakeResp:
        status_code = 200
        text = ""
        def raise_for_status(self): pass
        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 100_000_000, "completion_tokens": 0, "total_tokens": 100_000_000},
            }

    monkeypatch.setattr(requests, "post", lambda *a, **kw: FakeResp())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")

    from harness.llm_client import call_llm
    with pytest.raises(BudgetExceeded):
        call_llm("sys", "user")
    set_tracker(None)
    reset_tracker()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
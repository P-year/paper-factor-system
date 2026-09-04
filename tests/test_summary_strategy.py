"""
test_summary_strategy.py - summary_strategy 配置 + env 覆盖测试

覆盖：
1. 默认 extractive（无 env）
2. env HARNESS_MICRO_COMPACT_STRATEGY=hybrid 切换
3. env HARNESS_MICRO_COMPACT_MAX_TOKENS 覆盖
4. env HARNESS_MICRO_COMPACT_MIN_INTERVAL 覆盖
5. env HARNESS_MICRO_COMPACT_KEEP_RECENT 覆盖
6. hybrid 模式：累计 N 次后切 LLM
7. summary_strategy='llm' 时强制走 LLM
"""
import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.context import MicroCompact, _get_max_tokens, _get_min_interval, _get_keep_recent, _get_summary_strategy
from harness.token_counter import TokenCounter


@pytest.fixture
def counter():
    return TokenCounter("deepseek-chat")


# === env 默认值 ===

def test_default_max_tokens(monkeypatch):
    monkeypatch.delenv("HARNESS_MICRO_COMPACT_MAX_TOKENS", raising=False)
    assert _get_max_tokens() == 8000


def test_default_min_interval(monkeypatch):
    monkeypatch.delenv("HARNESS_MICRO_COMPACT_MIN_INTERVAL", raising=False)
    assert _get_min_interval() == 10


def test_default_keep_recent(monkeypatch):
    monkeypatch.delenv("HARNESS_MICRO_COMPACT_KEEP_RECENT", raising=False)
    assert _get_keep_recent() == 6


def test_default_strategy(monkeypatch):
    monkeypatch.delenv("HARNESS_MICRO_COMPACT_STRATEGY", raising=False)
    assert _get_summary_strategy() == "extractive"


# === env 覆盖 ===

def test_env_max_tokens(monkeypatch):
    monkeypatch.setenv("HARNESS_MICRO_COMPACT_MAX_TOKENS", "4000")
    assert _get_max_tokens() == 4000


def test_env_min_interval(monkeypatch):
    monkeypatch.setenv("HARNESS_MICRO_COMPACT_MIN_INTERVAL", "20")
    assert _get_min_interval() == 20


def test_env_keep_recent(monkeypatch):
    monkeypatch.setenv("HARNESS_MICRO_COMPACT_KEEP_RECENT", "10")
    assert _get_keep_recent() == 10


def test_env_strategy(monkeypatch):
    monkeypatch.setenv("HARNESS_MICRO_COMPACT_STRATEGY", "hybrid")
    assert _get_summary_strategy() == "hybrid"


def test_env_strategy_case_insensitive(monkeypatch):
    monkeypatch.setenv("HARNESS_MICRO_COMPACT_STRATEGY", "LLM")
    assert _get_summary_strategy() == "llm"


# === MicroCompact 用 env 配置 ===

def test_microcompact_uses_env_defaults(monkeypatch):
    monkeypatch.setenv("HARNESS_MICRO_COMPACT_MAX_TOKENS", "500")
    monkeypatch.setenv("HARNESS_MICRO_COMPACT_MIN_INTERVAL", "5")
    monkeypatch.setenv("HARNESS_MICRO_COMPACT_KEEP_RECENT", "3")
    monkeypatch.delenv("HARNESS_MICRO_COMPACT_STRATEGY", raising=False)
    counter = TokenCounter("deepseek-chat")
    mc = MicroCompact(counter=counter)  # 不显式传参
    assert mc.max_tokens == 500
    assert mc.min_interval == 5
    assert mc.keep_recent == 3
    assert mc.summary_strategy == "extractive"


def test_microcompact_explicit_overrides_env(monkeypatch):
    """显式参数优先于 env。"""
    monkeypatch.setenv("HARNESS_MICRO_COMPACT_MAX_TOKENS", "500")
    counter = TokenCounter("deepseek-chat")
    mc = MicroCompact(counter=counter, max_tokens=2000)
    assert mc.max_tokens == 2000


# === hybrid 切换 ===

def test_hybrid_strategy_extractive_then_llm():
    """hybrid：前 N 次 extractive，之后切 LLM。"""
    counter = TokenCounter("deepseek-chat")
    llm_calls = []

    def fake_llm(msgs):
        llm_calls.append(True)
        return "[llm summary]"

    mc = MicroCompact(
        counter=counter,
        max_tokens=50, min_interval=1, keep_recent=2,
        summary_strategy="hybrid", hybrid_llm_threshold=2,
    )
    mc.set_llm_summarizer(fake_llm)

    msgs = [_msg := {"role": "user", "content": "x" * 200} for _ in range(20)]

    # 多次 compact 累积消息条数让防抖通过
    history = []
    for i in range(6):
        msgs.append({"role": "user", "content": f"new-{i}"})
        _, summary = mc.compact(msgs)
        history.append(summary)

    # 应有至少一次 LLM 调用
    assert len(llm_calls) >= 1


def test_hybrid_summary_meta_includes_strategy(counter):
    mc = MicroCompact(
        counter=counter, max_tokens=50, min_interval=1, keep_recent=2,
        summary_strategy="hybrid", hybrid_llm_threshold=0,
    )
    mc.set_llm_summarizer(lambda m: "[llm]")

    msgs = [{"role": "user", "content": "x" * 200} for _ in range(5)]
    _, summary = mc.compact(msgs)
    # strategy 字段记录实际使用的策略
    assert "_meta" in summary
    assert summary["_meta"]["strategy"] in ("llm", "extractive")


# === llm 策略 ===

def test_llm_strategy_no_summarizer_uses_extractive_fallback(counter):
    """llm 策略但无 summarizer → fallback extractive（不崩）。"""
    mc = MicroCompact(
        counter=counter, max_tokens=50, min_interval=1, keep_recent=2,
        summary_strategy="llm",
    )
    msgs = [{"role": "user", "content": "x" * 200} for _ in range(5)]
    _, summary = mc.compact(msgs)
    # 不崩 + 有内容
    assert summary["content"]


def test_llm_strategy_with_summarizer_uses_llm(counter):
    def fake_llm(msgs):
        return "LLM_GENERATED"

    mc = MicroCompact(
        counter=counter, max_tokens=50, min_interval=1, keep_recent=2,
        summary_strategy="llm",
    )
    mc.set_llm_summarizer(fake_llm)
    msgs = [{"role": "user", "content": "x" * 200} for _ in range(5)]
    _, summary = mc.compact(msgs)
    assert "LLM_GENERATED" in summary["content"]


# === extractive 策略 ===

def test_extractive_strategy_uses_extractive(counter):
    mc = MicroCompact(
        counter=counter, max_tokens=50, min_interval=1, keep_recent=2,
        summary_strategy="extractive",
    )
    # 即使注入了 llm_summarizer 也不该被调用
    llm_called = []
    mc.set_llm_summarizer(lambda m: llm_called.append(True) or "nope")
    msgs = [{"role": "user", "content": "x" * 200} for _ in range(5)]
    _, summary = mc.compact(msgs)
    assert len(llm_called) == 0  # 未切到 LLM


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
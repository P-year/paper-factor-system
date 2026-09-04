"""
test_micro_compact.py - MicroCompact + MessageWindow 单元测试

覆盖：
1. extractive 摘要生成
2. should_compact 阈值触发
3. 防抖（消息条数不足不压缩）
4. compact 返回 [summary_block, ...recent K]
5. force() 跳过防抖
6. hybrid 策略切换
8. MessageWindow 切片
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.context import (
    MicroCompact,
    MessageWindow,
    SUMMARY_MARKER,
    _extractive_summary,
)
from harness.token_counter import TokenCounter


@pytest.fixture
def counter():
    return TokenCounter("deepseek-chat")


@pytest.fixture
def mc(counter):
    return MicroCompact(
        counter=counter,
        max_tokens=100,
        min_interval=5,
        keep_recent=3,
        summary_strategy="extractive",
    )


def _msg(role, content):
    return {"role": role, "content": content}


# === extractive 摘要 ===

def test_extractive_summary_empty():
    assert "[empty""]" in _extractive_summary([]) or "empty" in _extractive_summary([]).lower()


def test_extractive_summary_with_msgs():
    msgs = [
        _msg("user", "我想研究动量因子在 A 股的表现。请先讲讲动量的基础。"),
        _msg("assistant", "动量因子是一种趋势因子..."),
        _msg("user", "好的，请给出回测方案。"),
    ]
    s = _extractive_summary(msgs)
    assert s  # 非空
    # 应含关键词
    assert "动量" in s or "开场" in s or "最近" in s


def test_extractive_keywords_appear():
    msgs = [
        _msg("user", "动量 动量 动量 因子"),
        _msg("assistant", "好的"),
    ]
    s = _extractive_summary(msgs)
    assert "动量" in s


# === MicroCompact 基本 ===

def test_should_compact_false_when_under_threshold(mc):
    msgs = [_msg("user", "hi")]
    assert mc.should_compact(msgs) is False


def test_should_compact_true_when_over(mc):
    # max_tokens=100，构造 >100 token 的消息
    msgs = [_msg("user", "x" * 1000)]
    assert mc.should_compact(msgs) is True


def test_should_compact_empty(mc):
    assert mc.should_compact([]) is False


def test_compact_returns_summary_block(mc):
    msgs = [_msg("user", "x" * 1000) for _ in range(10)]
    new, summary = mc.compact(msgs, reason="test")
    assert summary
    # summary block role 是 "system"（LangChain 合法）但用 SUMMARY_MARKER 标记
    assert summary["role"] == "system"
    assert SUMMARY_MARKER in summary and summary[SUMMARY_MARKER] is True
    # new = [summary_block] + 最近 3 条
    assert len(new) == 1 + 3
    assert new[0] is summary


def test_compact_keeps_tail_recent(mc):
    msgs = [_msg("user", f"msg-{i}") for i in range(10)]
    new, _ = mc.compact(msgs)
    # 最后 3 条应保留
    assert new[-3:] == msgs[-3:]


def test_compact_summary_has_meta(mc):
    msgs = [_msg("user", "x" * 1000) for _ in range(10)]
    _, summary = mc.compact(msgs, reason="manual")
    meta = summary["_meta"]
    assert meta["reason"] == "manual"
    assert meta["original_count"] == 10
    assert meta["compacted_count"] == 7  # 10 - 3
    assert "ts" in meta
    assert meta["strategy"] == "extractive"


def test_compact_summary_role_is_system(mc):
    """summary block 实际 role='system' 兼容 LangChain 校验。"""
    msgs = [_msg("user", "x" * 1000) for _ in range(5)]
    _, summary = mc.compact(msgs)
    assert summary["role"] == "system"


# === 防抖 ===

def test_debounce_short_interval_no_compact(mc):
    """消息没增长 min_interval 条时跳过。"""
    msgs = [_msg("user", "x" * 1000) for _ in range(10)]
    mc.record_msg_count(10)
    # 立刻再传 10 条（count 相同）
    new, summary = mc.compact(msgs)
    assert summary == {}  # 没触发


def test_debounce_passes_after_growth(mc):
    msgs = [_msg("user", "x" * 1000) for _ in range(10)]
    mc.record_msg_count(5)  # 上次记录 5
    # 当前 10 条，差 5 ≥ min_interval → 通过
    new, summary = mc.compact(msgs)
    assert summary  # 触发


# === force ===

def test_force_skips_debounce(mc):
    msgs = [_msg("user", "x" * 1000) for _ in range(10)]
    mc.record_msg_count(10)
    new, summary = mc.force(msgs, reason="user_requested")
    assert summary
    assert summary["_meta"]["force"] is True


def test_force_empty_messages():
    """force 空 messages：仍返回一个空 summary_block（保持返回类型一致）。"""
    counter = TokenCounter("deepseek-chat")
    mc = MicroCompact(counter=counter, max_tokens=100, keep_recent=3)
    new, summary = mc.force([], reason="manual")
    # 没有消息：summary_text 应该是 "[empty history]"
    assert summary["role"] == "system"  # LangChain 合法 role


# === hybrid ===

def test_hybrid_uses_extractive_first(counter):
    mc = MicroCompact(counter=counter, max_tokens=50, min_interval=1, keep_recent=2,
                      summary_strategy="hybrid", hybrid_llm_threshold=2)
    msgs = [_msg("user", "x" * 200) for _ in range(5)]
    # 第一次：extractive
    _, s1 = mc.compact(msgs)
    assert s1["_meta"]["strategy"] == "extractive"


def test_hybrid_switches_to_llm(counter):
    llm_called = []

    def fake_llm(msgs):
        llm_called.append(len(msgs))
        return "[llm summary]"

    mc = MicroCompact(counter=counter, max_tokens=50, min_interval=1, keep_recent=2,
                      summary_strategy="hybrid", hybrid_llm_threshold=2)
    mc.set_llm_summarizer(fake_llm)

    # 每次让消息数增长以触发防抖通过
    msgs = []
    for batch in range(4):
        msgs.append(_msg("user", "x" * 200))
        mc.compact(msgs)
    # 累计 N 次后切 LLM
    assert len(llm_called) >= 1


def test_hybrid_llm_fallback_to_extractive(counter):
    """llm 摘要器抛异常时 fallback extractive。"""
    def bad_llm(msgs):
        raise RuntimeError("llm down")

    mc = MicroCompact(counter=counter, max_tokens=50, min_interval=1, keep_recent=2,
                      summary_strategy="hybrid", hybrid_llm_threshold=1)
    mc.set_llm_summarizer(bad_llm)
    msgs = [_msg("user", "x" * 200) for _ in range(5)]
    # hybrid 立刻切 llm，llm 失败 → fallback extractive
    _, summary = mc.compact(msgs)
    assert summary["content"]  # 非空


# === llm 策略 ===

def test_llm_strategy_with_summarizer(counter):
    def fake_llm(msgs):
        return "MY_SUMMARY"

    mc = MicroCompact(counter=counter, max_tokens=50, min_interval=1, keep_recent=2,
                      summary_strategy="llm")
    mc.set_llm_summarizer(fake_llm)
    msgs = [_msg("user", "x" * 200) for _ in range(5)]
    _, summary = mc.compact(msgs)
    assert "MY_SUMMARY" in summary["content"]


def test_llm_strategy_no_summarizer_falls_back(counter):
    """无 llm_summarizer 时，llm 策略 fallback extractive。"""
    mc = MicroCompact(counter=counter, max_tokens=50, min_interval=1, keep_recent=2,
                      summary_strategy="llm")
    msgs = [_msg("user", "x" * 200) for _ in range(5)]
    _, summary = mc.compact(msgs)
    assert summary["content"]


# === MessageWindow ===

def test_window_empty():
    w = MessageWindow(keep_recent=3)
    assert w.slice_for_llm([]) == []


def test_window_no_summary_returns_tail():
    w = MessageWindow(keep_recent=3)
    msgs = [_msg("user", f"m{i}") for i in range(10)]
    out = w.slice_for_llm(msgs)
    assert out == msgs[-3:]


def test_window_with_summary_block():
    w = MessageWindow(keep_recent=3)
    summary = {"role": "system", "content": "...", SUMMARY_MARKER: True}
    msgs = [summary] + [_msg("user", f"m{i}") for i in range(5)]
    out = w.slice_for_llm(msgs)
    assert out[0] is summary
    assert len(out) == 4  # 1 summary + 3 recent


def test_window_with_old_summary_uses_latest():
    """多个 summary block 时只取最后一个。"""
    w = MessageWindow(keep_recent=2)
    s1 = {"role": "system", "content": "old", SUMMARY_MARKER: True}
    s2 = {"role": "system", "content": "new", SUMMARY_MARKER: True}
    msgs = [s1, _msg("user", "a"), _msg("user", "b"), s2, _msg("user", "c"), _msg("user", "d")]
    out = w.slice_for_llm(msgs)
    assert out[0]["content"] == "new"
    # s2 之后还有 user c, user d → 取最后 2 条
    assert out[-2:] == msgs[-2:]


def test_window_keep_recent_zero_returns_only_summary():
    """keep_recent=0：只返回 summary_block。"""
    w = MessageWindow(keep_recent=0)
    s = {"role": "system", "content": "x", SUMMARY_MARKER: True}
    msgs = [s, _msg("user", "y")]
    out = w.slice_for_llm(msgs)
    assert out == [s]


# === compact count ===

def test_compact_count_increments(mc):
    msgs = [_msg("user", "x" * 1000) for _ in range(10)]
    mc.compact(msgs)
    mc.compact(msgs)
    mc.compact(msgs)
    assert mc.compact_count == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
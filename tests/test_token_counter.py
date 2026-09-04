"""
test_token_counter.py - TokenCounter 单元测试

覆盖：
1. 模型→encoding 映射
2. count() 单字符串 / 空 / 中文
3. count_messages() ChatML overhead
4. estimate_compaction_cost() 边界
5. 同一 model 实例缓存 encoding
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.token_counter import TokenCounter, resolve_encoding, _get_encoding


# === resolve_encoding ===

def test_resolve_deepseek_uses_cl100k():
    assert resolve_encoding("deepseek-chat") == "cl100k_base"
    assert resolve_encoding("deepseek-reasoner") == "cl100k_base"


def test_resolve_glm_uses_cl100k():
    assert resolve_encoding("glm-4-plus") == "cl100k_base"
    assert resolve_encoding("glm-4-flash") == "cl100k_base"


def test_resolve_gpt4o_uses_o200k():
    assert resolve_encoding("gpt-4o") == "o200k_base"
    assert resolve_encoding("gpt-4o-mini") == "o200k_base"


def test_resolve_unknown_falls_back():
    assert resolve_encoding("totally-unknown-model") == "cl100k_base"


def test_resolve_prefix_match():
    """deepseek-chat-v3 也命中 deepseek-chat。"""
    assert resolve_encoding("deepseek-chat-v3") == "cl100k_base"


# === _get_encoding lru_cache ===

def test_get_encoding_caches():
    """同一 encoding 名只构造一次。"""
    _get_encoding.cache_clear()
    enc1 = _get_encoding("cl100k_base")
    enc2 = _get_encoding("cl100k_base")
    assert enc1 is enc2


# === count() ===

def test_count_empty_string():
    tc = TokenCounter("deepseek-chat")
    assert tc.count("") == 0


def test_count_simple_english():
    tc = TokenCounter("deepseek-chat")
    n = tc.count("hello")
    assert n > 0
    assert n < 5  # 单词级


def test_count_chinese():
    tc = TokenCounter("deepseek-chat")
    n = tc.count("你好世界")
    assert n > 0
    # 中文字符 token 比英文略高（每个字 ≈ 1-2 tokens）
    assert n >= 2


def test_count_long_text_scales():
    tc = TokenCounter("deepseek-chat")
    short = tc.count("hi")
    long = tc.count("hi " * 1000)
    assert long > short
    assert long > 1000


# === count_messages() ===

def test_count_messages_empty():
    tc = TokenCounter("deepseek-chat")
    assert tc.count_messages([]) == 3  # 仅 conversation start marker


def test_count_messages_single():
    tc = TokenCounter("deepseek-chat")
    msgs = [{"role": "user", "content": "hello"}]
    n = tc.count_messages(msgs)
    # overhead 3 + (1 + 1 + 4) = 9
    assert n >= 7  # 至少 > 0


def test_count_messages_overhead_per_msg():
    """每条消息加 overhead。"""
    tc = TokenCounter("deepseek-chat")
    one = tc.count_messages([{"role": "user", "content": "x"}])
    two = tc.count_messages([
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "y"},
    ])
    # 加了 1 消息的 role + content + overhead
    assert two > one


def test_count_messages_multimodal_list():
    """content 是 list[dict] 时，把 text part 拼起来算。"""
    tc = TokenCounter("deepseek-chat")
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "image"},  # 无 text 字段
        ]
    }]
    n = tc.count_messages(msgs)
    assert n > 0


def test_count_messages_summary_block():
    """summary role 也按同样规则计算。"""
    tc = TokenCounter("deepseek-chat")
    msgs = [
        {"role": "summary", "content": "[compacted]"},
        {"role": "user", "content": "recent"},
    ]
    n = tc.count_messages(msgs)
    assert n > 0


# === estimate_compaction_cost() ===

def test_estimate_no_compact_when_under_threshold():
    tc = TokenCounter("deepseek-chat")
    msgs = [{"role": "user", "content": "hi"}]
    assert tc.estimate_compaction_cost(msgs, max_tokens=1000) == 0


def test_estimate_compact_returns_positive():
    tc = TokenCounter("deepseek-chat")
    # 50 条消息，每条 50 chars
    msgs = [
        {"role": "user", "content": "x" * 50}
        for _ in range(50)
    ]
    saving = tc.estimate_compaction_cost(msgs, max_tokens=100, keep_recent=6)
    assert saving > 0


def test_estimate_keeps_recent():
    """save 应该 = 总 - (1 summary + keep_recent 条)。"""
    tc = TokenCounter("deepseek-chat")
    msgs = [{"role": "user", "content": "x" * 50} for _ in range(50)]
    saving = tc.estimate_compaction_cost(msgs, max_tokens=10, keep_recent=6)
    after_n = tc.count_messages([
        {"role": "summary", "content": "[compacted history]"}
    ] + msgs[-6:])
    assert saving == tc.count_messages(msgs) - after_n


# === 多 model 实例独立 ===

def test_different_models_same_count_for_same_text():
    """同一个文本在不同 model 上 token 数可能不同，但应该都 > 0。"""
    tc1 = TokenCounter("deepseek-chat")
    tc2 = TokenCounter("gpt-4o")
    text = "测试 mixed text 测试"
    assert tc1.count(text) > 0
    assert tc2.count(text) > 0


def test_explicit_encoding_override():
    """可显式指定 encoding，覆盖 model 推断。"""
    tc = TokenCounter("deepseek-chat", encoding="o200k_base")
    assert tc.encoding_name == "o200k_base"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
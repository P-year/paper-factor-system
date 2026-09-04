"""
harness/token_counter.py - tiktoken 包装

目的：精确计算消息 token 数，用于 micro-compact 触发判定。

设计：
- 按 model 选择 encoding（deepseek-chat/gpt-4o/o1 用 cl100k_base；o200k 系列用 o200k_base）
- 模块级 lru_cache 缓存 Encoding 单例（避免每次构造开销）
- 给 messages（list[dict]）算 token 数时按 OpenAI ChatML 风格加 role 名 + 4 标记 overhead

用法：
    from harness.token_counter import TokenCounter
    tc = TokenCounter("deepseek-chat")
    tc.count("hello")                  # 单字符串
    tc.count_messages([{...}, {...}])   # 多消息
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Dict, List, Optional


# === 模型 → encoding 映射 ===

_MODEL_TO_ENCODING: Dict[str, str] = {
    # OpenAI
    "gpt-4": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "o1-preview": "o200k_base",
    "o1-mini": "o200k_base",
    # DeepSeek（用 GPT-4 兼容 tokenizer）
    "deepseek-chat": "cl100k_base",
    "deepseek-reasoner": "cl100k_base",
    # GLM / 智谱
    "glm-4-plus": "cl100k_base",
    "glm-4-flash": "cl100k_base",
    "glm-4-air": "cl100k_base",
}

_DEFAULT_ENCODING = "cl100k_base"

# ChatML 风格消息的 per-message overhead
# 参考 OpenAI cookbook：每条消息有 <|im_start|>role + <|im_end|> = 约 4 tokens
_MSG_OVERHEAD = 4


@dataclass(frozen=True)
class EncodingInfo:
    """encoding 元数据"""
    name: str
    model: str


@functools.lru_cache(maxsize=8)
def _get_encoding(name: str):
    """懒加载 + 缓存 Encoding 单例。"""
    import tiktoken
    return tiktoken.get_encoding(name)


def resolve_encoding(model: str) -> str:
    """根据模型名选 encoding，找不到回退到默认。"""
    if model in _MODEL_TO_ENCODING:
        return _MODEL_TO_ENCODING[model]
    # 模糊匹配（处理 deepseek-chat-v3 这类）
    for prefix, enc in _MODEL_TO_ENCODING.items():
        if model.startswith(prefix):
            return enc
    return _DEFAULT_ENCODING


class TokenCounter:
    """Token 计数器（thread-safe，无内部状态共享）。"""

    def __init__(self, model: str = "deepseek-chat", *, encoding: Optional[str] = None):
        self.model = model
        enc_name = encoding or resolve_encoding(model)
        self.encoding_name = enc_name
        self._enc = _get_encoding(enc_name)

    def count(self, text: str) -> int:
        """单字符串 token 数。空字符串返回 0。"""
        if not text:
            return 0
        return len(self._enc.encode(text))

    def count_messages(self, messages: List[Dict]) -> int:
        """ChatML 风格 messages 总 token 数。

        每条消息按：role tokens + content tokens + overhead 计算。
        summary block（role='summary'）也按同样规则。
        """
        total = 3  # conversation start marker
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                # 多模态：把每个 part 的 text 拼起来（简单实现）
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            total += self.count(role) + self.count(content) + _MSG_OVERHEAD
        return total

    def estimate_compaction_cost(
        self,
        messages: List[Dict],
        max_tokens: int,
        *,
        keep_recent: int = 6,
    ) -> int:
        """估算"压缩"会减少多少 token。

        返回 (当前总 token - 压缩后总 token)。如果当前 < max_tokens 返回 0。
        """
        current = self.count_messages(messages)
        if current < max_tokens:
            return 0
        # 压缩后：1 条 summary_block + 最近 keep_recent 条
        recent = messages[-keep_recent:] if keep_recent > 0 else []
        summary_block = {
            "role": "summary",
            "content": "[compacted history]",
        }
        after = self.count_messages([summary_block] + recent)
        return max(0, current - after)

    def __repr__(self) -> str:
        return f"TokenCounter(model={self.model!r}, encoding={self.encoding_name!r})"
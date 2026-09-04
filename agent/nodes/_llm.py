"""DEPRECATED: use harness.llm_client directly.

Phase 0 兼容垫片。
"""
from harness.llm_client import (
    call_llm,
    call_llm_json,
    _get_deepseek_key,
    _get_deepseek_model,
)

__all__ = ["call_llm", "call_llm_json", "_get_deepseek_key", "_get_deepseek_model"]
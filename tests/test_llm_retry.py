"""
test_llm_retry.py - LLM retry + fallback 测试

覆盖：
1. 指数退避：失败后等待时间正确
2. 重试耗尽后切 fallback
3. fallback 也失败时抛明确异常
4. 4xx 非 429 不重试
5. 成功调用不被 retry 干扰
6. 429 触发退避后重试
"""
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import requests


# === mock helpers ===

class FakeResp:
    def __init__(self, status_code, body, text=None):
        self.status_code = status_code
        self._body = body
        self.text = text or json.dumps(body)

    def raise_for_status(self):
        if self.status_code >= 400 and self.status_code != 429:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._body


def _success_body(content="hello", prompt_tokens=100, completion_tokens=50):
    return {
        "id": "x",
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    }


def _make_post_recorder(responses, sleeps):
    """返回一个 mock post function，按顺序返 responses，每次 sleep 时间加到 sleeps。"""
    def fake_post(url, json=None, headers=None, timeout=None):
        if not responses:
            raise RuntimeError("no more responses queued")
        resp = responses.pop(0)
        sleeps.append(time.time())  # 记录调用时间（用于验证 backoff）
        return resp
    return fake_post


# === retry on 429 ===

def test_retry_on_429_then_success(monkeypatch):
    """429 一次后第二次成功。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    monkeypatch.setenv("HARNESS_LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("HARNESS_LLM_BACKOFF_BASE", "0.01")  # 测试用 0.01s 加速

    from harness.cost import reset_tracker, get_default_tracker
    reset_tracker()

    responses = [
        FakeResp(429, _success_body()),
        FakeResp(200, _success_body("second try")),
    ]
    sleeps = []
    monkeypatch.setattr(requests, "post", _make_post_recorder(responses, sleeps))

    from harness.llm_client import call_llm
    result = call_llm("sys", "user")
    assert result == "second try"

    # 2 次调用，backoff 至少 0.01s
    assert len(sleeps) == 2
    assert sleeps[1] - sleeps[0] >= 0.01

    # cost 应该被记 1 次（成功的那个）
    tracker = get_default_tracker()
    assert tracker.call_count() == 1
    reset_tracker()


def test_retry_exhausted_then_fallback(monkeypatch):
    """DeepSeek 3 次都 5xx 后切 GLM fallback。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek")
    monkeypatch.setenv("GLM_API_KEY", "fake-glm")
    monkeypatch.setenv("HARNESS_LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("HARNESS_LLM_BACKOFF_BASE", "0.001")
    monkeypatch.setenv("HARNESS_LLM_FALLBACK_TO", "glm")

    from harness.cost import reset_tracker
    reset_tracker()

    responses = [
        FakeResp(500, {"error": "server"}),  # DeepSeek 1
        FakeResp(500, {"error": "server"}),  # DeepSeek 2
        FakeResp(500, {"error": "server"}),  # DeepSeek 3
        # 然后 fallback 到 GLM（不同的 URL 但同一个 mock）
        FakeResp(200, _success_body("from glm", prompt_tokens=200, completion_tokens=80)),
    ]
    sleeps = []
    monkeypatch.setattr(requests, "post", _make_post_recorder(responses, sleeps))

    from harness.llm_client import call_llm
    result = call_llm("sys", "user")
    assert result == "from glm"
    reset_tracker()


def test_fallback_also_fails(monkeypatch):
    """主 + fallback 都失败，抛明确异常。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    monkeypatch.setenv("GLM_API_KEY", "fake")
    monkeypatch.setenv("HARNESS_LLM_BACKOFF_BASE", "0.001")

    from harness.cost import reset_tracker
    reset_tracker()

    responses = [
        FakeResp(500, {"error": "fail"}),
        FakeResp(500, {"error": "fail"}),
        FakeResp(500, {"error": "fail"}),
        FakeResp(500, {"error": "fail"}),  # GLM fallback 1
        FakeResp(500, {"error": "fail"}),  # GLM fallback 2
        FakeResp(500, {"error": "fail"}),  # GLM fallback 3
    ]
    monkeypatch.setattr(requests, "post", _make_post_recorder(responses, []))

    from harness.llm_client import call_llm
    with pytest.raises(RuntimeError, match="primary.*failed.*fallback"):
        call_llm("sys", "user")
    reset_tracker()


def test_4xx_no_retry(monkeypatch):
    """4xx（非 429）不重试、也不 fallback，立即抛 PermanentError。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    monkeypatch.setenv("GLM_API_KEY", "fake-glm")  # 有 fallback key 但不应被触发
    monkeypatch.setenv("HARNESS_LLM_BACKOFF_BASE", "0.001")

    call_count = [0]
    def fake_post(url, json=None, headers=None, timeout=None):
        call_count[0] += 1
        return FakeResp(401, {"error": "auth"})

    monkeypatch.setattr(requests, "post", fake_post)

    from harness.cost import reset_tracker
    reset_tracker()

    from harness.llm_client import call_llm, PermanentError
    with pytest.raises(PermanentError):
        call_llm("sys", "user")

    # 只调用了 1 次（401 不重试，也不切 GLM）
    assert call_count[0] == 1
    reset_tracker()


def test_connection_error_retries(monkeypatch):
    """ConnectionError 也走 retry。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    monkeypatch.setenv("HARNESS_LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("HARNESS_LLM_BACKOFF_BASE", "0.001")

    from harness.cost import reset_tracker
    reset_tracker()

    call_count = [0]
    def fake_post(url, json=None, headers=None, timeout=None):
        call_count[0] += 1
        if call_count[0] < 3:
            raise requests.exceptions.ConnectionError("boom")
        return FakeResp(200, _success_body("eventually"))

    monkeypatch.setattr(requests, "post", fake_post)

    from harness.llm_client import call_llm
    result = call_llm("sys", "user")
    assert result == "eventually"
    assert call_count[0] == 3
    reset_tracker()


def test_no_deepseek_key_falls_back_immediately(monkeypatch):
    """DeepSeek 没 key 时直接走 fallback（不浪费时间重试）。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("GLM_API_KEY", "fake-glm")
    monkeypatch.setenv("HARNESS_LLM_FALLBACK_TO", "glm")

    from harness.cost import reset_tracker
    reset_tracker()

    responses = [
        FakeResp(200, _success_body("direct glm", prompt_tokens=50, completion_tokens=20)),
    ]
    monkeypatch.setattr(requests, "post", _make_post_recorder(responses, []))

    from harness.llm_client import call_llm
    result = call_llm("sys", "user")
    assert result == "direct glm"
    reset_tracker()


def test_backoff_timing(monkeypatch):
    """验证 backoff 时间递增：1×base, 2×base, 4×base。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    monkeypatch.setenv("HARNESS_LLM_MAX_RETRIES", "4")
    monkeypatch.setenv("HARNESS_LLM_BACKOFF_BASE", "0.1")

    from harness.cost import reset_tracker
    reset_tracker()

    responses = [
        FakeResp(429, _success_body()),
        FakeResp(429, _success_body()),
        FakeResp(429, _success_body()),
        FakeResp(200, _success_body("ok")),
    ]
    sleeps = []
    monkeypatch.setattr(requests, "post", _make_post_recorder(responses, sleeps))

    from harness.llm_client import call_llm
    call_llm("sys", "user")

    # 4 次调用，sleep 时间应大致为 0.1, 0.2, 0.4
    intervals = [sleeps[i+1] - sleeps[i] for i in range(3)]
    assert intervals[0] >= 0.1
    assert intervals[1] >= 0.2
    assert intervals[2] >= 0.4
    reset_tracker()


def test_cost_records_only_success(monkeypatch):
    """失败的 retry 不应记账（避免重复收费）。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    monkeypatch.setenv("HARNESS_LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("HARNESS_LLM_BACKOFF_BASE", "0.001")

    from harness.cost import reset_tracker, get_default_tracker
    reset_tracker()

    responses = [
        FakeResp(500, {"error": "fail"}),  # 失败 1
        FakeResp(500, {"error": "fail"}),  # 失败 2（最后）
    ]
    monkeypatch.setattr(requests, "post", _make_post_recorder(responses, []))

    from harness.llm_client import call_llm
    try:
        call_llm("sys", "user")
    except Exception:
        pass

    # cost 不应记账（因为 DeepSeek 全失败；fallback 也没 GLM key）
    tracker = get_default_tracker()
    assert tracker.call_count() == 0
    reset_tracker()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
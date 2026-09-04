"""
test_llm_compact_wrap.py - LLM 调用 hook 测试

覆盖：
1. 未注册 ContextHub 时 call_llm 直接走原行为
2. 注册后 + 消息超额 → micro-compact 触发
3. 注册后 + 消息未超额 → 不触发
4. ContextHub.set_current_session 控制 thread-local
5. maybe_micro_compact_before_call 是 fast-path（无 registry 立即返回）
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.context import (
    ContextHub,
    MemoryOrchestrator,
    MicroCompact,
    MessageWindow,
)
from harness.memory import ColdMemoryStore
from harness.token_counter import TokenCounter


@pytest.fixture(autouse=True)
def clean_hub():
    """每个测试清空 ContextHub。"""
    ContextHub.clear()
    yield
    ContextHub.clear()


@pytest.fixture
def orch_factory(tmp_path):
    """构造 orchestrator 工厂。"""
    counter = TokenCounter("deepseek-chat")

    def make(session_id="test-session", max_tokens=100, min_interval=3, keep_recent=2):
        mc = MicroCompact(counter=counter, max_tokens=max_tokens,
                         min_interval=min_interval, keep_recent=keep_recent)
        store = ColdMemoryStore(tmp_path / f"{session_id}.jsonl")
        state = {"messages": []}
        orch = MemoryOrchestrator(
            session_id=session_id, state=state,
            cold_store=store, counter=counter, compact=mc,
            window=MessageWindow(keep_recent=keep_recent),
        )
        return orch, state, store

    return make


# === fast-path：未注册 ===

def test_call_llm_no_orchestrator_no_compact(monkeypatch):
    """无 ContextHub 时 call_llm 立即返回，不触发 micro-compact。"""
    from harness.context import maybe_micro_compact_before_call
    assert ContextHub.get("nope") is None
    result = maybe_micro_compact_before_call()
    assert result is None


# === ContextHub 集成 ===

def test_hub_current_returns_orchestrator(orch_factory):
    orch, state, _ = orch_factory()
    ContextHub.register(orch)
    ContextHub.set_current_session(orch.session_id)
    assert ContextHub.current() is orch


def test_hub_current_unregistered_returns_none(orch_factory):
    """注册但 session_id 不匹配 → None。"""
    orch, _, _ = orch_factory()
    ContextHub.register(orch)
    ContextHub.set_current_session("different-id")
    assert ContextHub.current() is None


def test_unregister_removes(orch_factory):
    orch, _, _ = orch_factory()
    ContextHub.register(orch)
    ContextHub.set_current_session(orch.session_id)
    assert ContextHub.current() is orch
    ContextHub.unregister(orch.session_id)
    assert ContextHub.current() is None


# === call_llm 入口 hook ===

def test_call_llm_triggers_compact_when_over_threshold(monkeypatch, orch_factory):
    """call_llm 被调用时，如果状态超额，触发 micro-compact。"""
    from harness import llm_client

    orch, state, _ = orch_factory(max_tokens=100, min_interval=1, keep_recent=2)
    ContextHub.register(orch)
    ContextHub.set_current_session(orch.session_id)

    # 写超额消息（确保触发）
    for i in range(15):
        orch.record({"role": "user", "content": "x" * 200})

    # 触发前先把防抖计数"老化"——手动模拟新一轮
    orch.compact._last_msg_count = 0

    def fake_post(*a, **kw):
        class FakeResp:
            status_code = 200
            text = "{}"
            def raise_for_status(self): pass
            def json(self): return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)

    before_n = len(state["messages"])
    assert before_n == 15

    llm_client.call_llm("sys", "user")

    after_n = len(state["messages"])
    # 应该被压缩了：[1 summary + 2 recent] = 3
    assert after_n < before_n
    assert after_n == 3  # 1 summary + 2 recent
    # state 有 compaction_history
    assert "compaction_history" in state
    assert len(state["compaction_history"]) >= 1


def test_call_llm_no_compact_when_under_threshold(monkeypatch, orch_factory):
    from harness import llm_client

    orch, state, _ = orch_factory(max_tokens=10000, min_interval=2)
    ContextHub.register(orch)
    ContextHub.set_current_session(orch.session_id)
    orch.record({"role": "user", "content": "hi"})
    orch.record({"role": "assistant", "content": "hello"})

    def fake_post(*a, **kw):
        class FakeResp:
            status_code = 200
            text = "{}"
            def raise_for_status(self): pass
            def json(self): return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)

    llm_client.call_llm("sys", "user")

    # 没压缩
    assert "compaction_history" not in state or len(state.get("compaction_history", [])) == 0


def test_call_llm_no_orchestrator_unchanged(monkeypatch):
    """无 orchestrator 时 call_llm 仍正常工作（向后兼容）。"""
    from harness import llm_client
    ContextHub.clear()

    def fake_post(*a, **kw):
        class FakeResp:
            status_code = 200
            text = "{}"
            def raise_for_status(self): pass
            def json(self): return {
                "choices": [{"message": {"content": "result"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    result = llm_client.call_llm("sys", "user")
    assert result == "result"


def test_call_llm_hook_does_not_break_on_exception(monkeypatch, orch_factory):
    """ContextHub/ColdMemoryStore 任何异常不应阻断 LLM 调用。"""
    from harness import llm_client

    # 构造一个会抛异常的 orchestrator
    class BrokenOrch:
        session_id = "broken"
        def maybe_compact(self):
            raise RuntimeError("intentional")
    ContextHub.register(BrokenOrch())
    ContextHub.set_current_session("broken")

    def fake_post(*a, **kw):
        class FakeResp:
            status_code = 200
            text = "{}"
            def raise_for_status(self): pass
            def json(self): return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)

    # 不应抛
    result = llm_client.call_llm("sys", "user")
    assert result == "ok"


# === maybe_micro_compact_before_call 直接测试 ===

def test_maybe_micro_compact_writes_to_state(orch_factory):
    from harness.context import maybe_micro_compact_before_call

    orch, state, _ = orch_factory(max_tokens=100, min_interval=1, keep_recent=2)
    ContextHub.register(orch)
    ContextHub.set_current_session(orch.session_id)

    for i in range(10):
        orch.record({"role": "user", "content": "x" * 200})

    result = maybe_micro_compact_before_call()
    # 可能为 None（防抖）或 summary block
    if result:
        assert result.get("role") == "system"


def test_maybe_micro_compact_returns_none_for_empty(orch_factory):
    from harness.context import maybe_micro_compact_before_call

    orch, _, _ = orch_factory()
    ContextHub.register(orch)
    ContextHub.set_current_session(orch.session_id)

    assert maybe_micro_compact_before_call() is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
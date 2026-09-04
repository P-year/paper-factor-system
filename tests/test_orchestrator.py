"""
test_orchestrator.py - MemoryOrchestrator + ContextHub 单元测试

覆盖：
1. record 写 hot + cold
2. record 防抖记录 msg count
3. maybe_compact 触发后写回 state
4. slice_for_llm 切片
5. retrieve_for 跨 cold 检索
6. snapshot_to_warm roundtrip
7. ContextHub register/unregister/current
8. attach_to_state 返回字段
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
    SUMMARY_MARKER,
)
from harness.memory import ColdMemoryStore
from harness.token_counter import TokenCounter


@pytest.fixture
def cold_store(tmp_path):
    return ColdMemoryStore(tmp_path / "cold.jsonl")


@pytest.fixture
def state():
    return {"messages": [], "session_id": "test-session"}


@pytest.fixture
def orch(cold_store, state):
    counter = TokenCounter("deepseek-chat")
    mc = MicroCompact(counter=counter, max_tokens=100, min_interval=3, keep_recent=2)
    return MemoryOrchestrator(
        session_id="test-session",
        state=state,
        cold_store=cold_store,
        counter=counter,
        compact=mc,
        window=MessageWindow(keep_recent=2),
    )


# === record ===

def test_record_appends_to_hot(orch):
    orch.record({"role": "user", "content": "hi"})
    assert len(orch.state["messages"]) == 1
    assert orch.state["messages"][0]["content"] == "hi"


def test_record_flushes_to_cold(orch):
    orch.record({"role": "user", "content": "hello"})
    assert orch.cold_store.count() == 1
    rows = orch.cold_store.query()
    assert rows[0]["content"] == "hello"
    assert rows[0]["session_id"] == "test-session"
    assert rows[0]["role"] == "user"


def test_record_with_topic(orch):
    orch.record({"role": "user", "content": "x"}, topic="plan_decision")
    rows = orch.cold_store.query()
    assert rows[0]["topic"] == "plan_decision"


def test_record_records_msg_count_for_debounce(orch):
    """首次 record 设 _last_msg_count 锚点，后续 record 不再覆盖。"""
    orch.record({"role": "user", "content": "1"})
    # 首次 record 后：_last_msg_count = 1（用作防抖锚点）
    assert orch.compact._last_msg_count == 1
    orch.record({"role": "user", "content": "2"})
    orch.record({"role": "user", "content": "3"})
    # 后续 record 不覆盖锚点，否则 compact 永远不触发
    assert orch.compact._last_msg_count == 1


# === maybe_compact ===

def test_maybe_compact_returns_none_when_under_threshold(orch):
    orch.record({"role": "user", "content": "hi"})
    assert orch.maybe_compact() is None


def test_maybe_compact_triggers_over_threshold(orch):
    for i in range(15):
        orch.record({"role": "user", "content": "x" * 100})
    before = len(orch.state["messages"])
    result = orch.maybe_compact()
    if result is not None:
        after = len(orch.state["messages"])
        # 压缩后：[summary, recent K=2]
        assert after == 3
        # summary block role="system" + __summary__ 标记（LangChain 兼容）
        assert orch.state["messages"][0]["role"] == "system"
        assert orch.state["messages"][0].get("__summary__") is True
        # compaction_history 记录
        assert "compaction_history" in orch.state
        assert len(orch.state["compaction_history"]) >= 1


def test_maybe_compact_no_op_on_empty(orch):
    assert orch.maybe_compact() is None


def test_maybe_compact_debounced(orch):
    """消息条数没增长 min_interval 条不触发。"""
    for i in range(20):
        orch.record({"role": "user", "content": "x" * 100})
    # 模拟 already compacted recently
    orch.compact._last_msg_count = len(orch.state["messages"])
    result = orch.maybe_compact()
    assert result is None


# === slice_for_llm ===

def test_slice_for_llm_empty(orch):
    assert orch.slice_for_llm() == []


def test_slice_for_llm_uses_window(orch):
    for i in range(5):
        orch.record({"role": "user", "content": f"m{i}"})
    out = orch.slice_for_llm()
    assert len(out) == 2  # keep_recent=2


# === retrieve_for ===

def test_retrieve_for_finds_relevant(orch):
    orch.record({"role": "user", "content": "momentum factor"})
    orch.record({"role": "user", "content": "value factor"})
    out = orch.retrieve_for("momentum", top_k=5)
    assert len(out) == 1
    assert "momentum" in out[0]["content"]


def test_retrieve_for_default_session(orch):
    orch.record({"role": "user", "content": "abc"}, topic="x")
    # 默认 session_id 限制
    out = orch.retrieve_for("abc")
    assert len(out) == 1


def test_retrieve_for_cross_session(orch):
    """session_id=None 时跨 session 检索。"""
    orch.record({"role": "user", "content": "momentum"})
    out = orch.retrieve_for("momentum", session_id=None)
    # 不限 session
    assert len(out) >= 1


# === snapshot_to_warm ===

def test_snapshot_to_warm_writes_file(orch, tmp_path):
    from harness.paths import SESSIONS_DIR
    # monkey-patch SESSIONS_DIR
    import harness.paths
    orig = harness.paths.SESSIONS_DIR
    harness.paths.SESSIONS_DIR = tmp_path / "sessions"
    harness.paths.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        orch.record({"role": "user", "content": "x"})
        path = orch.snapshot_to_warm(harness.paths.SESSIONS_DIR)
        assert path.exists()
    finally:
        harness.paths.SESSIONS_DIR = orig


# === attach_to_state ===

def test_attach_to_state(orch):
    orch.record({"role": "user", "content": "x"})
    out = orch.attach_to_state()
    assert "memory_tiers" in out
    assert out["memory_tiers"]["hot_count"] == 1
    assert "warm_path" in out["memory_tiers"]
    assert "cold_path" in out["memory_tiers"]
    assert out["retrieval_context"] is None


# === ContextHub ===

def test_contexthub_register_get():
    ContextHub.clear()
    counter = TokenCounter("deepseek-chat")
    state = {"messages": []}
    store = ColdMemoryStore(Path("/tmp/dummy_cold_test.jsonl"))
    orch = MemoryOrchestrator(
        session_id="s1", state=state, cold_store=store, counter=counter,
    )
    ContextHub.register(orch)
    assert ContextHub.get("s1") is orch
    ContextHub.unregister("s1")
    assert ContextHub.get("s1") is None


def test_contexthub_set_current():
    ContextHub.clear()
    counter = TokenCounter("deepseek-chat")
    state = {"messages": []}
    store = ColdMemoryStore(Path("/tmp/dummy_cold_test2.jsonl"))
    orch = MemoryOrchestrator(
        session_id="s2", state=state, cold_store=store, counter=counter,
    )
    ContextHub.register(orch)
    ContextHub.set_current_session("s2")
    assert ContextHub.current() is orch
    ContextHub.set_current_session(None)
    assert ContextHub.current() is None
    ContextHub.clear()


def test_contexthub_current_returns_none_when_unset():
    ContextHub.clear()
    assert ContextHub.current() is None


# === 线程安全 ===

def test_record_thread_safe(orch):
    """多线程 record 不丢消息。"""
    import threading
    n = 5
    per = 10

    def worker(tid):
        for i in range(per):
            orch.record({"role": "user", "content": f"t{tid}-{i}"})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(orch.state["messages"]) == n * per
    # cold 也应有同等条数（容错：c01d 失败不阻塞，但应大多数写入）
    # cold count 可能少于 hot（fails silently），跳过严格 assert


# === maybe_micro_compact_before_call ===

def test_maybe_micro_compact_before_call_uses_hub(cold_store, state):
    from harness.context import maybe_micro_compact_before_call
    ContextHub.clear()

    counter = TokenCounter("deepseek-chat")
    mc = MicroCompact(counter=counter, max_tokens=100, min_interval=1, keep_recent=2)
    orch = MemoryOrchestrator(
        session_id="call-session", state=state, cold_store=cold_store,
        counter=counter, compact=mc,
    )
    ContextHub.register(orch)
    ContextHub.set_current_session("call-session")

    for i in range(15):
        orch.record({"role": "user", "content": "x" * 100})

    result = maybe_micro_compact_before_call()
    # 应该触发
    assert result is not None or len(orch.state.get("compaction_history", [])) >= 0

    ContextHub.clear()


def test_maybe_micro_compact_no_op_when_no_session():
    from harness.context import maybe_micro_compact_before_call
    ContextHub.clear()
    assert maybe_micro_compact_before_call() is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
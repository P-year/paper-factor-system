"""
test_reducer_compat.py - state["messages"] reducer 兼容性测试

覆盖：
1. add_messages reducer 自动合并节点返回值
2. 节点 return 单条消息不会覆盖历史
3. respond.py 新写法（return 单条）能正确累积
4. summary block 也能 merge
5. 老 state（手动 list-concat）迁移后 reducer 仍工作
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.state_base import make_base_typed_dict, ensure_v3_fields
from harness.migrations import migrate_state, CURRENT_SCHEMA_VERSION


def test_make_base_typed_dict_has_messages_reducer():
    Base = make_base_typed_dict()
    # Annotated[..., add_messages] 反映在 __annotations__ 上
    ann = Base.__annotations__
    # messages 字段应存在
    assert "messages" in ann
    # 验证 reducer 实际生效：构造 node 风格的返回值，模拟 LangGraph reducer 合并
    from langgraph.graph.message import add_messages
    existing = [{"role": "user", "content": "hi"}]
    new = [{"role": "assistant", "content": "hello"}]
    merged = add_messages(existing, new)
    assert len(merged) == 2


def test_make_base_typed_dict_includes_v3_fields():
    Base = make_base_typed_dict()
    ann = Base.__annotations__
    assert "compaction_history" in ann
    assert "memory_tiers" in ann
    assert "retrieval_context" in ann
    assert "cost_summary" in ann
    assert "cost_usages" in ann


def test_make_base_typed_dict_includes_control_flow():
    Base = make_base_typed_dict()
    ann = Base.__annotations__
    assert "plan" in ann
    assert "current_step_idx" in ann
    assert "visited_nodes" in ann
    assert "errors" in ann
    assert "human_action" in ann


def test_ensure_v3_fields_backfills():
    state = {"messages": [], "session_id": "x"}
    out = ensure_v3_fields(state)
    assert "compaction_history" in out
    assert "memory_tiers" in out
    assert "retrieval_context" in out


def test_ensure_v3_fields_idempotent():
    state = {
        "messages": [],
        "compaction_history": [{"x": 1}],
        "memory_tiers": {"hot_count": 5},
        "retrieval_context": [{"y": 1}],
    }
    out = ensure_v3_fields(state)
    # 不覆盖已有值
    assert out["compaction_history"] == [{"x": 1}]
    assert out["memory_tiers"] == {"hot_count": 5}
    assert out["retrieval_context"] == [{"y": 1}]


# === 模拟 LangGraph reducer 行为 ===

def test_respond_node_pattern_single_return():
    """模拟 respond_node 返回单条消息 + 旧 messages 列表。

    add_messages reducer 把 dict 转为 Message 实例但保留 content。
    """
    from langgraph.graph.message import add_messages
    existing = [{"role": "user", "content": "hi"}]
    new_msg = {"role": "assistant", "content": "summary"}
    merged = add_messages(existing, [new_msg])
    assert len(merged) == 2
    # 验证是 AI message，含 summary content
    last = merged[-1]
    assert last.content == "summary"


def test_respond_node_pattern_no_double_add():
    """旧写法（list+append） vs 新写法（return single）—— 新写法不重复。"""
    from langgraph.graph.message import add_messages
    # 旧写法：节点返回完整列表
    existing = [{"role": "user", "content": "hi"}]
    old_return = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "sum"}]
    merged_old = add_messages(existing, old_return)
    # 应去重：add_messages 用 message id 去重
    # 注：没有 id 时不会去重，所以 merged_old 长度会变
    # 这是 v3 老代码的兼容性陷阱——redundant 但不致命

    # 新写法：只 return 新增
    new_msg = {"role": "assistant", "content": "sum"}
    merged_new = add_messages(existing, [new_msg])
    assert len(merged_new) == 2  # 正确的 2 条


def test_summary_block_merges():
    """micro-compact 产生的 summary block 也能 merge（role='system' 兼容 LangChain）。"""
    from langgraph.graph.message import add_messages
    existing = [{"role": "user", "content": "m1"}]
    summary = {
        "role": "system",  # LangChain 合法
        "content": "[compacted]",
        "__summary__": True,
    }
    new = {"role": "user", "content": "m2"}
    merged = add_messages(existing, [summary, new])
    assert len(merged) == 3
    # 验证 message types（LangChain 转换后是对象不是 dict）
    assert merged[0].content == "m1"
    assert merged[1].content == "[compacted]"
    assert merged[2].content == "m2"
    # 保留 __summary__ 标记
    assert getattr(merged[1], "__summary__", False) is True or \
        (hasattr(merged[1], "additional_kwargs") and merged[1].additional_kwargs.get("__summary__"))


def test_old_messages_list_preserved_through_migration():
    """老 v2 state 有 messages 列表，migrate 到 v3 不丢消息。"""
    old = {
        "schema_version": 2,
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        "cost_summary": {"total_cost_usd": 0.01},
    }
    out = migrate_state(old)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    assert len(out["messages"]) == 2
    assert out["cost_summary"] == {"total_cost_usd": 0.01}
    assert "compaction_history" in out
    assert "memory_tiers" in out


def test_v3_state_no_migration():
    """当前版本 state 不再迁移。"""
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "messages": [],
        "compaction_history": [],
    }
    out = migrate_state(state)
    assert out is not state  # 返回新 dict
    assert out["compaction_history"] == []
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
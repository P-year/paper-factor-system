"""
test_migrations.py - state schema 版本化测试

覆盖：
1. v1 → v2 迁移（加 cost_summary / cost_usages / schema_version）
2. migrate_state 链式迁移
3. needs_migration 判定
4. load_session_state 自动跑迁移（向后兼容）
5. 当前 version 已是 latest 时不迁移
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.migrations import (
    migrate_state,
    get_schema_version,
    needs_migration,
    CURRENT_SCHEMA_VERSION,
    MIGRATIONS,
)


# === 基本 ===

def test_get_schema_version_default_v1():
    assert get_schema_version({}) == 1
    assert get_schema_version({"random": "stuff"}) == 1


def test_get_schema_version_explicit():
    assert get_schema_version({"schema_version": 2}) == 2
    assert get_schema_version({"schema_version": 3}) == 3
    assert get_schema_version({"schema_version": 5}) == 5


def test_needs_migration():
    assert needs_migration({}) is True  # v1 → CURRENT
    assert needs_migration({"schema_version": 1}) is True
    assert needs_migration({"schema_version": CURRENT_SCHEMA_VERSION}) is False
    # 未来版本：不迁移
    assert needs_migration({"schema_version": 99}) is False


def test_current_version_is_latest():
    assert CURRENT_SCHEMA_VERSION >= 3


# === v1 → v2 ===

def test_v1_to_v2_adds_cost_fields():
    state_v1 = {
        "goal": "test",
        "collected_paper_ids": [],
        "extracted_factors": [],
        # 无 cost 字段
    }
    out = migrate_state(state_v1)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    assert "cost_summary" in out
    assert out["cost_summary"] == {}
    assert "cost_usages" in out
    assert out["cost_usages"] == []
    # v3 字段也补上
    assert "compaction_history" in out
    assert "memory_tiers" in out
    assert "retrieval_context" in out


def test_v1_to_v2_preserves_existing_fields():
    """迁移不应丢失其他字段。"""
    state_v1 = {
        "goal": "test",
        "extracted_factors": [{"factor_name": "x"}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = migrate_state(state_v1)
    assert out["goal"] == "test"
    assert out["extracted_factors"] == [{"factor_name": "x"}]
    assert out["messages"] == [{"role": "user", "content": "hi"}]


def test_v1_to_v2_keeps_existing_cost_fields():
    """如果 v1 state 已经有 cost 字段（罕见但可能），迁移不覆盖。"""
    state_v1 = {
        "cost_summary": {"total_cost_usd": 1.0},
        "cost_usages": [{"id": "x"}],
    }
    out = migrate_state(state_v1)
    assert out["cost_summary"] == {"total_cost_usd": 1.0}
    assert out["cost_usages"] == [{"id": "x"}]


# === CURRENT 不变时 ===

def test_migrate_current_returns_same():
    state_v2 = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "goal": "test",
    }
    out = migrate_state(state_v2)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    assert out["goal"] == "test"


def test_migrate_future_version_returns_unchanged():
    """未来 schema version 不应崩——保留原样。"""
    state_future = {"schema_version": 99, "goal": "future"}
    out = migrate_state(state_future)
    assert out == state_future


# === 链式迁移 ===

def test_migration_chain():
    """验证 MIGRATIONS dict 有 CURRENT-1 条目。"""
    # 至少要有一条迁移（v1 → v2）
    assert 1 in MIGRATIONS
    # 当前是 v2
    assert CURRENT_SCHEMA_VERSION >= 2


def test_migrate_v1_chains_all_steps():
    """v1 state 完整迁移链执行一遍。"""
    state_v1 = {"goal": "x"}
    out = migrate_state(state_v1)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION


# === 与 checkpoints 集成 ===

def test_load_session_state_auto_migrates(tmp_path):
    """旧 session.json 没 schema_version，load 时自动跑迁移。"""
    from harness.checkpoints import save_session_state, load_session_state
    from harness.paths import SESSIONS_DIR

    # 写一个老格式 session
    import json
    test_session_id = "test_migration_session"
    session_file = SESSIONS_DIR / f"{test_session_id}.json"
    session_file.parent.mkdir(parents=True, exist_ok=True)

    # 写 v1 格式（没 schema_version）
    old_data = {
        "session_id": test_session_id,
        "saved_at": "2026-01-01T00:00:00",
        "state": {
            "goal": "old session",
            "extracted_factors": [{"factor_name": "x"}],
            # 没 schema_version / cost 字段
        },
    }
    session_file.write_text(json.dumps(old_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # load
    loaded = load_session_state(test_session_id)
    assert loaded is not None

    state = loaded["state"]
    # 自动迁移：应有 cost_summary / cost_usages / schema_version=3
    assert "cost_summary" in state
    assert "cost_usages" in state
    assert state["schema_version"] >= 3
    # v3 字段
    assert "compaction_history" in state
    assert "memory_tiers" in state
    assert "retrieval_context" in state
    # 老字段保留
    assert state["goal"] == "old session"
    assert state["extracted_factors"] == [{"factor_name": "x"}]


def test_load_session_state_current_no_migration(tmp_path):
    """最新版本 state 不应再迁移。"""
    from harness.checkpoints import save_session_state, load_session_state
    from harness.paths import SESSIONS_DIR

    test_session_id = "test_current_session"
    # 写最新 version
    state_v_current = {
        "goal": "current",
        "schema_version": CURRENT_SCHEMA_VERSION,
        "cost_summary": {"total_cost_usd": 0.5},
    }
    save_session_state(test_session_id, state_v_current)

    loaded = load_session_state(test_session_id)
    assert loaded["state"]["schema_version"] == CURRENT_SCHEMA_VERSION
    assert loaded["state"]["cost_summary"] == {"total_cost_usd": 0.5}


# === 损坏的 state ===

def test_migrate_garbage_state_doesnt_crash():
    """不期望的 state 形状不应让 migrate 崩。"""
    bad_state = {"random": "field", "no_schema_version": True}
    # 应该返回 migrated state 或原 state（取决于实现）
    try:
        out = migrate_state(bad_state)
        # 至少有 schema_version 字段
        assert "schema_version" in out
    except Exception as e:
        # 或者未来版本直接返回
        pytest.fail(f"migrate_state crashed on garbage: {e}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
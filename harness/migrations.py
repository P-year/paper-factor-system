"""
harness/migrations.py - State schema 版本化 + 自动迁移

目的：当 AgentState 字段变更时，老的 session JSON 能自动升级。

设计：
    CURRENT_SCHEMA_VERSION = N
    MIGRATIONS: Dict[int, Callable[[dict], dict]]  从 v_k 到 v_(k+1) 的迁移

加载逻辑（harness/checkpoints.py:load_session_state）：
    raw → 检测 schema_version（缺省 1）
    → 从 raw.version 升到 CURRENT_SCHEMA_VERSION
    → 返回新 state（带 schema_version 字段）

用法：
    from harness.migrations import migrate_state, CURRENT_SCHEMA_VERSION
    new_state = migrate_state(old_state)

新增 schema 变更：
    1. CURRENT_SCHEMA_VERSION += 1
    2. 在 MIGRATIONS[v_k] 加函数：state_k → state_(k+1)
"""
from typing import Any, Callable, Dict


CURRENT_SCHEMA_VERSION = 3


def _v1_to_v2(state: Dict[str, Any]) -> Dict[str, Any]:
    """v1 → v2：加 cost_summary / cost_usages 字段（v3-1）。

    老 state 没 cost 字段是 OK 的（默认空 dict）。
    只需标 schema_version=2。
    """
    out = dict(state)
    out.setdefault("cost_summary", {})
    out.setdefault("cost_usages", [])
    out["schema_version"] = 2
    return out


def _v2_to_v3(state: Dict[str, Any]) -> Dict[str, Any]:
    """v2 → v3：加上下文/记忆字段（v4）。

    compaction_history: micro-compact 触发记录
    memory_tiers: hot/warm/cold 三层元数据
    retrieval_context: RAG 检索上下文（v4 暂留 None）
    """
    out = dict(state)
    out.setdefault("compaction_history", [])
    out.setdefault("memory_tiers", {})
    out.setdefault("retrieval_context", None)
    out["schema_version"] = 3
    return out


MIGRATIONS: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    1: _v1_to_v2,
    2: _v2_to_v3,
}


def migrate_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """把任意老 version 的 state 升到 CURRENT_SCHEMA_VERSION。

    Args:
        state: 老 state（可能没有 schema_version 字段，视为 v1）

    Returns:
        升到 CURRENT_SCHEMA_VERSION 的新 state
    """
    current = state.get("schema_version", 1)
    if current > CURRENT_SCHEMA_VERSION:
        # 未来版本：警告但不崩（保留原样）
        return state

    out = dict(state)
    while current < CURRENT_SCHEMA_VERSION:
        migrate_fn = MIGRATIONS.get(current)
        if migrate_fn is None:
            raise RuntimeError(f"no migration from v{current} defined")
        out = migrate_fn(out)
        current = out.get("schema_version", current + 1)

    return out


def get_schema_version(state: Dict[str, Any]) -> int:
    """读 state 的 schema_version（缺省 1）。"""
    return state.get("schema_version", 1)


def needs_migration(state: Dict[str, Any]) -> bool:
    """state 是否需要迁移到 CURRENT_SCHEMA_VERSION。"""
    return get_schema_version(state) < CURRENT_SCHEMA_VERSION
"""
harness/checkpoints.py - JSON 持久化层

策略：
- MemorySaver 用于进程内 checkpoint（graph 自带）
- 每次 invoke / interrupt 落盘到 memory/sessions/{session_id}.json
- 启动时若指定 --resume-session=<id>，从 memory/sessions/{id}.json 恢复
- 决策审计追加到 memory/audit/decisions.jsonl
- v3-9：加载时自动跑 schema migrations

Phase 0：从 agent/checkpoints.py 搬运而来。Phase 1 加 pipeline 字段写入。
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from harness.paths import SESSIONS_DIR, RUNS_DIR, AUDIT_DIR


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def save_session_state(session_id: str, state: Dict[str, Any]) -> Path:
    """保存当前会话状态到 memory/sessions/{session_id}.json"""
    payload = {
        "session_id": session_id,
        "saved_at": datetime.now().isoformat(),
        "schema_version": state.get("schema_version", 1),
        "state": state,
    }
    path = _session_path(session_id)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def load_session_state(session_id: str) -> Optional[Dict[str, Any]]:
    """从 memory/sessions/{session_id}.json 恢复状态。

    v3-9：自动跑 schema migrations。
    老 state 没 schema_version 字段视为 v1。
    """
    path = _session_path(session_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # 自动迁移
    if data and "state" in data:
        try:
            from harness.migrations import migrate_state, needs_migration
            if needs_migration(data["state"]):
                data["state"] = migrate_state(data["state"])
                data["schema_version"] = data["state"].get("schema_version", 1)
        except Exception as e:
            # 迁移失败：保留原 state，记录 warning
            import warnings
            warnings.warn(f"schema migration failed for session {session_id}: {e}")

    return data


def list_sessions() -> list:
    """列出所有已保存的 session"""
    out = []
    for p in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "session_id": meta.get("session_id"),
                "saved_at": meta.get("saved_at"),
                "filename": p.name,
            })
        except Exception:
            continue
    return out


def append_audit(event_type: str, payload: Dict[str, Any]) -> None:
    """追加一条决策审计到 memory/audit/decisions.jsonl"""
    record = {
        "ts": datetime.now().isoformat(),
        "event": event_type,
        **payload,
    }
    audit_file = AUDIT_DIR / "decisions.jsonl"
    with audit_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def save_run_log(run_id: str, state: Dict[str, Any], report_path: Optional[str] = None) -> Path:
    """保存调度器运行日志到 memory/runs/{run_id}/state.json"""
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "saved_at": datetime.now().isoformat(),
        "state": state,
        "report_path": report_path,
    }
    path = run_dir / "state.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path
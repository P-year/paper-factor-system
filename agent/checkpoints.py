"""DEPRECATED: use harness.checkpoints directly.

Phase 0 兼容垫片。
"""
from harness.checkpoints import (
    save_session_state,
    load_session_state,
    list_sessions,
    append_audit,
    save_run_log,
)

__all__ = [
    "save_session_state",
    "load_session_state",
    "list_sessions",
    "append_audit",
    "save_run_log",
]
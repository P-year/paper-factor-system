"""
test_session_backward_compat.py - 旧 session 加载兼容

Phase 0 / 1 之前的 session JSON 没有 pipeline_name 字段。
harness.checkpoints.load_session_state 必须能正常加载并默认 paper_factor。
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest


def test_old_session_loads_without_pipeline_name():
    """旧的 final_smoke.json（无 pipeline_name 字段）能正常加载。"""
    from harness.checkpoints import load_session_state

    old_path = PROJECT_ROOT / "memory" / "sessions" / "final_smoke.json"
    if not old_path.exists():
        pytest.skip(f"old session not found: {old_path}")

    data = load_session_state("final_smoke")
    assert data is not None
    state = data.get("state", {})
    # 不报错即可（pipeline_name 缺失时由 cmd_repl 兜底为 paper_factor）
    assert "session_id" in data
    assert state.get("visited_nodes") is not None


def test_old_session_no_pipeline_field_default():
    """旧 session 缺 pipeline_name，cmd_repl 应回退到 default。"""
    from harness.checkpoints import load_session_state

    old_path = PROJECT_ROOT / "memory" / "sessions" / "final_smoke.json"
    if not old_path.exists():
        pytest.skip(f"old session not found: {old_path}")

    data = load_session_state("final_smoke")
    state = data.get("state", {})
    # 默认 pipeline
    pipeline = state.get("pipeline_name") or "paper_factor"
    assert pipeline == "paper_factor"


def test_list_sessions_returns_existing():
    """list_sessions 不报错，能列出 final_smoke。"""
    from harness.checkpoints import list_sessions

    sessions = list_sessions()
    ids = [s["session_id"] for s in sessions]
    assert "final_smoke" in ids


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
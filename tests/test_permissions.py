"""
test_permissions.py - tool 权限分级测试

覆盖：
1. read_only tool 无审批即可调
2. requires_approval tool 没批准时抛
3. requires_approval tool 有正确 human_action 时通过
4. 未知 tool 默认允许（宽松）
5. pipeline 限制
6. enforcement 开关
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.permissions import (
    PermissionDenied,
    PermissionRegistry,
    ToolPermissions,
    DEFAULT_PERMISSIONS,
    get_default_permissions,
    reset_permissions,
    set_enforce,
)


# === read_only ===

def test_read_only_tool_no_approval_needed():
    perms = PermissionRegistry()
    perms.assert_can_call("list_factors")  # 不抛


def test_requires_approval_without_human_action_raises():
    perms = PermissionRegistry()
    with pytest.raises(PermissionDenied) as exc:
        perms.assert_can_call("update_factor_status")
    assert exc.value.tool_name == "update_factor_status"
    assert "human approval" in exc.value.reason


def test_requires_approval_with_wrong_human_action_raises():
    perms = PermissionRegistry()
    with pytest.raises(PermissionDenied):
        perms.assert_can_call(
            "update_factor_status",
            human_action={"tool_name": "other_tool", "decision": "approved"},
        )


def test_requires_approval_with_correct_human_action_passes():
    perms = PermissionRegistry()
    perms.assert_can_call(
        "update_factor_status",
        human_action={"tool_name": "update_factor_status", "decision": "approved"},
    )


# === unknown tool ===

def test_unknown_tool_default_allowed():
    perms = PermissionRegistry()
    perms.assert_can_call("totally_unknown_tool")  # 不抛（宽松）


# === pipeline 限制 ===

def test_pipeline_restriction_blocks():
    perm = ToolPermissions(
        tool_name="special_tool",
        allowed_pipelines={"signal_research"},
    )
    perms = PermissionRegistry({"special_tool": perm})
    perms.assert_can_call("special_tool", pipeline="signal_research")  # ok
    with pytest.raises(PermissionDenied):
        perms.assert_can_call("special_tool", pipeline="paper_factor")


def test_pipeline_none_passes():
    perm = ToolPermissions(tool_name="x", allowed_pipelines={"signal_research"})
    perms = PermissionRegistry({"x": perm})
    perms.assert_can_call("x")  # 没传 pipeline → 通过


# === enforce 开关 ===

def test_enforce_false_bypasses():
    perms = PermissionRegistry()
    perms.set_enforce(False)
    # 即使 requires_approval 也通过
    perms.assert_can_call("update_factor_status")


def test_enforce_toggle():
    perms = PermissionRegistry()
    perms.set_enforce(False)
    perms.assert_can_call("update_factor_status")
    perms.set_enforce(True)
    with pytest.raises(PermissionDenied):
        perms.assert_can_call("update_factor_status")


# === 默认注册表 ===

def test_default_registry_has_known_tools():
    perms = get_default_permissions()
    assert "search_arxiv" in perms.all_tool_names()
    assert "update_factor_status" in perms.all_tool_names()
    assert "run_backtest" in perms.all_tool_names()


def test_default_update_factor_requires_approval():
    perms = get_default_permissions()
    p = perms.get("update_factor_status")
    assert p.requires_approval is True
    assert p.risk_level == "high"


def test_default_search_arxiv_medium_risk():
    perms = get_default_permissions()
    p = perms.get("search_arxiv")
    assert p.risk_level == "medium"
    assert p.read_only is False


def test_default_list_factors_read_only():
    perms = get_default_permissions()
    p = perms.get("list_factors")
    assert p.read_only is True
    assert p.risk_level == "low"


# === 注册自定义 ===

def test_register_custom_tool():
    perms = PermissionRegistry()
    new_perm = ToolPermissions(
        tool_name="custom_tool",
        read_only=False,
        requires_approval=True,
        risk_level="high",
    )
    perms.register(new_perm)

    assert perms.get("custom_tool") is new_perm
    with pytest.raises(PermissionDenied):
        perms.assert_can_call("custom_tool")


# === human_action 校验 ===

def test_assert_human_action_valid():
    perms = PermissionRegistry()
    assert perms.assert_human_action_valid(None) is True
    assert perms.assert_human_action_valid({"decision": "approve"}) is True
    assert perms.assert_human_action_valid({"tool_name": "x"}) is True
    assert perms.assert_human_action_valid({"random": "stuff"}) is False
    assert perms.assert_human_action_valid("not a dict") is False


# === 全局 set_enforce ===

def test_global_set_enforce(monkeypatch):
    reset_permissions()
    perms = get_default_permissions()
    set_enforce(False)
    perms.assert_can_call("update_factor_status")
    set_enforce(True)
    with pytest.raises(PermissionDenied):
        perms.assert_can_call("update_factor_status")
    reset_permissions()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
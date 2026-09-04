"""
harness/permissions.py - Tool 权限分级

目的：区分 read_only / mutating / requires_approval 三类 tool，
让 harness 能在调 tool 前做权限检查。

工具元数据（默认）：
    ToolName → {read_only, requires_approval, risk_level, allowed_pipelines}

risk_level: "low" | "medium" | "high"
    low       读操作，无副作用（search_arxiv, read_paper, list_factors, ...）
    medium    写操作但可逆（persist_factors, save_papers, ...）
    high      写操作不可逆或成本高（update_factor_status 改 DB / LLM call / external API）

用法：
    from harness.permissions import get_default_permissions, PermissionDenied

    perms = get_default_permissions()
    perms.assert_can_call("update_factor_status", pipeline="paper_factor")  # ok
    perms.assert_can_call("some_forbidden_tool")  # raise

    # 运行时检查 human_action 是否批准了 mutating tool
    perms.assert_human_approved(tool_name, state)
"""
import os
from dataclasses import dataclass
from typing import Dict, Optional, Set


class PermissionDenied(RuntimeError):
    """tool 调用未通过权限检查。"""
    def __init__(self, tool_name: str, reason: str):
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"permission denied '{tool_name}': {reason}")


@dataclass
class ToolPermissions:
    """单 tool 的权限元数据。"""
    tool_name: str
    read_only: bool = True
    requires_approval: bool = False      # 调之前需要 human_action 批准
    risk_level: str = "low"              # low / medium / high
    allowed_pipelines: Optional[Set[str]] = None  # None = 所有 pipeline 允许
    description: str = ""


# === 默认权限表（覆盖 paper_factor 当前所有 tool） ===

DEFAULT_PERMISSIONS: Dict[str, ToolPermissions] = {
    # 读类工具
    "search_arxiv": ToolPermissions(
        tool_name="search_arxiv", read_only=False, requires_approval=False, risk_level="medium",
        description="外部 API（外部数据，可能被 rate-limit）",
    ),
    "read_paper": ToolPermissions(
        tool_name="read_paper", read_only=True, risk_level="low",
        description="读本地文件",
    ),
    "extract_factor_tool": ToolPermissions(
        tool_name="extract_factor_tool", read_only=True, risk_level="medium",
        description="LLM 调用（中等成本）",
    ),
    "check_data_feasibility": ToolPermissions(
        tool_name="check_data_feasibility", read_only=True, risk_level="low",
        description="启发式纯函数",
    ),
    "llm_check_data_feasibility": ToolPermissions(
        tool_name="llm_check_data_feasibility", read_only=True, risk_level="medium",
        description="LLM 调用（中等成本）",
    ),
    "run_backtest": ToolPermissions(
        tool_name="run_backtest", read_only=True, risk_level="medium",
        description="外部 AKShare API + 读计算",
    ),
    "list_factors": ToolPermissions(
        tool_name="list_factors", read_only=True, risk_level="low",
        description="读 factor DB",
    ),
    "generate_report": ToolPermissions(
        tool_name="generate_report", read_only=False, risk_level="low",
        description="读 DB + 写文件（output/report.md）",
    ),
    # 写类工具（高风险，需要审批）
    "update_factor_status": ToolPermissions(
        tool_name="update_factor_status", read_only=False, requires_approval=True, risk_level="high",
        description="改 factor DB 状态（不可逆）",
    ),
}


class PermissionRegistry:
    """全局 tool 权限注册表。"""

    def __init__(self, permissions: Optional[Dict[str, ToolPermissions]] = None, *, enforce: bool = True):
        self._perms: Dict[str, ToolPermissions] = dict(permissions or DEFAULT_PERMISSIONS)
        self._enforce = enforce

    def set_enforce(self, enforce: bool) -> None:
        """运行时切换 enforcement（测试 / 调试用）。"""
        self._enforce = enforce

    def register(self, perm: ToolPermissions) -> None:
        self._perms[perm.tool_name] = perm

    def get(self, tool_name: str) -> Optional[ToolPermissions]:
        return self._perms.get(tool_name)

    def all_tool_names(self) -> list:
        return list(self._perms.keys())

    def assert_can_call(
        self,
        tool_name: str,
        *,
        pipeline: Optional[str] = None,
        human_action: Optional[dict] = None,
    ) -> None:
        """检查 tool 是否允许调用。违反时抛 PermissionDenied。"""
        if not self._enforce:
            return

        perm = self._perms.get(tool_name)
        if perm is None:
            # 未知 tool：默认允许但记日志（生产应该 raise）
            return

        # pipeline 限制
        if perm.allowed_pipelines is not None and pipeline is not None:
            if pipeline not in perm.allowed_pipelines:
                raise PermissionDenied(
                    tool_name,
                    f"pipeline '{pipeline}' not in allowed {perm.allowed_pipelines}",
                )

        # requires_approval 检查
        if perm.requires_approval:
            if human_action is None or human_action.get("tool_name") != tool_name:
                raise PermissionDenied(
                    tool_name,
                    f"tool requires human approval (set human_action={{tool_name, decision, ...}})",
                )

    def assert_human_action_valid(self, human_action: Optional[dict]) -> bool:
        """校验 human_action 格式。"""
        if human_action is None:
            return True
        if not isinstance(human_action, dict):
            return False
        # 至少要有 tool_name + decision（或 factor_name）
        return "decision" in human_action or "tool_name" in human_action


# === 全局 registry ===

_default_registry: Optional[PermissionRegistry] = None


def get_default_permissions() -> PermissionRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = PermissionRegistry()
    return _default_registry


def reset_permissions() -> None:
    """测试用。"""
    global _default_registry
    _default_registry = None


def set_enforce(enforce: bool) -> None:
    """全局 enforcement 开关（agent.tools.TOOL_LIST 调 tool 时会查）。"""
    get_default_permissions().set_enforce(enforce)
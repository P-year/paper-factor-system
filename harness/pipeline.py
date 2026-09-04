"""
harness/pipeline.py - Pipeline 配置对象

一个 Pipeline = yaml 声明（name / prompts / hard_rules / max_iter / interrupt_before / terminal_nodes / tool_modules）
              + python 解析（state_schema / nodes dict / auto_approve_fn / extra）

yaml 文件位于 pipelines/<name>/config.yaml；python 解析通过 @register_pipeline 装饰器或
显式 PipelineConfig(state_schema=..., nodes=..., auto_approve_fn=..., yaml_path=...) 传入。

Phase 0：只放 dataclass + yaml loader，不接 graph_builder。
Phase 1：被 graph_builder / plan_node / registry 使用。
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import yaml

from harness.state_base import make_base_typed_dict


@dataclass
class PipelineConfig:
    """一个 pipeline 的完整配置。

    字段：
    - name: pipeline 名（与 yaml 的 name 字段一致）
    - yaml_path: 配置文件路径（debug 用）
    - version: yaml 的 version 字段

    - state_schema: TypedDict 子类（如 PaperFactorState）
    - nodes: Dict[node_name, callable(state)->dict]
    - auto_approve_fn: scheduler 模式自动审批回调，签名 (state, config) -> Optional[dict]
    - extra: 任意附加字段（pipeline 自定义用）

    - max_iter: plan_node 最大迭代数（>= 后强制 respond）
    - interrupt_before: 中断节点列表（LangGraph interrupt_before）
    - terminal_nodes: 路由到 END 的节点名集合
    - tool_modules: tool 所在的 python 模块路径列表（字符串），用于聚合 TOOL_LIST

    - prompts: prompt 模板 dict，key=prompt 名，value=字符串
    - hard_rules: 硬规则列表，每条 {id, when, force, reason}
    """

    name: str
    yaml_path: Optional[Path] = None
    version: int = 1

    state_schema: Type = None  # TypedDict 子类（如 PaperFactorState）
    nodes: Dict[str, Callable] = field(default_factory=dict)
    auto_approve_fn: Optional[Callable] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    max_iter: int = 8
    interrupt_before: List[str] = field(default_factory=list)
    terminal_nodes: List[str] = field(default_factory=lambda: ["respond"])
    tool_modules: List[str] = field(default_factory=list)

    prompts: Dict[str, str] = field(default_factory=dict)
    hard_rules: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.state_schema is None:
            self.state_schema = make_base_typed_dict()

    # === 校验 ===

    def __post_init__(self):
        # 节点名集合 vs terminal_nodes
        if self.nodes:
            unknown = [n for n in self.terminal_nodes if n not in self.nodes]
            if unknown:
                raise ValueError(
                    f"Pipeline '{self.name}': terminal_nodes {unknown} not in nodes {list(self.nodes)}"
                )
        # hard_rules 每条至少要有 id / when / force
        for r in self.hard_rules:
            for key in ("id", "when", "force"):
                if key not in r:
                    raise ValueError(
                        f"Pipeline '{self.name}': hard_rule missing key '{key}': {r}"
                    )


def load_yaml_config(yaml_path: Path) -> Dict[str, Any]:
    """读取 yaml 文件，返回 dict。Phase 0 阶段纯数据层。"""
    if not yaml_path.exists():
        raise FileNotFoundError(f"Pipeline yaml not found: {yaml_path}")
    with yaml_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Pipeline yaml must be a dict, got {type(data).__name__}: {yaml_path}")
    return data


def make_pipeline_from_yaml(
    yaml_path: Path,
    *,
    state_schema: Type,
    nodes: Dict[str, Callable],
    auto_approve_fn: Optional[Callable] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> PipelineConfig:
    """读 yaml + 接收 python 部分 → 构造 PipelineConfig。

    用法（pipelines/<name>/__init__.py）：
        from pipelines.paper_factor.state import PaperFactorState
        from pipelines.paper_factor.nodes import NODES
        from pipelines.paper_factor.auto_approve import auto_approve

        def _register():
            yaml_path = Path(__file__).parent / "config.yaml"
            return make_pipeline_from_yaml(
                yaml_path,
                state_schema=PaperFactorState,
                nodes=NODES,
                auto_approve_fn=auto_approve,
            )
    """
    raw = load_yaml_config(yaml_path)
    return PipelineConfig(
        name=raw["name"],
        yaml_path=yaml_path,
        version=raw.get("version", 1),
        state_schema=state_schema,
        nodes=nodes,
        auto_approve_fn=auto_approve_fn,
        extra=extra or {},
        max_iter=raw.get("max_iter", 8),
        interrupt_before=list(raw.get("interrupt_before", [])),
        terminal_nodes=list(raw.get("terminal_nodes", ["respond"])),
        tool_modules=list(raw.get("tool_modules", [])),
        prompts=dict(raw.get("prompts", {})),
        hard_rules=list(raw.get("hard_rules", [])),
    )
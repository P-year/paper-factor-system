"""
harness/registry.py - pipeline 注册中心

用法（pipelines/<name>/__init__.py）：
    from harness.registry import register_pipeline, load_all, get

    @register_pipeline
    def _register_paper_factor():
        from pipelines.paper_factor.state import PaperFactorState
        from pipelines.paper_factor.nodes import NODES
        from pipelines.paper_factor.auto_approve import auto_approve
        from harness.pipeline import make_pipeline_from_yaml
        from pathlib import Path

        return make_pipeline_from_yaml(
            Path(__file__).parent / "config.yaml",
            state_schema=PaperFactorState,
            nodes=NODES,
            auto_approve_fn=auto_approve,
        )

Phase 0：只有骨架，调用 load_all() 时打印提示但不加载任何 pipeline。
Phase 1：pipelines/paper_factor/__init__.py 加 @register_pipeline 装饰后，load_all() 自动发现。
"""
from typing import Callable, Dict, List, Optional

from harness.pipeline import PipelineConfig

_REGISTRY: Dict[str, PipelineConfig] = {}
_REGISTRARS: Dict[str, Callable[[], PipelineConfig]] = {}
_LOADED = False


def register_pipeline(fn: Callable[[], PipelineConfig]) -> Callable[[], PipelineConfig]:
    """装饰器：把"返回一个 PipelineConfig 的工厂函数"注册到全局表。

    注意：装饰器只登记工厂函数，不立即执行（避免循环导入）。
    load_all() 时才真正调用工厂函数。
    """
    # 通过 __name__ 约定 key：如 "_register_paper_factor" → "paper_factor"
    name = fn.__name__.lstrip("_register_").lstrip("register_")
    if name == fn.__name__:
        # 没匹配约定，用模块路径猜
        name = fn.__module__.split(".")[-1]
    _REGISTRARS[name] = fn
    return fn


def load_all(verbose: bool = False) -> List[str]:
    """遍历 _REGISTRARS，调用每个工厂函数，把 PipelineConfig 存入 _REGISTRY。

    返回已加载的 pipeline 名列表。
    Phase 0：空列表（无注册）。
    """
    global _LOADED
    if _LOADED:
        return list(_REGISTRY.keys())

    # Phase 1 才需要 import pipelines.<name> 的副作用（触发 @register_pipeline）
    # Phase 0 保持空
    try:
        from pipelines import paper_factor  # noqa: F401
    except ImportError as e:
        if verbose:
            print(f"[registry] pipelines/ not found yet (Phase 0): {e}")

    for name, factory in _REGISTRARS.items():
        try:
            pipeline = factory()
            _REGISTRY[name] = pipeline
            if verbose:
                print(f"[registry] loaded pipeline: {name} (nodes={list(pipeline.nodes)})")
        except Exception as e:
            print(f"[registry] failed to load pipeline '{name}': {e}")

    _LOADED = True
    return list(_REGISTRY.keys())


def get(name: str) -> Optional[PipelineConfig]:
    """按名取 PipelineConfig。未加载则先 load_all()。"""
    if not _LOADED:
        load_all()
    return _REGISTRY.get(name)


def all_pipelines() -> List[PipelineConfig]:
    if not _LOADED:
        load_all()
    return list(_REGISTRY.values())


def reset() -> None:
    """测试用：清空 registry。"""
    global _LOADED
    _REGISTRY.clear()
    _REGISTRARS.clear()
    _LOADED = False
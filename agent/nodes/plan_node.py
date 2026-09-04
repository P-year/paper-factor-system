"""DEPRECATED: use pipelines.paper_factor.nodes.plan directly.

Phase 1 兼容垫片。

plan_node 现在由 harness.plan_node.make_plan_node(pipeline) 动态构造，
绑定 pipeline 的 prompts / hard_rules / max_iter。

旧的 `from agent.nodes.plan_node import plan_node` 仍可用，但实际拿到的是
运行时根据 paper_factor pipeline 配置生成的 plan_node。
"""
from pipelines.paper_factor.nodes.plan import make_paper_factor_plan_node
from harness.registry import get as _registry_get


def _get_plan_node():
    """惰性构造 plan_node（避免循环导入）。"""
    pipeline = _registry_get("paper_factor")
    if pipeline is None:
        raise RuntimeError("paper_factor pipeline not loaded; call harness.registry.load_all() first")
    return make_paper_factor_plan_node(pipeline)


# 暴露工厂函数，旧代码可以这样用：
#   from agent.nodes.plan_node import make_plan_node
#   plan_node = make_plan_node(pipeline)
def make_plan_node(pipeline=None):
    if pipeline is None:
        pipeline = _registry_get("paper_factor")
    return make_paper_factor_plan_node(pipeline)


# 同时暴露一个 bound 版本（在第一次 import 时构造）
# 旧代码 `from agent.nodes.plan_node import plan_node` 直接拿到可调用对象
try:
    plan_node = _get_plan_node()
except RuntimeError:
    # registry 还未加载（旧代码 import 早于 load_all），用懒加载代理
    class _LazyPlanNode:
        def __call__(self, state):
            return _get_plan_node()(state)
        def __getattr__(self, name):
            return getattr(_get_plan_node(), name)
    plan_node = _LazyPlanNode()

__all__ = ["plan_node", "make_plan_node"]
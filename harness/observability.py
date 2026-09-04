"""
observability.py - Langfuse 可观测性集成

提供：
1. @observe_node() 装饰器：自动追踪每个 LangGraph node
2. get_traced_callbacks()：返回 LangChain 可用的 Langfuse handler
3. graceful fallback：未配置 Langfuse 时不报错

环境变量：
- LANGFUSE_PUBLIC_KEY
- LANGFUSE_SECRET_KEY
- LANGFUSE_HOST (默认 https://cloud.langfuse.com)

Phase 0：从 agent/observability.py 搬运而来，行为不变。
"""
import os
import functools
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# 延迟导入：避免硬依赖
_langfuse_decorator = None
_langfuse_handler_cls = None
_langfuse_initialized = False


def _init_langfuse():
    """初始化 Langfuse。如果环境变量未配置，返回 False。"""
    global _langfuse_decorator, _langfuse_handler_cls, _langfuse_initialized

    if _langfuse_initialized:
        return _langfuse_decorator is not None

    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    if not pk or not sk:
        logger.info("Langfuse not configured (missing keys), tracing disabled")
        _langfuse_initialized = True
        return False

    try:
        from langfuse.decorators import observe, langfuse_context
        from langfuse.callback import CallbackHandler

        _langfuse_decorator = observe
        _langfuse_handler_cls = CallbackHandler
        _langfuse_initialized = True
        logger.info("Langfuse tracing enabled")
        return True
    except ImportError:
        logger.warning("langfuse package not installed, tracing disabled")
        _langfuse_initialized = True
        return False


def observe_node(name: Optional[str] = None):
    """
    装饰器：追踪 LangGraph node 调用。

    Usage:
        @observe_node("plan_node")
        def plan_node(state: AgentState) -> dict:
            ...
    """
    enabled = _init_langfuse()

    def decorator(func: Callable) -> Callable:
        node_name = name or func.__name__

        if enabled:
            @functools.wraps(func)
            def wrapper(state):
                # 调用 Langfuse 的 @observe 装饰器
                observed = _langfuse_decorator(name=node_name)(func)
                try:
                    result = observed(state)
                    # 记录关键 state 信息
                    _update_current_observation(state, result, node_name)
                    return result
                except Exception as e:
                    _record_error(e)
                    raise
            return wrapper
        else:
            # 无 Langfuse：直接透传
            @functools.wraps(func)
            def wrapper(state):
                return func(state)
            return wrapper

    return decorator


def _update_current_observation(state, result, node_name: str):
    """把 state 中的关键信息记录到当前 Langfuse trace。"""
    try:
        from langfuse.decorators import langfuse_context

        # 提取可序列化信息
        metadata = {
            "iteration": state.get("iteration_count", 0),
            "visited_nodes": state.get("visited_nodes", [])[-5:],
            "collected_paper_count": len(state.get("collected_paper_ids", [])),
            "extracted_factor_count": len(state.get("extracted_factors", [])),
            "backtest_result_count": len(state.get("backtest_results", [])),
        }

        # 不同 node 的特殊字段
        if node_name == "plan_node" and isinstance(result, dict):
            metadata["next_node"] = result.get("next_node")
            metadata["reason"] = result.get("reason", "")[:200]
        elif node_name == "collect_node":
            metadata["new_papers"] = len(result.get("collected_paper_ids", []))
        elif node_name == "analyze_node":
            metadata["new_factors"] = len(result.get("extracted_factors", []))

        langfuse_context.update_current_observation(metadata=metadata)
    except Exception as e:
        logger.debug(f"Failed to update langfuse observation: {e}")


def _record_error(error: Exception):
    """记录错误到 Langfuse。"""
    try:
        from langfuse.decorators import langfuse_context
        langfuse_context.update_current_observation(
            level="ERROR",
            status_message=str(error)[:500],
        )
    except Exception:
        pass


def get_traced_callbacks():
    """
    返回 LangChain 的 callbacks 列表，注入 Langfuse CallbackHandler。
    用于 .invoke(..., config={"callbacks": [...]})。

    Usage:
        llm.invoke(prompt, config={"callbacks": get_traced_callbacks()})
    """
    if not _init_langfuse():
        return []

    try:
        handler = _langfuse_handler_cls()
        return [handler]
    except Exception as e:
        logger.warning(f"Failed to create Langfuse handler: {e}")
        return []


def flush():
    """flush 上报数据到 Langfuse（程序退出前调用）。"""
    if not _init_langfuse():
        return
    try:
        from langfuse import Langfuse
        Langfuse().flush()
    except Exception:
        pass
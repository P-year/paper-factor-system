"""
api/stream.py - 流式 Agent 运行器（独立模块）

提供 stream_run() 异步生成器，可被多种后端复用。
"""
import asyncio
from typing import AsyncIterator, Dict, Any
from datetime import datetime

from agent.graph import build_graph


_graph_cache = None


async def get_graph():
    """懒加载 graph。"""
    global _graph_cache
    if _graph_cache is None:
        _graph_cache, _ = build_graph(interrupt_before_decide=False)
    return _graph_cache


async def stream_run(goal: str, thread_id: str = None) -> AsyncIterator[Dict[str, Any]]:
    """
    流式运行 Agent，逐步产出事件。

    Yields:
        dict: {"event": "node"|"done"|"error", "node": str, "data": dict}
    """
    graph = await get_graph()
    thread_id = thread_id or f"stream-{int(datetime.now().timestamp())}"
    config = {"configurable": {"thread_id": thread_id}}

    yield {"event": "start", "thread_id": thread_id, "goal": goal}

    try:
        async for event in graph.astream(
            {"goal": goal, "iteration_count": 0},
            config=config,
        ):
            for node_name, node_output in event.items():
                if node_name == "__end__":
                    continue
                yield {
                    "event": "node",
                    "node": node_name,
                    "output": node_output,
                }
                # 让上层有时间处理（控制节奏）
                await asyncio.sleep(0.05)
        yield {"event": "done", "thread_id": thread_id}
    except Exception as e:
        yield {"event": "error", "error": str(e)}


async def reset_graph_cache():
    """重置 graph 缓存（用于测试）。"""
    global _graph_cache
    _graph_cache = None
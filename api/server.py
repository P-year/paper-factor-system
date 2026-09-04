"""
api/server.py - Paper Factor System FastAPI 服务

提供 REST + SSE 接口，让 Agent 可被任何 HTTP 客户端调用。
"""
import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, AsyncIterator
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import build_graph

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="Paper Factor Agent API",
    description="基于 LangGraph 的论文因子挖掘 Agent",
    version="1.0.0",
)

# 全局 graph 实例（启动时构建）
_graph = None
_checkpointer = None


@app.on_event("startup")
async def startup():
    """启动时构建 graph。"""
    global _graph, _checkpointer
    _graph, _checkpointer = build_graph(interrupt_before_decide=False)
    print("[startup] Agent graph initialized")


# ============================================================
# 请求/响应模型
# ============================================================
class RunRequest(BaseModel):
    """运行 Agent 的请求体"""
    goal: str = Field(..., description="用户目标，如'分析动量因子论文'")
    thread_id: Optional[str] = Field(None, description="会话 ID（用于 checkpoint 续跑）")


class RunResponse(BaseModel):
    """同步运行结果"""
    thread_id: str
    goal: str
    status: str
    iteration_count: int
    visited_nodes: list
    collected_paper_count: int
    extracted_factor_count: int
    backtest_result_count: int
    final_message: Optional[str] = None
    elapsed_seconds: float


# ============================================================
# 接口
# ============================================================
@app.get("/")
async def root():
    """健康检查 + 服务信息"""
    return {
        "service": "Paper Factor Agent API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "POST /agent/run       同步运行",
            "POST /agent/stream    流式（SSE）",
            "GET  /sessions/:tid    查看会话状态",
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/agent/run", response_model=RunResponse)
async def run_agent(req: RunRequest):
    """
    同步运行 Agent 直到结束。返回最终状态。
    适合不需要实时观察的场景。
    """
    if _graph is None:
        raise HTTPException(503, "Graph not initialized")

    thread_id = req.thread_id or f"api-{int(datetime.now().timestamp())}"
    config = {"configurable": {"thread_id": thread_id}}

    start = datetime.now()
    try:
        final_state = await _graph.ainvoke(
            {"goal": req.goal, "iteration_count": 0},
            config=config,
        )
        elapsed = (datetime.now() - start).total_seconds()

        # 提取 respond 节点的最终消息
        msgs = final_state.get("messages", []) if isinstance(final_state, dict) else []
        final_msg = None
        if msgs:
            last = msgs[-1]
            if isinstance(last, dict):
                final_msg = last.get("content")
            else:
                final_msg = str(last)

        return RunResponse(
            thread_id=thread_id,
            goal=req.goal,
            status="completed",
            iteration_count=final_state.get("iteration_count", 0) if isinstance(final_state, dict) else 0,
            visited_nodes=final_state.get("visited_nodes", []) if isinstance(final_state, dict) else [],
            collected_paper_count=len(final_state.get("collected_paper_ids", [])) if isinstance(final_state, dict) else 0,
            extracted_factor_count=len(final_state.get("extracted_factors", [])) if isinstance(final_state, dict) else 0,
            backtest_result_count=len(final_state.get("backtest_results", [])) if isinstance(final_state, dict) else 0,
            final_message=final_msg,
            elapsed_seconds=round(elapsed, 2),
        )
    except Exception as e:
        raise HTTPException(500, f"Agent run failed: {e}")


@app.post("/agent/stream")
async def stream_agent(req: RunRequest):
    """
    流式运行 Agent（SSE）。每个 node 完成时推送一条事件。
    适合需要可视化执行流的场景。
    """
    if _graph is None:
        raise HTTPException(503, "Graph not initialized")

    async def event_generator() -> AsyncIterator[str]:
        thread_id = req.thread_id or f"api-{int(datetime.now().timestamp())}"
        config = {"configurable": {"thread_id": thread_id}}

        yield f"event: start\ndata: {{\"thread_id\": \"{thread_id}\", \"goal\": \"{req.goal}\"}}\n\n"

        try:
            async for event in _graph.astream(
                {"goal": req.goal, "iteration_count": 0},
                config=config,
            ):
                for node_name, node_output in event.items():
                    if node_name == "__end__":
                        continue
                    # 把每个 node 的输出简化后推送
                    summary = _summarize_node_output(node_name, node_output)
                    yield f"event: node\ndata: {{\"node\": \"{node_name}\", \"summary\": {summary}}}\n\n"
                    await asyncio.sleep(0.05)  # 让前端有动画时间

            yield f"event: done\ndata: {{\"thread_id\": \"{thread_id}\"}}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {{\"error\": \"{e}\"}}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/sessions/{thread_id}")
async def get_session(thread_id: str):
    """获取会话状态（checkpoint）"""
    if _graph is None:
        raise HTTPException(503, "Graph not initialized")

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = _graph.get_state(config)
    if snapshot is None or not snapshot.values:
        raise HTTPException(404, "Session not found")

    state = snapshot.values
    return JSONResponse({
        "thread_id": thread_id,
        "goal": state.get("goal"),
        "iteration_count": state.get("iteration_count", 0),
        "visited_nodes": state.get("visited_nodes", []),
        "collected_paper_count": len(state.get("collected_paper_ids", [])),
        "extracted_factor_count": len(state.get("extracted_factors", [])),
        "backtest_result_count": len(state.get("backtest_results", [])),
        "next_step": snapshot.next,
    })


# ============================================================
# 工具函数
# ============================================================
def _summarize_node_output(node_name: str, output) -> str:
    """把 node 输出压缩成可传输的 JSON 字符串。"""
    import json
    if not isinstance(output, dict):
        return json.dumps({"raw": str(output)[:200]}, ensure_ascii=False)

    summary = {}
    if "plan" in output:
        plan = output["plan"]
        if plan:
            last = plan[-1]
            summary["next_node"] = last.get("node")
            summary["reason"] = last.get("reason", "")[:200]
    if "iteration_count" in output:
        summary["iteration"] = output["iteration_count"]
    if "collected_paper_ids" in output:
        summary["new_papers"] = len(output["collected_paper_ids"])
    if "extracted_factors" in output:
        summary["new_factors"] = len(output["extracted_factors"])
    if "backtest_results" in output:
        summary["new_results"] = len(output["backtest_results"])

    return json.dumps(summary, ensure_ascii=False)


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
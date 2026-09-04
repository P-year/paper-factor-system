# Paper Factor System API

FastAPI 化部署，让 Agent 可被任意 HTTP 客户端调用。

## 快速启动

```bash
# 本地开发
uvicorn api.server:app --reload --port 8000

# 或直接跑
python -m api.server

# Docker
docker build -t paper-factor-api .
docker run -p 8000:8000 --env-file .env paper-factor-api
```

## 接口列表

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 服务信息 |
| GET | `/health` | 健康检查 |
| POST | `/agent/run` | 同步运行（返回最终结果） |
| POST | `/agent/stream` | 流式运行（SSE，逐步推送 node 事件） |
| GET | `/sessions/{thread_id}` | 查询会话 checkpoint |

## 用法示例

### 同步调用

```bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"goal": "分析过去90天arxiv上关于动量因子的论文"}'
```

返回：
```json
{
    "thread_id": "api-1724234567",
    "status": "completed",
    "iteration_count": 5,
    "visited_nodes": ["plan", "collect", "analyze", ...],
    "collected_paper_count": 3,
    "extracted_factor_count": 2,
    "backtest_result_count": 2,
    "final_message": "...",
    "elapsed_seconds": 42.3
}
```

### 流式调用（SSE）

```bash
curl -N -X POST http://localhost:8000/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"goal": "挖掘质量因子"}'
```

返回事件流：
```
event: start
data: {"thread_id": "...", "goal": "..."}

event: node
data: {"node": "collect", "summary": {...}}

event: node
data: {"node": "analyze", "summary": {...}}

event: done
data: {"thread_id": "..."}
```

### 会话恢复

```bash
# 查询某个 thread_id 的 checkpoint
curl http://localhost:8000/sessions/api-1724234567
```

## 架构

```
HTTP Client
    │
    ▼
FastAPI (api/server.py)
    │
    ▼
LangGraph Agent (graph.py)
    │
    ├─ interrupt_before_decide: 关键环节可挂起
    ├─ MemorySaver: 会话持久化
    └─ Langfuse (optional): 全链路追踪
```

## 配置

通过环境变量：

```bash
DEEPSEEK_API_KEY=sk-xxx      # 主 LLM
GLM_API_KEY=glm-xxx          # 备 LLM
LANGFUSE_PUBLIC_KEY=pk-xxx   # 可观测性（可选）
LANGFUSE_SECRET_KEY=sk-xxx
```
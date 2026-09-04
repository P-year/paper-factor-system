"""
harness/api.py - 简洁 Python API（给论文 → 跑 pipeline → 返回结果）

目标用户：research notebook / 一次性脚本 / Web 后端。

用法：
    from harness.api import run_paper_through_pipeline

    # 1. 给 dict
    result = run_paper_through_pipeline({
        "title": "Momentum in A-shares",
        "summary": "...",
        "link": "https://arxiv.org/abs/2501.00000",
    })

    # 2. 给 arxiv_id
    result = run_paper_through_pipeline("2502.12345")

    # 3. 给多个（混合类型）
    result = run_paper_through_pipeline([
        "2502.12345",
        {"title": "...", "summary": "..."},
        "path/to/local_papers.json",
    ])

    # 4. 指定 pipeline
    result = run_paper_through_pipeline(paper, pipeline_name="paper_factor")

    # 5. 不跳过 collect（罕见场景）
    result = run_paper_through_pipeline(paper, skip_collect=False)

返回结构：
    {
      "factors": List[Dict],       # extracted_factors
      "backtests": List[Dict],     # backtest_results
      "visited_nodes": List[str],
      "state": Dict,               # 完整 state（return_full_state=True 时）
      "duration_s": float,
      "error": str | None,
    }
"""
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from harness.registry import get as _registry_get
from harness.graph_builder import build_graph as _build_graph
from harness.eval.paper_loader import (
    normalize_papers,
    write_papers_to_collection,
)


PaperArg = Union[Dict, str, Path]
PaperArgList = Union[PaperArg, List[PaperArg]]


def run_paper_through_pipeline(
    paper_input: PaperArgList,
    *,
    pipeline_name: str = "paper_factor",
    goal: Optional[str] = None,
    skip_collect: bool = True,
    skip_approve: bool = True,
    return_full_state: bool = False,
) -> Dict[str, Any]:
    """
    给论文 → 跑 pipeline → 返回结果。

    Args:
        paper_input: 单个或多个论文输入。支持：
            - dict: {source: "dict", title, summary, link}
            - dict: {source: "arxiv_id", id: "2502.12345"}
            - dict: {source: "json_file", path: "..."}
            - str: "2502.12345"（便捷写法，视为 arxiv_id）
            - Path: 本地文件路径（视为 json_file）
            - List[上述任意]
        pipeline_name: 默认 "paper_factor"
        goal: 可选目标描述（默认 "extract and backtest"）
        skip_collect: True = 跳过 collect（推荐）
        skip_approve: True = 不挂 decide（直接跑完）
        return_full_state: True = 返回完整 state

    Returns:
        dict 包含 factors / backtests / visited_nodes / duration_s / error
        可选 state（return_full_state=True 时）
    """
    return asyncio.run(
        _arun_paper_through_pipeline(
            paper_input=paper_input,
            pipeline_name=pipeline_name,
            goal=goal,
            skip_collect=skip_collect,
            skip_approve=skip_approve,
            return_full_state=return_full_state,
        )
    )


async def _arun_paper_through_pipeline(
    paper_input: PaperArgList,
    pipeline_name: str,
    goal: Optional[str],
    skip_collect: bool,
    skip_approve: bool,
    return_full_state: bool,
) -> Dict[str, Any]:
    started = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pipeline = _registry_get(pipeline_name)
    if pipeline is None:
        return {
            "factors": [],
            "backtests": [],
            "visited_nodes": [],
            "state": {},
            "duration_s": 0,
            "error": f"pipeline '{pipeline_name}' not registered",
        }

    # 1. 标准化 paper 输入
    if not isinstance(paper_input, list):
        paper_input = [paper_input]
    try:
        papers = normalize_papers(paper_input)
    except Exception as e:
        return {
            "factors": [],
            "backtests": [],
            "visited_nodes": [],
            "state": {},
            "duration_s": 0,
            "error": f"paper normalize failed: {e}",
        }

    if not papers:
        return {
            "factors": [],
            "backtests": [],
            "visited_nodes": [],
            "state": {},
            "duration_s": 0,
            "error": "no papers provided",
        }

    # 2. 写到 PAPER_DIR（让 analyze_node 能 load）
    write_papers_to_collection(papers)

    # 3. 构造 state
    state: Dict[str, Any] = {
        # BasePipelineState
        "plan": [],
        "current_step_idx": 0,
        "iteration_count": 0,
        "visited_nodes": [],
        "errors": [],
        "messages": [],
        "session_id": f"api_{timestamp}",
        "run_mode": "api",
        "pipeline_name": pipeline_name,
        "human_action": None,
        # PaperFactorState
        "collected_paper_ids": [p["arxiv_id"] for p in papers] if skip_collect else [],
        "extracted_factors": [],
        "persisted_count": 0,
        "persist_attempted": False,
        "quality_results": [],
        "backtest_results": [],
        "pending_decisions": [],
    }

    state["goal"] = goal or "extract factors from provided paper and backtest"

    # 4. 构造图
    interrupt_before: List[str] = []
    if not skip_approve:
        interrupt_before = ["decide"]

    graph, _ = _build_graph(pipeline, interrupt_before=interrupt_before)
    config = {"configurable": {"thread_id": f"api_{timestamp}"}}

    # 5. invoke
    final_state: Dict[str, Any] = {}
    try:
        if skip_collect:
            # state 已经有 papers，hard_rule 1 + fallback 会直接 force analyze
            final_state = await graph.ainvoke(state, config=config)
        else:
            final_state = await graph.ainvoke(state, config=config)
    except Exception as e:
        return {
            "factors": [],
            "backtests": [],
            "visited_nodes": [],
            "state": {},
            "duration_s": round(time.time() - started, 2),
            "error": f"graph invoke failed: {e}",
        }

    # v3-1: 注入 cost_summary 到 final_state
    from harness.cost import get_default_tracker
    tracker = get_default_tracker()
    final_state["cost_summary"] = tracker.summary()
    final_state["cost_usages"] = [
        {k: v for k, v in u.__dict__.items() if v != ""}  # strip 空字段
        for u in tracker.usages
    ]

    # 6. 提取结果
    result = {
        "factors": final_state.get("extracted_factors", []),
        "backtests": final_state.get("backtest_results", []),
        "visited_nodes": final_state.get("visited_nodes", []),
        "duration_s": round(time.time() - started, 2),
        "error": None,
    }
    if return_full_state:
        result["state"] = final_state
    return result


# === 便捷函数 ===

def list_papers_in_collection() -> List[Dict[str, Any]]:
    """列出 data/papers/ 下所有已采集/已注入的论文。"""
    from harness.paths import PAPER_DIR
    import json

    out = []
    for fp in sorted(PAPER_DIR.glob("papers_*.json"), reverse=True):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            for p in data.get("papers", []):
                p["_source_file"] = fp.name
                out.append(p)
        except Exception:
            continue
    return out
"""
test_harness_api.py - harness.api 单测（mock graph.invoke，避免真实跑 LLM）

覆盖：
1. 错误处理：pipeline 不存在、paper 解析失败、无 papers
2. 返回结构完整（factors / backtests / visited_nodes / duration_s）
3. list_papers_in_collection 列出 data/papers/
4. return_full_state 开关
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.api import run_paper_through_pipeline, list_papers_in_collection


# === 错误处理 ===

def test_invalid_pipeline_name():
    result = run_paper_through_pipeline(
        {"source": "dict", "title": "T", "summary": "S"},
        pipeline_name="nonexistent",
    )
    assert "error" in result
    assert "nonexistent" in result["error"]
    assert result["factors"] == []


def test_empty_paper_input_list():
    result = run_paper_through_pipeline([])
    assert "error" in result
    assert "no papers" in result["error"]


def test_invalid_paper_input_type():
    """数字等无法解析的类型。"""
    result = run_paper_through_pipeline(42)
    assert "error" in result


def test_arxiv_id_missing_required():
    """arxiv_id source 但缺 id 字段。"""
    result = run_paper_through_pipeline({"source": "arxiv_id"})
    assert "error" in result


# === 正常流程（mock graph） ===

def test_dict_input_mocks_graph(monkeypatch):
    """用 dict 输入 + mock graph.invoke 验证 state 注入正确。"""
    captured = {}

    async def mock_ainvoke(state, config=None):
        captured["state"] = dict(state)
        # 模拟返回：提取了一个因子、跑了一个回测
        return {
            **state,
            "extracted_factors": [{"factor_name": "momentum_1m", "paper_title": "Test"}],
            "backtest_results": [{"factor_name": "momentum_1m", "IC": 0.03}],
            "visited_nodes": ["plan", "analyze", "persist_factors", "quality_check", "backtest", "respond"],
        }

    # mock build_graph 避免构造真实图
    class FakeGraph:
        async def ainvoke(self, state, config=None):
            return await mock_ainvoke(state, config)

    def fake_build_graph(pipeline, interrupt_before=None):
        return FakeGraph(), None

    from harness.api import _arun_paper_through_pipeline
    import harness.api

    monkeypatch.setattr(harness.api, "_build_graph", fake_build_graph)

    import asyncio
    result = asyncio.run(_arun_paper_through_pipeline(
        paper_input={
            "source": "dict",
            "title": "Test momentum paper",
            "summary": "abstract here",
            "link": "https://test/p1",
        },
        pipeline_name="paper_factor",
        goal="test goal",
        skip_collect=True,
        skip_approve=True,
        return_full_state=False,
    ))

    # 验证 state 注入
    assert "https://test/p1" in captured["state"]["collected_paper_ids"]
    assert captured["state"]["goal"] == "test goal"
    assert captured["state"]["pipeline_name"] == "paper_factor"

    # 验证返回
    assert len(result["factors"]) == 1
    assert result["factors"][0]["factor_name"] == "momentum_1m"
    assert len(result["backtests"]) == 1
    assert result["backtests"][0]["IC"] == 0.03
    assert "analyze" in result["visited_nodes"]
    assert result["error"] is None
    assert "duration_s" in result


def test_list_input_mixed_types(monkeypatch):
    """混合多种 paper 输入类型。"""
    captured = {}

    async def mock_ainvoke(state, config=None):
        captured["state"] = dict(state)
        return {
            **state,
            "extracted_factors": [],
            "backtest_results": [],
            "visited_nodes": ["plan", "analyze", "respond"],
        }

    class FakeGraph:
        async def ainvoke(self, state, config=None):
            return await mock_ainvoke(state, config)

    def fake_build_graph(pipeline, interrupt_before=None):
        return FakeGraph(), None

    import harness.api
    monkeypatch.setattr(harness.api, "_build_graph", fake_build_graph)

    from harness.api import _arun_paper_through_pipeline
    import asyncio

    # 第二个是 dict（无 arxiv_id 无 link）
    result = asyncio.run(_arun_paper_through_pipeline(
        paper_input=[
            {"source": "dict", "title": "P1", "summary": "S1", "link": "https://x/p1"},
            {"source": "dict", "title": "P2", "summary": "S2"},  # 无 link → 自动 id
        ],
        pipeline_name="paper_factor",
        goal=None,
        skip_collect=True,
        skip_approve=True,
        return_full_state=False,
    ))

    assert "https://x/p1" in captured["state"]["collected_paper_ids"]
    # 第二篇的 arxiv_id 应是 manual:xxx 形式
    assert any(pid.startswith("manual:") for pid in captured["state"]["collected_paper_ids"])
    assert len(captured["state"]["collected_paper_ids"]) == 2


def test_return_full_state(monkeypatch):
    """return_full_state=True 返回完整 state。"""
    async def mock_ainvoke(state, config=None):
        return {**state, "extracted_factors": [{"factor_name": "f"}], "backtest_results": [], "visited_nodes": ["plan"]}

    class FakeGraph:
        async def ainvoke(self, state, config=None):
            return await mock_ainvoke(state, config)

    def fake_build_graph(pipeline, interrupt_before=None):
        return FakeGraph(), None

    import harness.api
    monkeypatch.setattr(harness.api, "_build_graph", fake_build_graph)

    from harness.api import _arun_paper_through_pipeline
    import asyncio

    result = asyncio.run(_arun_paper_through_pipeline(
        paper_input={"source": "dict", "title": "T", "summary": "S", "link": "L1"},
        pipeline_name="paper_factor",
        goal=None,
        skip_collect=True,
        skip_approve=True,
        return_full_state=True,
    ))
    assert "state" in result
    assert result["state"]["collected_paper_ids"] == ["L1"]
    assert result["state"]["extracted_factors"][0]["factor_name"] == "f"


def test_no_return_full_state(monkeypatch):
    """默认不返回完整 state。"""
    async def mock_ainvoke(state, config=None):
        return {**state, "extracted_factors": [], "backtest_results": [], "visited_nodes": []}

    class FakeGraph:
        async def ainvoke(self, state, config=None):
            return await mock_ainvoke(state, config)

    def fake_build_graph(pipeline, interrupt_before=None):
        return FakeGraph(), None

    import harness.api
    monkeypatch.setattr(harness.api, "_build_graph", fake_build_graph)

    from harness.api import _arun_paper_through_pipeline
    import asyncio

    result = asyncio.run(_arun_paper_through_pipeline(
        paper_input={"source": "dict", "title": "T", "summary": "S", "link": "L"},
        pipeline_name="paper_factor",
        goal=None,
        skip_collect=True,
        skip_approve=True,
        return_full_state=False,
    ))
    assert "state" not in result


def test_graph_invoke_exception(monkeypatch):
    """graph.invoke 抛错时返回 error。"""
    class FakeGraph:
        async def ainvoke(self, state, config=None):
            raise RuntimeError("graph boom")

    def fake_build_graph(pipeline, interrupt_before=None):
        return FakeGraph(), None

    import harness.api
    monkeypatch.setattr(harness.api, "_build_graph", fake_build_graph)

    from harness.api import _arun_paper_through_pipeline
    import asyncio

    result = asyncio.run(_arun_paper_through_pipeline(
        paper_input={"source": "dict", "title": "T", "summary": "S", "link": "L"},
        pipeline_name="paper_factor",
        goal=None,
        skip_collect=True,
        skip_approve=True,
        return_full_state=False,
    ))
    assert result["error"] is not None
    assert "graph boom" in result["error"]


# === list_papers_in_collection ===

def test_list_papers_in_collection_returns_list():
    papers = list_papers_in_collection()
    # 不为空（之前已写过 paper 文件）
    assert isinstance(papers, list)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
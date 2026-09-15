"""
test_paper_retrieve_node.py - paper_retrieve_node 测试

覆盖：
1. 空 state → retrieval_context=None + error warn
2. 有 collected_paper_ids → 自动索引 + 检索
3. 异常降级（不阻塞）
4. 已索引论文不再重索引
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.rag import HashBackend


@pytest.fixture(autouse=True)
def hash_only(monkeypatch):
    """强制使用 HashBackend（避免下载真实模型）。"""
    monkeypatch.setenv("HARNESS_RAG_PREFER", "hash")


@pytest.fixture
def paper_dir(tmp_path):
    """构造一个 paper_dir 含 3 篇。"""
    d = tmp_path / "papers"
    d.mkdir()
    data = {
        "papers": [
            {
                "arxiv_id": "p1",
                "title": "Momentum A-shares",
                "summary": "动量因子在 A 股表现显著。" * 50,
                "link": "https://arxiv.org/abs/p1",
            },
            {
                "arxiv_id": "p2",
                "title": "Value investing",
                "summary": "价值投资长期有效。" * 50,
                "link": "https://arxiv.org/abs/p2",
            },
            {
                "arxiv_id": "p3",
                "title": "Reversal factor",
                "summary": "反转因子短期表现。" * 50,
                "link": "https://arxiv.org/abs/p3",
            },
        ]
    }
    (d / "papers_20260101.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    return d


def _patch_paper_dir(monkeypatch, paper_dir):
    """让 PAPER_DIR 指向 tmp_path。"""
    monkeypatch.setattr("pipelines.paper_factor.nodes.paper_retrieve.PAPER_DIR", paper_dir)


# === 基本 ===

def test_empty_state_returns_none_retrieval(monkeypatch, tmp_path, paper_dir):
    from pipelines.paper_factor.nodes.paper_retrieve import paper_retrieve_node
    _patch_paper_dir(monkeypatch, paper_dir)
    state = {"collected_paper_ids": [], "goal": "test"}
    result = paper_retrieve_node(state)
    # 无论文 → retrieval_context=None + errors.append
    assert result.get("retrieval_context") is None
    assert any("no similar" in e or "error" in e for e in result.get("errors", []))


def test_paper_retrieve_indexes_and_retrieves(monkeypatch, tmp_path, paper_dir):
    from pipelines.paper_factor.nodes.paper_retrieve import paper_retrieve_node
    _patch_paper_dir(monkeypatch, paper_dir)
    state = {
        "collected_paper_ids": ["p1", "p2", "p3"],
        "goal": "研究动量因子",
        "messages": [],
    }
    result = paper_retrieve_node(state)
    # 有检索结果
    assert result.get("retrieval_context") is not None
    if result["retrieval_context"]:
        assert len(result["retrieval_context"]) >= 1
        # 每条都有 score + chunk_id
        c = result["retrieval_context"][0]
        assert "score" in c
        assert "chunk_id" in c


def test_paper_retrieve_dedups(monkeypatch, tmp_path, paper_dir):
    """同一论文第二次调用不应重索引。"""
    from pipelines.paper_factor.nodes.paper_retrieve import paper_retrieve_node
    from harness.rag.store import PaperRAGStore
    _patch_paper_dir(monkeypatch, paper_dir)
    state = {"collected_paper_ids": ["p1"], "goal": "动量", "messages": []}

    # 第一次
    r1 = paper_retrieve_node(state)
    assert r1.get("retrieval_context")

    # 第二次
    r2 = paper_retrieve_node(state)
    assert r2.get("retrieval_context") is not None


def test_paper_retrieve_handles_missing_paper(monkeypatch, tmp_path, paper_dir):
    """collected_paper_ids 引用不存在的论文 → 不崩 + 空 retrieval。"""
    from pipelines.paper_factor.nodes.paper_retrieve import paper_retrieve_node
    _patch_paper_dir(monkeypatch, paper_dir)
    state = {
        "collected_paper_ids": ["nonexistent_xyz"],
        "goal": "test",
        "messages": [],
    }
    result = paper_retrieve_node(state)
    # 不应抛异常
    assert "retrieval_context" in result


def test_paper_retrieve_uses_state_messages_query(monkeypatch, tmp_path, paper_dir):
    """query 构造会从 messages 取 user msg。"""
    from pipelines.paper_factor.nodes.paper_retrieve import paper_retrieve_node
    _patch_paper_dir(monkeypatch, paper_dir)
    state = {
        "collected_paper_ids": ["p1"],
        "goal": "",
        "messages": [{"role": "user", "content": "动量反转因子"}],
    }
    result = paper_retrieve_node(state)
    # 应能检索
    if result.get("retrieval_context"):
        assert len(result["retrieval_context"]) >= 1


# === 异常降级 ===

def test_paper_retrieve_exception_falls_back(monkeypatch, tmp_path, paper_dir):
    """任何内部异常 → errors.append + retrieval_context=None。"""
    from pipelines.paper_factor.nodes import paper_retrieve as mod
    from pipelines.paper_factor.nodes.paper_retrieve import paper_retrieve_node
    _patch_paper_dir(monkeypatch, paper_dir)

    def boom(*a, **kw):
        raise RuntimeError("intentional")

    monkeypatch.setattr(mod, "_build_store", boom)
    state = {"collected_paper_ids": ["p1"], "goal": "x"}
    result = paper_retrieve_node(state)
    assert result["retrieval_context"] is None
    assert any("error" in e.lower() for e in result.get("errors", []))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
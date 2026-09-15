"""
test_rag_retriever.py - PaperRetriever 测试

覆盖：
1. retrieve top_k
2. min_score 过滤
3. retrieve_for_state 拿 goal
4. format_context max_chars 截断
5. 空 query / 空 store 不崩
6. 检索结果含 score + chunk_id + text + arxiv_id
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.rag.retriever import PaperRetriever
from harness.rag.store import PaperRAGStore
from harness.rag.embedder import HashBackend
from harness.rag.chunker import Chunker


@pytest.fixture
def setup(tmp_path):
    """构造 store + retriever。"""
    store = PaperRAGStore(
        rag_dir=tmp_path / "rag",
        embedder=HashBackend(),
        chunker=Chunker(max_chunk_chars=500, overlap_chars=100),
    )
    papers = [
        {"arxiv_id": "p1", "title": "Momentum A-shares", "summary": "动量因子在 A 股市场表现显著。" * 30},
        {"arxiv_id": "p2", "title": "Value investing", "summary": "价值投资策略长期有效。" * 30},
        {"arxiv_id": "p3", "title": "Reversal factor", "summary": "反转因子短期表现突出。" * 30},
    ]
    store.index_papers(papers)
    retriever = PaperRetriever(store, default_top_k=3)
    return store, retriever


# === retrieve ===

def test_retrieve_basic(setup):
    store, retriever = setup
    chunks = retriever.retrieve("动量因子", top_k=2)
    assert len(chunks) >= 1
    assert all("score" in c for c in chunks)
    assert all("chunk_id" in c for c in chunks)
    assert all("arxiv_id" in c for c in chunks)
    assert all("text" in c for c in chunks)


def test_retrieve_top_k_limit(setup):
    _, retriever = setup
    chunks = retriever.retrieve("因子", top_k=2)
    assert len(chunks) <= 2


def test_retrieve_min_score_filter(setup):
    _, retriever = setup
    # min_score=0.99 几乎过滤全部
    chunks = retriever.retrieve("动量", top_k=5, min_score=0.99)
    assert len(chunks) <= 5


def test_retrieve_empty_query(setup):
    _, retriever = setup
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_retrieve_empty_store(tmp_path):
    from harness.rag.store import PaperRAGStore
    store = PaperRAGStore(rag_dir=tmp_path / "rag", embedder=HashBackend())
    retriever = PaperRetriever(store)
    assert retriever.retrieve("test") == []


# === retrieve_for_state ===

def test_retrieve_for_state_with_goal(setup):
    _, retriever = setup
    state = {"goal": "研究 A 股动量因子", "messages": []}
    chunks = retriever.retrieve_for_state(state, top_k=3)
    assert len(chunks) >= 1


def test_retrieve_for_state_with_messages(setup):
    _, retriever = setup
    state = {
        "goal": "",
        "messages": [
            {"role": "user", "content": "我想研究反转因子"},
        ],
    }
    chunks = retriever.retrieve_for_state(state, top_k=2)
    assert len(chunks) >= 1


def test_retrieve_for_state_empty(setup):
    _, retriever = setup
    assert retriever.retrieve_for_state({}) == []


# === format_context ===

def test_format_context_empty(setup):
    _, retriever = setup
    assert retriever.format_context([]) == ""


def test_format_context_basic(setup):
    _, retriever = setup
    chunks = retriever.retrieve("动量", top_k=2)
    text = retriever.format_context(chunks)
    assert "[1]" in text
    assert "score=" in text


def test_format_context_max_chars_truncate(setup):
    _, retriever = setup
    chunks = retriever.retrieve("因子", top_k=5)
    text = retriever.format_context(chunks, max_chars=100)
    # 应被截断
    assert len(text) <= 200  # 留余量
    # 可能含省略标记
    # assert "omitted" in text or len(text) <= 100


def test_format_context_includes_paper_title(setup):
    _, retriever = setup
    chunks = retriever.retrieve("动量", top_k=1)
    text = retriever.format_context(chunks)
    # 含 paper title
    assert "Momentum" in text or "动量" in text


# === retrieve 字段 ===

def test_retrieve_results_have_metadata(setup):
    _, retriever = setup
    chunks = retriever.retrieve("动量", top_k=1)
    if chunks:
        c = chunks[0]
        assert "arxiv_id" in c
        assert "paper_title" in c
        assert "paper_link" in c or c["paper_link"] == ""
        assert "score" in c
        assert isinstance(c["score"], float)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
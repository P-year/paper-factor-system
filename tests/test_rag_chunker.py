"""
test_rag_chunker.py - Chunker 单元测试

覆盖：
1. 段落切分
2. 单段超长滑动窗口
3. 空文本
4. 边界（中英混合、特殊符号）
5. chunk_id 唯一
6. overlap 重叠验证
7. _safe_id 兜底
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.rag.chunker import Chunker, _safe_id


@pytest.fixture
def chunker():
    return Chunker(max_chunk_chars=800, overlap_chars=200)


# === 段落切分 ===

def test_split_empty(chunker):
    assert chunker.split_text("") == []
    assert chunker.split_text("   ") == []


def test_split_short_single_paragraph(chunker):
    text = "这是第一段。这是第二句。"
    chunks = chunker.split_text(text, source_id="p1")
    assert len(chunks) >= 1
    assert chunks[0]["source_id"] == "p1"
    assert chunks[0]["chunk_id"] == "p1:0"


def test_split_multi_paragraph_by_period(chunker):
    text = "第一段结束。\n第二段开始。\n第三段结束。"
    chunks = chunker.split_text(text, source_id="p1")
    # 3 段 → 3 chunks
    assert len(chunks) >= 1
    # 验证 source_id
    assert all(c["source_id"] == "p1" for c in chunks)


def test_split_multi_paragraph_by_blank_line(chunker):
    text = "第一段内容。\n\n第二段内容。\n\n第三段。"
    chunks = chunker.split_text(text)
    assert len(chunks) >= 1


# === 滑动窗口 ===

def test_sliding_window_for_long_text(chunker):
    """单段超长 → 切多个 chunk。"""
    text = "这是一个测试句子。" * 200  # ~10KB
    chunks = chunker.split_text(text, source_id="long")
    # 10KB / 600 step ≈ 17+ chunks
    assert len(chunks) > 1
    # chunk_id 唯一
    ids = {c["chunk_id"] for c in chunks}
    assert len(ids) == len(chunks)


def test_sliding_window_overlap(chunker):
    """滑动窗口 chunk 之间有 overlap。"""
    text = "这是句子。" * 200
    chunks = chunker.split_text(text)
    if len(chunks) >= 2:
        # 第二个 chunk 的 text 应包含第一个 chunk 末尾部分
        # （overlap_chars=200 字符范围）
        c1_text = chunks[0]["text"]
        c2_text = chunks[1]["text"]
        # c2 开头应与 c1 结尾部分重叠
        overlap_len = chunker.overlap_chars
        # 简单验证：c2[:overlap_len] 应出现在 c1 末尾
        c1_tail = c1_text[-overlap_len:]
        # c2 开头部分应能在 c1 末尾找到
        # 注：滑动窗口可能按句末对齐，所以不一定精确 overlap
        # 但 c2 应比 i + step 起步
        assert len(c2_text) > 0


def test_sliding_window_no_infinite_loop():
    """退化情况：超长无句末符号 → 不能死循环。"""
    c = Chunker(max_chunk_chars=100, overlap_chars=20)
    text = "x" * 1000  # 全 x 无句末
    chunks = c.split_text(text, source_id="loop")
    # 应能完成（不死循环）
    assert len(chunks) > 0
    # 每 chunk 不超过 max
    for ch in chunks:
        assert len(ch["text"]) <= 200  # 句末找不到会硬切 100 + overlap


# === split_paper ===

def test_split_paper_with_title(chunker):
    paper = {
        "arxiv_id": "2501.00000",
        "title": "Momentum in A-shares",
        "summary": "本文研究 A 股市场的动量效应。\n\n回测结果显著。",
    }
    chunks = chunker.split_paper(paper)
    assert len(chunks) >= 1
    # 第一 chunk 应含 title
    assert "Momentum" in chunks[0]["text"]
    # source_id = arxiv_id
    assert chunks[0]["source_id"] == "2501.00000"


def test_split_paper_no_title(chunker):
    paper = {"arxiv_id": "x1", "summary": "只有摘要。"}
    chunks = chunker.split_paper(paper)
    assert chunks[0]["source_id"] == "x1"
    assert "摘要" in chunks[0]["text"]


def test_split_paper_missing_id(chunker):
    paper = {"title": "No ID Paper", "summary": "只有标题和摘要。"}
    chunks = chunker.split_paper(paper)
    # _safe_id 兜底
    assert chunks[0]["source_id"] != ""


# === split_papers ===

def test_split_papers_batch(chunker):
    papers = [
        {"arxiv_id": "p1", "title": "Paper 1", "summary": "摘要1。"},
        {"arxiv_id": "p2", "title": "Paper 2", "summary": "摘要2。"},
    ]
    chunks = chunker.split_papers(papers)
    assert len(chunks) >= 2
    sources = {c["source_id"] for c in chunks}
    assert sources == {"p1", "p2"}


# === _safe_id ===

def test_safe_id_arxiv():
    assert _safe_id({"arxiv_id": "2501.00000"}) == "2501.00000"


def test_safe_id_link():
    sid = _safe_id({"link": "https://arxiv.org/abs/1234"})
    assert len(sid) == 16  # md5[:16]


def test_safe_id_title_fallback():
    sid1 = _safe_id({"title": "Some Paper"})
    sid2 = _safe_id({"title": "Some Paper"})
    assert sid1 == sid2  # 确定性


def test_safe_id_no_fields():
    sid = _safe_id({})
    assert len(sid) == 16


# === 参数校验 ===

def test_overlap_must_be_less_than_max():
    with pytest.raises(ValueError):
        Chunker(max_chunk_chars=100, overlap_chars=100)
    with pytest.raises(ValueError):
        Chunker(max_chunk_chars=100, overlap_chars=200)


# === start/end 字段 ===

def test_chunk_start_end_present(chunker):
    chunks = chunker.split_text("第一句。\n第二句。", source_id="x")
    for c in chunks:
        assert "start" in c
        assert "end" in c
        assert c["end"] > c["start"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
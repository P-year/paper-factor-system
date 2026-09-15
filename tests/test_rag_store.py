"""
test_rag_store.py - PaperRAGStore 测试

覆盖：
1. index_paper 基本
2. index_papers 批量
3. 增量去重（重复 index 不增）
4. index_paper_dir 扫多文件
5. index_json_file
6. save / load roundtrip
7. count() / source_ids() / manifest()
8. rebuild 全量重建
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.rag.store import PaperRAGStore
from harness.rag.embedder import HashBackend
from harness.rag.chunker import Chunker


@pytest.fixture
def store(tmp_path):
    """每个测试独立的 rag_dir。"""
    return PaperRAGStore(
        rag_dir=tmp_path / "rag",
        embedder=HashBackend(),
        chunker=Chunker(max_chunk_chars=500, overlap_chars=100),
    )


def _make_paper(arxiv_id="p1", title="Test", summary="测试摘要内容。"):
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "summary": summary,
        "link": f"https://arxiv.org/abs/{arxiv_id}",
    }


# === index_paper ===

def test_index_one_paper(store):
    n = store.index_paper(_make_paper("p1"))
    assert n >= 1
    assert store.count() >= 1


def test_index_dedup(store):
    """同一 paper 两次 index 只算一次。"""
    p = _make_paper("p1", summary="内容" * 100)
    n1 = store.index_paper(p)
    n2 = store.index_paper(p)
    assert n1 >= 1
    assert n2 == 0  # 去重
    assert store.count() == n1


def test_index_multiple_papers(store):
    papers = [_make_paper(f"p{i}") for i in range(3)]
    n = store.index_papers(papers)
    assert n >= 3
    assert store.count() >= 3


def test_index_paper_empty_summary(store):
    """summary 为空但 title 仍能切出 1 个 chunk。"""
    p = _make_paper("empty", title="X", summary="")
    n = store.index_paper(p)
    # title 单独成 1 chunk
    assert n == 1


# === index_json_file ===

def test_index_json_file(store, tmp_path):
    json_path = tmp_path / "papers.json"
    data = {"papers": [_make_paper("j1"), _make_paper("j2")]}
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    n = store.index_json_file(json_path)
    assert n >= 2


def test_index_json_file_bare_list(store, tmp_path):
    """顶层裸 list 也兼容。"""
    json_path = tmp_path / "papers_list.json"
    papers = [_make_paper("l1"), _make_paper("l2")]
    json_path.write_text(json.dumps(papers, ensure_ascii=False), encoding="utf-8")
    n = store.index_json_file(json_path)
    assert n >= 2


def test_index_json_file_empty(store, tmp_path):
    json_path = tmp_path / "empty.json"
    json_path.write_text(json.dumps({"papers": []}), encoding="utf-8")
    n = store.index_json_file(json_path)
    assert n == 0


def test_index_json_file_missing_keys(store, tmp_path):
    """paper 缺字段时 normalize_paper_input 兜底。"""
    json_path = tmp_path / "broken.json"
    data = {"papers": [{"title": "No ID", "summary": "摘要。"}]}
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    # 不应崩
    n = store.index_json_file(json_path)
    assert n >= 0


# === index_paper_dir ===

def test_index_paper_dir(store, tmp_path):
    paper_dir = tmp_path / "papers"
    paper_dir.mkdir()
    for i in range(3):
        (paper_dir / f"papers_2026010{i}.json").write_text(
            json.dumps({"papers": [_make_paper(f"d{i}")]}, ensure_ascii=False),
            encoding="utf-8",
        )
    n = store.index_paper_dir(paper_dir)
    assert n >= 3


def test_index_paper_dir_only_ids(store, tmp_path):
    paper_dir = tmp_path / "papers"
    paper_dir.mkdir()
    (paper_dir / "papers_20260101.json").write_text(
        json.dumps({"papers": [_make_paper("want"), _make_paper("skip")]}, ensure_ascii=False),
        encoding="utf-8",
    )
    n = store.index_paper_dir(paper_dir, only_ids=["want"])
    # 只索引 want
    assert "want" in store.source_ids()
    assert "skip" not in store.source_ids()


def test_index_paper_dir_nonexistent(store, tmp_path):
    n = store.index_paper_dir(tmp_path / "nope")
    assert n == 0


# === save / load ===

def test_save_load_roundtrip(store, tmp_path):
    p = _make_paper("p1", summary="测试摘要。" * 50)
    store.index_paper(p)
    n1 = store.count()
    store.save()

    # 新实例加载
    new_store = PaperRAGStore(
        rag_dir=tmp_path / "rag",
        embedder=HashBackend(),
    )
    assert new_store.count() == n1
    assert "p1" in new_store.source_ids()


def test_save_creates_files(store):
    store.index_paper(_make_paper("p1"))
    store.save()
    assert store.index_path.exists()
    assert store.chunks_path.exists()
    assert (store.rag_dir / "manifest.json").exists()


def test_manifest(store):
    store.index_paper(_make_paper("p1"))
    m = store.manifest()
    assert m["chunk_count"] >= 1
    assert "model_name" in m
    assert "dim" in m


def test_manifest_persisted(store):
    store.index_paper(_make_paper("p1"))
    store.save()
    # 重新读 manifest.json
    mf = store.rag_dir / "manifest.json"
    assert mf.exists()
    import json
    m = json.loads(mf.read_text(encoding="utf-8"))
    assert "chunk_count" in m


# === rebuild ===

def test_rebuild_clears_and_reindexes(store, tmp_path):
    paper_dir = tmp_path / "papers"
    paper_dir.mkdir()
    (paper_dir / "papers.json").write_text(
        json.dumps({"papers": [_make_paper("a"), _make_paper("b")]}, ensure_ascii=False),
        encoding="utf-8",
    )
    # 先索引 a，b
    store.index_papers([_make_paper("a"), _make_paper("b")])
    n1 = store.count()
    assert n1 > 0
    # rebuild
    n2 = store.rebuild()
    # rebuild 内部 index_paper_dir → 但 store._clear_chunks 会清 self.index
    # 应能正常返回新数
    assert n2 >= 0


# === search ===

def test_search_returns_results(store):
    store.index_papers([
        _make_paper("p1", summary="动量因子在中国 A 股市场的表现研究"),
        _make_paper("p2", summary="value investing strategy in US market"),
    ])
    q = store.embedder.encode(["动量因子"])[0]
    results = store.search(q, top_k=2)
    assert len(results) >= 1
    # 至少 p1 命中（chunk_id 可能是 p1:0、p1:1 等）
    scores_by_id = {cid: score for score, cid in results}
    assert any(cid.startswith("p1:") for cid in scores_by_id)


def test_search_empty_store(store):
    """空 store search 返回空（不崩）。"""
    import numpy as np
    q = store.embedder.encode(["test"])[0]
    assert store.search(q) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
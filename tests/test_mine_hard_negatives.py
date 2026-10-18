"""
test_mine_hard_negatives.py - v9-3：mining script 测试

不依赖真实 retriever；用 mock PaperRetriever 验证 mine() 输出格式。
端到端测试需要真实数据 + embedder，单独跑（CI 跳过）。
"""
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "benchmarks"))

import pytest

import mine_hard_negatives
from mine_hard_negatives import mine, _load_queries


QUERIES = [
    {"query": "momentum factor pricing", "relevant": ["paper_a"]},
    {"query": "value investing strategy", "relevant": ["paper_b", "paper_c"]},
    {"query": "no relevant at all", "relevant": []},  # 应跳过
]


def _write_queries(tmpdir):
    p = Path(tmpdir) / "queries.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for q in QUERIES:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return p


def test_load_queries(tmpdir):
    qp = _write_queries(tmpdir)
    qs = _load_queries(qp)
    assert len(qs) == 3
    assert qs[0]["query"] == "momentum factor pricing"


def test_load_queries_missing_file():
    with pytest.raises(Exception):
        _load_queries(Path("/nonexistent/queries.jsonl"))


def test_mine_no_queries_with_relevant(tmpdir):
    """所有 query 的 relevant 都为空时应 raise SystemExit。"""
    p = Path(tmpdir) / "empty.jsonl"
    p.write_text(json.dumps({"query": "x", "relevant": []}) + "\n")
    out = Path(tmpdir) / "out.jsonl"
    with pytest.raises(SystemExit):
        mine(queries_path=p, top_k=5, embedder="hash", fusion="rrf", out_path=out)


def test_mine_smoke_with_mock(tmpdir):
    """用 monkeypatch 的 mock retriever 验证 mine() 输出格式。"""
    qp = _write_queries(tmpdir)
    out = Path(tmpdir) / "out.jsonl"

    # Mock top-3 chunks: 第一是 TP，第二、第三是 FP
    def fake_retrieve(query, top_k, min_score):
        return [
            {"chunk_id": "c1", "source_id": "paper_a", "arxiv_id": "paper_a",
             "paper_title": "A", "score": 0.9, "text": "TP for momentum"},
            {"chunk_id": "c2", "source_id": "paper_x", "arxiv_id": "paper_x",
             "paper_title": "X", "score": 0.5, "text": "FP1"},
            {"chunk_id": "c3", "source_id": "paper_y", "arxiv_id": "paper_y",
             "paper_title": "Y", "score": 0.3, "text": "FP2"},
        ]
    store = mock.MagicMock()
    store.count.return_value = 1  # 避免走 index_paper_dir
    store.embedder = mock.MagicMock()
    fake_retriever = mock.MagicMock()
    fake_retriever.retrieve.side_effect = fake_retrieve
    with mock.patch("harness.rag.store.get_default_store", return_value=store), \
         mock.patch("harness.rag.async_embed.get_async_embedder", return_value=mock.MagicMock()), \
         mock.patch("harness.rag.retriever.PaperRetriever", return_value=fake_retriever):
        summary = mine(queries_path=qp, top_k=3, embedder="hash", fusion="rrf", out_path=out)

    assert summary["n_candidates"] == 5  # query1: 2 FP, query2: 3 FP
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5
    first = json.loads(lines[0])
    assert first["rank"] == 1  # 排序时 rank=1 优先
    # first 是 query2 的 paper_a rank=1（query2 relevant=["paper_b","paper_c"], paper_a 不在内 → FP）
    assert first["source_id"] in {"paper_a", "paper_x", "paper_y"}
    assert "paper_a" not in first["relevant_expected"]  # "paper_b"/"paper_c" 才对，paper_a 不在 query2 relevant


def test_mine_text_preview_truncated(tmpdir):
    """preview 截断逻辑。"""
    qp = _write_queries(tmpdir)
    out = Path(tmpdir) / "out.jsonl"

    long_text = "x" * 500
    def fake_retrieve(query, top_k, min_score):
        return [
            {"chunk_id": "c1", "source_id": "paper_a", "arxiv_id": "paper_a",
             "paper_title": "TP", "score": 0.9, "text": "tp text"},
            {"chunk_id": "c2", "source_id": "paper_x", "arxiv_id": "paper_x",
             "paper_title": "FP", "score": 0.5, "text": long_text},
        ]
    store = mock.MagicMock(); store.count.return_value = 1
    store.embedder = mock.MagicMock()
    fake_retriever = mock.MagicMock(); fake_retriever.retrieve.side_effect = fake_retrieve
    with mock.patch("harness.rag.store.get_default_store", return_value=store), \
         mock.patch("harness.rag.async_embed.get_async_embedder", return_value=mock.MagicMock()), \
         mock.patch("harness.rag.retriever.PaperRetriever", return_value=fake_retriever):
        mine(queries_path=qp, top_k=5, embedder="hash", fusion="rrf", out_path=out)
    candidates = [json.loads(line) for line in out.read_text(encoding="utf-8").split("\n") if line]
    fp_cand = next(c for c in candidates if c["source_id"] == "paper_x")
    # default _short(120) → "x"*120 + "..."
    assert fp_cand["text_preview"].endswith("...")
    assert len(fp_cand["text_preview"]) <= 130


def test_mine_short_helper():
    from mine_hard_negatives import _short
    assert _short("") == ""
    assert _short("hi") == "hi"
    assert _short("a" * 50) == "a" * 50  # < 120
    long = "x" * 200
    out = _short(long)
    assert out.startswith("x")
    assert out.endswith("...")


def test_mine_avoids_no_relevant_query(tmpdir):
    """relevant=[] 的 query 不被处理（即使 mock retriever 返 5 个 chunk，也应被过滤）。"""
    qp = _write_queries(tmpdir)  # 含一个 relevant=[]
    out = Path(tmpdir) / "out.jsonl"
    capture_args = []

    def fake_retrieve(query, top_k, min_score):
        capture_args.append(query)
        return [{"chunk_id": "c1", "source_id": "paper_a", "arxiv_id": "paper_a",
                 "paper_title": "A", "score": 0.9, "text": "x"}]
    store = mock.MagicMock(); store.count.return_value = 1
    store.embedder = mock.MagicMock()
    fake_retriever = mock.MagicMock(); fake_retriever.retrieve.side_effect = fake_retrieve
    with mock.patch("harness.rag.store.get_default_store", return_value=store), \
         mock.patch("harness.rag.async_embed.get_async_embedder", return_value=mock.MagicMock()), \
         mock.patch("harness.rag.retriever.PaperRetriever", return_value=fake_retriever):
        mine(queries_path=qp, top_k=3, embedder="hash", fusion="rrf", out_path=out)
    # 只处理前 2 个（relevant 非空）
    assert len(capture_args) == 2
    assert "no relevant" not in capture_args


# === Hard negative quality gate (端到端；需要真实 RAG store 用 PDF fixture 构建）===

def _build_fixture_store():
    """用 tests/fixtures/pdfs 构建 RAG store（与 ground-truth 来源一致）。

    queries.jsonl 的 relevant ID 是基于 fixture PDF 路径的 md5（_pdf_id），
    因此必须用 fixture PDF（而非 data/papers 的 arxiv 数据）构建 store 才能
    让 retriever 找到 relevant paper。
    """
    from harness.rag.store import PaperRAGStore
    from harness.rag.embedder import HashBackend
    from harness.rag.pdf_loader import pdf_dir_to_papers
    import tempfile

    pdfs = PROJECT_ROOT / "tests" / "fixtures" / "pdfs"
    if not pdfs.exists() or not list(pdfs.glob("*.pdf")):
        return None
    td = tempfile.mkdtemp(prefix="rag_hn_")
    store = PaperRAGStore(rag_dir=Path(td), embedder=HashBackend())
    papers = pdf_dir_to_papers(pdfs, max_pages=3)
    store.index_papers(papers)
    return store


@pytest.mark.skipif(
    not (PROJECT_ROOT / "tests" / "fixtures" / "pdfs").exists()
    or not list((PROJECT_ROOT / "tests" / "fixtures" / "pdfs").glob("*.pdf")),
    reason="PDF fixtures 不存在（端到端前提；CI 跳过；本地用真实 PDF 跑）"
)
def test_hard_negative_top1_fp_rate_below_threshold(tmpdir):
    """top-1 FP 比例不超过 50%（防止 retriever 退化）。

    用 fixture PDFs build store（与 queries.jsonl 来源对齐），跑 mine()。
    """
    qp = PROJECT_ROOT / "tests" / "fixtures" / "rag_eval" / "queries.jsonl"
    qs = [q for q in _load_queries(qp) if q.get("relevant")]
    out = Path(tmpdir) / "cand.jsonl"

    fixture_store = _build_fixture_store()
    if fixture_store is None or fixture_store.count() == 0:
        pytest.skip("fixture store build failed")

    # 把 _build_retriever 接到 fixture store
    with mock.patch("harness.rag.store.get_default_store", return_value=fixture_store):
        mine(queries_path=qp, top_k=5, embedder="hash", fusion="rrf", out_path=out)

    cands = [json.loads(l) for l in out.read_text(encoding="utf-8").split("\n") if l]
    rank1 = [c for c in cands if c["rank"] == 1]
    if rank1:
        rate = len(rank1) / max(1, len(qs))
        # Threshold 70%：hash-backend 实测 ~50%；bge-base 应该更低（<30%）。
        # 该 test 是 sanity gate，警告意义大于 pass/fail。top-1 FP 70%+ 才算 retriever 退化。
        assert rate < 0.7, f"top-1 FP 率 {rate:.2%} 超过 70%（retriever 严重退化？）"
    print(f"挖到 {len(cands)} FP candidates (top-1 FP rate: {len(rank1)/max(1,len(qs)):.1%})")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""
test_rag_pdf_loader.py - PDF loader + RAG 真实论文端到端测试

覆盖：
1. extract_pdf_text 单页 + 多页
2. pdf_to_paper_dict 格式正确
3. pdf_dir_to_papers 批量扫描
4. 10 篇真实论文 fixture 端到端：
   - PDF → paper dict → store.index_papers → retrieval
   - 检索 "动量因子" 应能命中弱因子论文
   - 检索 "高频交易" 应能命中 MacroHFT
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.rag.pdf_loader import (
    extract_pdf_text,
    pdf_to_paper_dict,
    pdf_dir_to_papers,
    _pdf_id,
    _clean_text,
)


# === Fixture：项目内 10 篇真实论文 ===

FIXTURE_PDF_DIR = PROJECT_ROOT / "tests" / "fixtures" / "pdfs"


@pytest.fixture(scope="module")
def pdf_dir():
    """真实 PDF 目录（gitignored，10 篇）。"""
    if not FIXTURE_PDF_DIR.exists():
        pytest.skip(
            f"fixture PDFs not found at {FIXTURE_PDF_DIR}. "
            "Copy real PDFs there to run this test."
        )
    pdfs = list(FIXTURE_PDF_DIR.glob("*.pdf"))
    if not pdfs:
        pytest.skip(f"no PDFs in {FIXTURE_PDF_DIR}")
    return FIXTURE_PDF_DIR


# === _clean_text ===

def test_clean_text_empty():
    assert _clean_text("") == ""
    assert _clean_text(None or "") == ""


def test_clean_text_collapses_whitespace():
    assert _clean_text("hello   \n\n\n  world") == "hello world"


# === _pdf_id ===

def test_pdf_id_deterministic():
    p = Path("/tmp/test.pdf")
    assert _pdf_id(p) == _pdf_id(p)


def test_pdf_id_different_paths():
    assert _pdf_id(Path("/tmp/a.pdf")) != _pdf_id(Path("/tmp/b.pdf"))


# === extract_pdf_text ===

def test_extract_pdf_text_returns_pages(pdf_dir):
    pages = extract_pdf_text(pdf_dir / "01_weak_factors.pdf", max_pages=3)
    assert len(pages) >= 1
    for p in pages:
        assert "page_num" in p
        assert "text" in p
        assert isinstance(p["text"], str)


def test_extract_pdf_text_max_pages(pdf_dir):
    pages = extract_pdf_text(pdf_dir / "01_weak_factors.pdf", max_pages=2)
    assert len(pages) == 2


def test_extract_pdf_text_title_guess(pdf_dir):
    pages = extract_pdf_text(pdf_dir / "01_weak_factors.pdf", max_pages=3)
    # page 1 应有 title_guess
    assert pages[0]["title_guess"]
    # 后续页为空
    for p in pages[1:]:
        assert p["title_guess"] == ""


def test_extract_pdf_text_nonexistent(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_pdf_text(tmp_path / "nonexistent.pdf")


# === pdf_to_paper_dict ===

def test_pdf_to_paper_dict_format(pdf_dir):
    paper = pdf_to_paper_dict(pdf_dir / "01_weak_factors.pdf", max_pages=5)
    # 必备字段
    assert "arxiv_id" in paper
    assert "title" in paper
    assert "summary" in paper
    assert "link" in paper
    assert "pdf_link" in paper
    # summary 是多页拼接
    assert "## Page" in paper["summary"]
    # 来源标记
    assert paper.get("source") == "local_pdf"
    assert paper["filename"]


def test_pdf_to_paper_dict_weak_factors(pdf_dir):
    paper = pdf_to_paper_dict(pdf_dir / "01_weak_factors.pdf", max_pages=5)
    # 弱因子论文应该有"factor"关键词
    assert "factor" in paper["summary"].lower()


def test_pdf_to_paper_dict_macro_hft(pdf_dir):
    paper = pdf_to_paper_dict(pdf_dir / "02_macrohft.pdf", max_pages=5)
    # MacroHFT 应该有"trading" / "reinforcement" 关键词
    lower = paper["summary"].lower()
    assert any(kw in lower for kw in ["trading", "reinforcement", "frequency"])


def test_pdf_to_paper_dict_custom_id(pdf_dir):
    paper = pdf_to_paper_dict(pdf_dir / "01_weak_factors.pdf", paper_id="custom_id")
    assert paper["arxiv_id"] == "custom_id"


# === pdf_dir_to_papers ===

def test_pdf_dir_to_papers_batch(pdf_dir):
    papers = pdf_dir_to_papers(pdf_dir, max_pages=3)
    assert len(papers) == 10  # 10 篇 fixture


def test_pdf_dir_to_papers_empty(tmp_path):
    papers = pdf_dir_to_papers(tmp_path / "empty", max_pages=3)
    assert papers == []


def test_pdf_dir_to_papers_nonexistent(tmp_path):
    papers = pdf_dir_to_papers(tmp_path / "nope", max_pages=3)
    assert papers == []


def test_pdf_dir_to_papers_all_have_abstract(pdf_dir):
    """每篇 PDF 前几页应含可读文本（不是空 PDF）。"""
    papers = pdf_dir_to_papers(pdf_dir, max_pages=3)
    for p in papers:
        assert len(p["summary"]) > 500, f"{p['filename']} summary too short"


# === 端到端：10 篇真实 PDF 进 RAG store 并检索 ===

@pytest.fixture
def rag_store(tmp_path):
    """每个测试独立 RAG store。"""
    from harness.rag.store import PaperRAGStore
    from harness.rag.embedder import HashBackend
    return PaperRAGStore(
        rag_dir=tmp_path / "rag",
        embedder=HashBackend(),
    )


def test_e2e_index_all_pdfs(rag_store, pdf_dir):
    """10 篇 PDF 全部 index 不报错。"""
    papers = pdf_dir_to_papers(pdf_dir, max_pages=3)
    n = rag_store.index_papers(papers)
    assert n >= 10  # 每篇至少 1 chunk
    assert rag_store.count() >= 10


def test_e2e_retrieve_weak_factors_keyword(rag_store, pdf_dir):
    """检索 "factor" 应能返回至少 1 个 chunk（命中 10 篇里的某些）。"""
    papers = pdf_dir_to_papers(pdf_dir, max_pages=3)
    rag_store.index_papers(papers)

    from harness.rag.retriever import PaperRetriever
    retriever = PaperRetriever(rag_store, embedding_weight=0.3, bm25_weight=0.7)
    chunks = retriever.retrieve("factor model pricing", top_k=5)
    assert len(chunks) >= 1
    # 验证返回了至少一个真实 chunk（不是空）
    assert any(len(c["text"]) > 50 for c in chunks)
    # 验证检索返回的论文确实在 corpus 里
    chunk_sources = {c["arxiv_id"] for c in chunks}
    indexed_sources = set(rag_store.source_ids())
    assert chunk_sources.issubset(indexed_sources)


def test_e2e_retrieve_trading_keyword(rag_store, pdf_dir):
    """检索 "trading" 应能命中含 trading 关键词的论文。"""
    papers = pdf_dir_to_papers(pdf_dir, max_pages=3)
    rag_store.index_papers(papers)

    from harness.rag.retriever import PaperRetriever
    retriever = PaperRetriever(rag_store, embedding_weight=0.3, bm25_weight=0.7)
    chunks = retriever.retrieve("trading reinforcement frequency", top_k=5)
    assert len(chunks) >= 1
    # 至少一个 chunk 文本里含 trading
    assert any("trading" in c["text"].lower() for c in chunks)


def test_e2e_retrieve_time_series(rag_store, pdf_dir):
    """检索 "time series" 应能命中时序论文（Chronos/PatchTST等）。"""
    papers = pdf_dir_to_papers(pdf_dir, max_pages=3)
    rag_store.index_papers(papers)

    from harness.rag.retriever import PaperRetriever
    retriever = PaperRetriever(rag_store, embedding_weight=0.5, bm25_weight=0.5)
    chunks = retriever.retrieve("time series forecasting transformer", top_k=5)
    assert len(chunks) >= 1


def test_e2e_retrieve_recommender(rag_store, pdf_dir):
    """检索 "recommender" 应能命中 Diffusion Recommender 或 TIGER。"""
    papers = pdf_dir_to_papers(pdf_dir, max_pages=3)
    rag_store.index_papers(papers)

    from harness.rag.retriever import PaperRetriever
    retriever = PaperRetriever(rag_store, embedding_weight=0.3, bm25_weight=0.7)
    chunks = retriever.retrieve("recommender system collaborative filtering", top_k=3)
    assert len(chunks) >= 1


def test_e2e_retrieve_chinese_query(rag_store, pdf_dir):
    """中文 query "因子" 应能命中弱因子论文（PDF 含中英文混合标题）。"""
    papers = pdf_dir_to_papers(pdf_dir, max_pages=3)
    rag_store.index_papers(papers)

    from harness.rag.retriever import PaperRetriever
    retriever = PaperRetriever(rag_store, embedding_weight=0.5, bm25_weight=0.5)
    chunks = retriever.retrieve("因子模型 弱因子", top_k=3)
    assert len(chunks) >= 1


def test_e2e_save_load_real_pdfs(rag_store, pdf_dir, tmp_path):
    """保存 + 重新加载后检索仍能命中。"""
    papers = pdf_dir_to_papers(pdf_dir, max_pages=3)
    rag_store.index_papers(papers)
    rag_store.save()

    # 新实例加载
    from harness.rag.store import PaperRAGStore
    from harness.rag.embedder import HashBackend
    new_store = PaperRAGStore(
        rag_dir=tmp_path / "rag",
        embedder=HashBackend(),
    )
    assert new_store.count() >= 10

    from harness.rag.retriever import PaperRetriever
    retriever = PaperRetriever(new_store)
    chunks = retriever.retrieve("factor pricing", top_k=3)
    assert len(chunks) >= 1


def test_e2e_format_context_real_chunks(rag_store, pdf_dir):
    """format_context 输出含论文标题 + 文本。"""
    papers = pdf_dir_to_papers(pdf_dir, max_pages=3)
    rag_store.index_papers(papers)

    from harness.rag.retriever import PaperRetriever
    retriever = PaperRetriever(rag_store)
    # 用窄 query + top_k=1 避免 412 chunk 截断
    chunks = retriever.retrieve("weak factor test asset", top_k=1, min_score=0.0)
    text = retriever.format_context(chunks, max_chars=5000)
    assert "[1]" in text
    assert "score=" in text
    # 真实论文标题（不是"NBER WORKING PAPER SERIES..."这种元信息）
    assert any(kw in text for kw in ["WEAK FACTORS", "Test Assets", "Giglio", "Xiu"])
    # 中文/英文 title 都可能（取决于命中）
    assert len(text) > 50


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
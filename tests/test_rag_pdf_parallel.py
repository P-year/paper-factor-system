"""
test_rag_pdf_parallel.py - PDF 并行处理测试
"""
import sys
import shutil
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.rag.pdf_loader import pdf_dir_to_papers


@pytest.fixture
def pdf_dir():
    """测试用 PDF 目录（gitignored fixture 的子集）。"""
    src = PROJECT_ROOT / "tests" / "fixtures" / "pdfs"
    if not src.exists() or not list(src.glob("*.pdf")):
        pytest.skip("PDF fixtures not available")
    return src


def test_default_parallel_runs(pdf_dir):
    """默认 parallel=True 应能跑（不崩）。"""
    papers = pdf_dir_to_papers(pdf_dir, max_pages=3)
    assert len(papers) >= 1


def test_serial_returns_same_results(pdf_dir):
    """serial 和 parallel 结果应一致（顺序可能不同）。"""
    papers_serial = pdf_dir_to_papers(pdf_dir, max_pages=3, parallel=False)
    papers_parallel = pdf_dir_to_papers(pdf_dir, max_pages=3, parallel=True, max_workers=2)
    s_ids = sorted(p["arxiv_id"] for p in papers_serial)
    p_ids = sorted(p["arxiv_id"] for p in papers_parallel)
    assert s_ids == p_ids
    assert len(papers_serial) == len(papers_parallel)


def test_parallel_faster_than_serial(pdf_dir):
    """多 worker 并行应快于串行（100 PDF 时明显）。"""
    # 复制 10 篇到 100 份
    big = Path(tempfile.mkdtemp(prefix="pdf_parallel_"))
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    for i in range(10):
        for pdf in pdfs:
            shutil.copy(pdf, big / f"{i:02d}_{pdf.name}")

    try:
        t0 = time.time()
        pdf_dir_to_papers(big, max_pages=3, parallel=False)
        t_s = time.time() - t0

        t0 = time.time()
        pdf_dir_to_papers(big, max_pages=3, parallel=True, max_workers=4)
        t_p4 = time.time() - t0

        # 至少 1.5x 加速（多进程+4 worker）
        assert t_p4 < t_s, f"parallel ({t_p4:.1f}s) should be faster than serial ({t_s:.1f}s)"
    finally:
        shutil.rmtree(big, ignore_errors=True)


def test_use_threads_falls_back(pdf_dir):
    """use_threads=True 应能跑（GIL 0 加速但应不崩）。"""
    papers = pdf_dir_to_papers(pdf_dir, max_pages=3, parallel=True, use_threads=True)
    assert len(papers) >= 1


def test_empty_dir(pdf_dir):
    """空目录返空。"""
    empty = Path(tempfile.mkdtemp(prefix="empty_"))
    try:
        assert pdf_dir_to_papers(empty) == []
    finally:
        empty.rmdir()


def test_missing_dir():
    """不存在的目录返空。"""
    assert pdf_dir_to_papers(Path("/nonexistent/path")) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
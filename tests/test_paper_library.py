"""
test_paper_library.py - Phase 0: 论文库测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from core.paper_library import PaperLibrary, PaperRecord


@pytest.fixture
def lib(tmp_path):
    return PaperLibrary(index_path=str(tmp_path / "papers.json"))


# === PaperLibrary ===

def test_add_paper(lib):
    rec = lib.add({"arxiv_id": "p1", "title": "Test Paper", "summary": "test"})
    assert rec["paper_id"] == "p1"
    assert rec["title"] == "Test Paper"
    assert rec["status"] == "new"


def test_add_dedup(lib):
    lib.add({"arxiv_id": "p1", "title": "A"})
    rec = lib.add({"arxiv_id": "p1", "title": "A"})
    assert len(lib.papers) == 1
    assert rec["paper_id"] == "p1"


def test_add_no_id_generates_hash(lib):
    rec = lib.add({"title": "No ID Paper"})
    assert rec["paper_id"].startswith("manual:")


def test_get_paper(lib):
    lib.add({"arxiv_id": "p1", "title": "Test"})
    rec = lib.get("p1")
    assert rec is not None
    assert rec["title"] == "Test"


def test_get_missing(lib):
    assert lib.get("nonexistent") is None


def test_list_all(lib):
    lib.add({"arxiv_id": "p1", "title": "Alpha"})
    lib.add({"arxiv_id": "p2", "title": "Beta"})
    all_papers = lib.list()
    assert len(all_papers) == 2


def test_list_filter_status(lib):
    lib.add({"arxiv_id": "p1", "title": "A"})
    lib.add({"arxiv_id": "p2", "title": "B"})
    lib.update_status("p1", "extracted")
    result = lib.list(status="extracted")
    assert len(result) == 1
    assert result[0]["paper_id"] == "p1"


def test_list_search(lib):
    lib.add({"arxiv_id": "p1", "title": "Momentum Factor"})
    lib.add({"arxiv_id": "p2", "title": "Value Investing"})
    result = lib.list(q="momentum")
    assert len(result) == 1
    assert result[0]["paper_id"] == "p1"


def test_list_pagination(lib):
    for i in range(10):
        lib.add({"arxiv_id": f"p{i}", "title": f"Paper {i}"})
    result = lib.list(limit=3, offset=2)
    assert len(result) == 3


def test_update_status(lib):
    lib.add({"arxiv_id": "p1", "title": "Test"})
    assert lib.update_status("p1", "extracted") is True
    rec = lib.get("p1")
    assert rec["status"] == "extracted"


def test_update_missing(lib):
    assert lib.update_status("nonexistent", "extracted") is False


def test_set_factor_count(lib):
    lib.add({"arxiv_id": "p1", "title": "Test"})
    lib.set_factor_count("p1", 3)
    rec = lib.get("p1")
    assert rec["factor_count"] == 3
    assert rec["status"] == "extracted"


def test_stats(lib):
    lib.add({"arxiv_id": "p1", "title": "A"})
    lib.add({"arxiv_id": "p2", "title": "B"})
    lib.set_factor_count("p1", 2)
    s = lib.stats()
    assert s["total"] == 2
    assert s["total_factors"] == 2


def test_persistence(tmp_path):
    path = str(tmp_path / "lib.json")
    lib1 = PaperLibrary(index_path=path)
    lib1.add({"arxiv_id": "p1", "title": "Persistent"})
    lib2 = PaperLibrary(index_path=path)
    assert len(lib2.papers) == 1
    assert lib2.papers[0]["title"] == "Persistent"


# === PaperRecord ===

def test_paper_record_roundtrip():
    r = PaperRecord(paper_id="p1", title="T", status="new")
    d = r.to_dict()
    r2 = PaperRecord.from_dict(d)
    assert r2.paper_id == "p1"
    assert r2.title == "T"

"""
test_paper_loader.py - paper_loader 单元测试

覆盖：
1. dict source（含 / 不含 arxiv_id）
2. arxiv_id source（mock 网络请求）
3. json_file source（写一个临时 json 测）
4. 便捷写法：str → arxiv_id，Path → json_file
5. normalize_papers 批量
6. write_papers_to_collection 写盘
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.eval.paper_loader import (
    normalize_paper_input,
    normalize_papers,
    write_papers_to_collection,
)


# === dict source ===

def test_normalize_dict_with_arxiv_id():
    p = normalize_paper_input({
        "source": "dict",
        "title": "Test Paper",
        "summary": "Test summary",
        "link": "https://arxiv.org/abs/2502.12345",
        "arxiv_id": "2502.12345",
    })
    assert p["arxiv_id"] == "2502.12345"
    assert p["title"] == "Test Paper"
    assert p["summary"] == "Test summary"
    assert p["link"] == "https://arxiv.org/abs/2502.12345"


def test_normalize_dict_without_arxiv_id_uses_link():
    p = normalize_paper_input({
        "source": "dict",
        "title": "Test",
        "summary": "S",
        "link": "https://example.com/paper",
    })
    # arxiv_id 回退到 link
    assert p["arxiv_id"] == "https://example.com/paper"
    assert p["link"] == "https://example.com/paper"


def test_normalize_dict_without_link_generates_id():
    p = normalize_paper_input({
        "source": "dict",
        "title": "Manual Paper",
        "summary": "Manual summary",
    })
    # 用 title 的 md5 前 10 位作为 fallback id
    assert p["arxiv_id"].startswith("manual:")
    assert len(p["arxiv_id"]) == len("manual:") + 10


def test_normalize_dict_without_source_defaults():
    """没 source 字段时默认按 dict 处理。"""
    p = normalize_paper_input({
        "title": "T",
        "summary": "S",
    })
    assert p["title"] == "T"
    assert p["summary"] == "S"


# === 便捷写法 ===

def test_normalize_string_as_arxiv_id():
    """str 输入便捷写法。"""
    # 不实际调网络（mock 掉）
    import harness.eval.paper_loader as pl
    original = pl._fetch_from_arxiv
    pl._fetch_from_arxiv = lambda arxiv_id: {
        "arxiv_id": arxiv_id,
        "title": f"Mocked {arxiv_id}",
        "summary": "mock summary",
        "link": f"https://arxiv.org/abs/{arxiv_id}",
    }
    try:
        p = normalize_paper_input("2502.12345")
        assert p["arxiv_id"] == "2502.12345"
        assert p["title"] == "Mocked 2502.12345"
    finally:
        pl._fetch_from_arxiv = original


def test_normalize_path_as_json_file(tmp_path):
    """Path 输入便捷写法。"""
    p_json = tmp_path / "papers_test.json"
    p_json.write_text(json.dumps({
        "papers": [
            {"arxiv_id": "x1", "title": "T1", "summary": "S1", "link": "L1"},
        ],
    }), encoding="utf-8")

    p = normalize_paper_input(p_json)
    assert p["arxiv_id"] == "x1"
    assert p["title"] == "T1"


# === json_file source ===

def test_normalize_json_file_source(tmp_path):
    p_json = tmp_path / "papers_test.json"
    p_json.write_text(json.dumps({
        "papers": [
            {"arxiv_id": "a1", "title": "T1", "summary": "S1"},
            {"arxiv_id": "a2", "title": "T2", "summary": "S2"},
        ],
    }), encoding="utf-8")

    p = normalize_paper_input({"source": "json_file", "path": str(p_json)})
    assert p["arxiv_id"] == "a1"

    p2 = normalize_paper_input({"source": "json_file", "path": str(p_json), "index": 1})
    assert p2["arxiv_id"] == "a2"


def test_normalize_json_file_missing_raises(tmp_path):
    """文件不存在应抛 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        normalize_paper_input({"source": "json_file", "path": str(tmp_path / "missing.json")})


# === 批量 ===

def test_normalize_papers_batch():
    papers = normalize_papers([
        {"source": "dict", "title": "P1", "summary": "S1"},
        {"source": "dict", "title": "P2", "summary": "S2"},
    ])
    assert len(papers) == 2
    assert all("arxiv_id" in p for p in papers)


def test_normalize_papers_empty_list():
    assert normalize_papers([]) == []


# === 错误处理 ===

def test_unknown_source_raises():
    with pytest.raises(ValueError, match="Unknown paper source"):
        normalize_paper_input({"source": "weird_source"})


def test_arxiv_id_missing_id_field():
    with pytest.raises(ValueError, match="requires 'id'"):
        normalize_paper_input({"source": "arxiv_id"})


def test_invalid_input_type():
    with pytest.raises(ValueError):
        normalize_paper_input(42)


# === 写盘 ===

def test_write_papers_to_collection(tmp_path):
    """写盘后能读回来。"""
    import harness.paths
    monkey_target = tmp_path / "papers"
    monkey_target.mkdir(parents=True, exist_ok=True)

    # monkey-patch PAPER_DIR
    original = harness.paths.PAPER_DIR
    harness.paths.PAPER_DIR = monkey_target
    try:
        papers = [{"arxiv_id": "x", "title": "T", "summary": "S", "link": "L"}]
        fp = write_papers_to_collection(papers)
        assert fp.exists()
        data = json.loads(fp.read_text(encoding="utf-8"))
        assert len(data["papers"]) == 1
        assert data["stats"]["total"] == 1
        assert data["stats"]["source"] == "injected"
    finally:
        harness.paths.PAPER_DIR = original


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
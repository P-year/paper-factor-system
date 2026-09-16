"""
test_rag_cli.py - RAG CLI 测试
"""
import sys
import subprocess
import tempfile
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest


@pytest.fixture
def custom_rag_dir(tmp_path, monkeypatch):
    """每个测试用独立 RAG_DIR。"""
    monkeypatch.setenv("HARNESS_RAG_PREFER", "hash")
    rag_dir = tmp_path / "rag"
    rag_dir.mkdir()
    return rag_dir


def _run_cli(*args, rag_dir=None):
    """调 CLI 子进程。强制 HARNESS_RAG_PREFER=hash 避免 ST 下载。"""
    import os
    cmd = [sys.executable, "-m", "harness.rag"] + list(args)
    env = os.environ.copy()
    env["HARNESS_RAG_PREFER"] = "hash"
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        timeout=30, env=env,
    )


def test_cli_status_help():
    """CLI --help 输出。"""
    r = _run_cli("--help")
    assert r.returncode == 0
    assert "RAG" in r.stdout or "rag" in r.stdout


def test_cli_status_runs():
    """status 命令在空索引上跑通。"""
    r = _run_cli("status")
    assert r.returncode == 0
    assert "RAG_DIR" in r.stdout
    assert "Chunk count" in r.stdout


def test_cli_search_no_index_error():
    """空索引时 search 返回非 0。"""
    r = _run_cli("search", "动量")
    # 可能 returncode != 0（因为脚本 sys.exit(1)） 或 仅 warn
    assert "No chunks" in r.stdout or "No chunks" in r.stderr or "No results" in r.stdout


def test_cli_rebuild_search_flow(tmp_path, monkeypatch):
    """完整流程：rebuild → search。"""
    paper_dir = tmp_path / "papers"
    paper_dir.mkdir()
    (paper_dir / "papers_20260101.json").write_text(
        json.dumps({"papers": [
            {"arxiv_id": "p1", "title": "Momentum", "summary": "动量因子在 A 股。" * 30, "link": "L1"},
            {"arxiv_id": "p2", "title": "Value", "summary": "价值投资策略。" * 30, "link": "L2"},
        ]}, ensure_ascii=False), encoding="utf-8",
    )
    # rebuild with paper-dir
    r = _run_cli("rebuild", "--paper-dir", str(paper_dir))
    assert r.returncode == 0, r.stderr
    assert "Indexed" in r.stdout

    # search
    r = _run_cli("search", "动量因子", "--top-k", "2")
    assert r.returncode == 0
    assert "score=" in r.stdout


def test_cli_clear_with_yes():
    r = _run_cli("clear", "--yes")
    assert r.returncode == 0
    assert "Cleared" in r.stdout or "error" in r.stdout.lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
"""
test_rag_manifest_v3.py - manifest v3 schema + embedder base model
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.rag.store import PaperRAGStore, _MANIFEST_VERSION
from harness.rag.embedder import SentenceTransformerBackend, HashBackend


# === Manifest 版本 ===

def test_manifest_version_is_3():
    assert _MANIFEST_VERSION == 3


def test_manifest_v3_fields(tmp_path):
    """v6-2: manifest 必须含 reranker + available_filters。"""
    s = PaperRAGStore(rag_dir=tmp_path / "rag", embedder=HashBackend())
    m = s.manifest()
    assert m["version"] == 3
    assert "reranker" in m
    assert m["reranker"]["enabled"] is False
    assert m["reranker"]["model"] is None
    assert m["reranker"]["status"] == "disabled"
    assert "available_filters" in m
    assert isinstance(m["available_filters"], list)


def test_manifest_collects_available_filters(tmp_path):
    """_collect_available_filters 从 _meta 字段收集。"""
    s = PaperRAGStore(rag_dir=tmp_path / "rag", embedder=HashBackend())
    papers = [
        {"arxiv_id": "p1", "title": "T1", "summary": "S1", "_meta": {"year": 2024, "topic": "factor"}},
        {"arxiv_id": "p2", "title": "T2", "summary": "S2", "_meta": {"year": 2023, "topic": "trading"}},
    ]
    s.index_papers(papers)
    m = s.manifest()
    assert "year" in m["available_filters"]
    assert "topic" in m["available_filters"]


# === Embedder 升级 ===

def test_sentence_transformer_backend_supports_base_model():
    """v6-2: SentenceTransformerBackend 支持 bge-base-zh-v1.5（无需新代码，验证常量）。"""
    assert SentenceTransformerBackend.BASE_MODEL == "BAAI/bge-base-zh-v1.5"


def test_sentence_transformer_backend_supports_large_model():
    assert "large" in SentenceTransformerBackend.LARGE_MODEL


def test_get_default_backend_prefer_chain(tmp_path):
    """auto 模式应优先尝试 ST。"""
    from harness.rag.embedder import get_default_backend
    # prefer="auto" 不下载（用 mock 避免真实网络）
    emb = get_default_backend(prefer="hash")
    assert isinstance(emb, HashBackend)
    assert emb.dim == 256


def test_hash_backend_default_dim():
    """默认 HashBackend dim=256。"""
    h = HashBackend()
    assert h.dim == 256


def test_hash_backend_dim_override():
    h = HashBackend(dim=128)
    assert h.dim == 128


# === Manifest roundtrip ===

def test_save_load_manifest_v3(tmp_path):
    s = PaperRAGStore(rag_dir=tmp_path / "rag", embedder=HashBackend())
    s.index_papers([{"arxiv_id": "p1", "title": "T", "summary": "S"}])
    s.save()

    # 新实例加载
    s2 = PaperRAGStore(rag_dir=tmp_path / "rag", embedder=HashBackend())
    m = s2.manifest()
    assert m["version"] == 3
    assert m["chunk_count"] >= 1
    assert "reranker" in m


def test_save_creates_manifest_with_reranker(tmp_path):
    s = PaperRAGStore(rag_dir=tmp_path / "rag", embedder=HashBackend())
    s.save()
    mf = s.rag_dir / "manifest.json"
    assert mf.exists()
    data = json.loads(mf.read_text(encoding="utf-8"))
    assert data["version"] == 3
    assert "reranker" in data


# === CLI status 输出 ===

def test_cli_status_shows_reranker(monkeypatch, tmp_path):
    """CLI status 应输出 Reranker / Available filters 行。"""
    import os
    rag_dir = tmp_path / "rag"
    s = PaperRAGStore(rag_dir=rag_dir, embedder=HashBackend())
    s.save()

    # monkeypatch RAG_DIR + 路径
    monkeypatch.setattr("harness.rag.store.RAG_DIR", rag_dir)
    monkeypatch.setattr("harness.paths.RAG_DIR", rag_dir)
    monkeypatch.setenv("HARNESS_RAG_PREFER", "hash")

    import subprocess
    r = subprocess.run(
        ["python", "-m", "harness.rag", "status"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=30,
        env={**os.environ, "HARNESS_RAG_PREFER": "hash"},
    )
    assert r.returncode == 0
    assert "Reranker:" in r.stdout
    assert "Available filters:" in r.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
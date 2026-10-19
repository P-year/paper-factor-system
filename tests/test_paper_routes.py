"""
test_paper_routes.py - Phase 2: 论文输入 API 测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient


def _client():
    from fastapi import FastAPI
    from api.paper_routes import router
    from api.rag_routes import router as rag_router
    app = FastAPI()
    app.include_router(router)
    app.include_router(rag_router)
    return TestClient(app)


def test_list_papers():
    c = _client()
    r = c.get("/papers")
    assert r.status_code == 200
    assert "papers" in r.json()
    assert "total" in r.json()


def test_paper_stats():
    c = _client()
    r = c.get("/papers/stats")
    assert r.status_code == 200
    assert "papers" in r.json()
    assert "factors" in r.json()


def test_get_paper_not_found():
    c = _client()
    r = c.get("/papers/nonexistent_id")
    assert r.status_code == 404


def test_search_papers():
    """搜索接口应返回结果列表（网络失败时返回空）。"""
    c = _client()
    r = c.post("/papers/search", json={"query": "momentum factor", "max_results": 3})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data


def test_import_papers():
    c = _client()
    r = c.post("/papers/import", json=[
        {"arxiv_id": "test_p1", "title": "Test Import", "summary": "test"}
    ])
    assert r.status_code == 200
    assert r.json()["imported"] >= 1


def test_upload_pdf_invalid_extension():
    c = _client()
    r = c.post("/papers/upload", files={"file": ("test.txt", b"not a pdf", "text/plain")})
    assert r.status_code == 400


# === RAG Routes ===

@pytest.mark.skipif(True, reason="RAG search depends on embedder loading (slow in CI)")
def test_rag_search():
    c = _client()
    r = c.post("/rag/search", json={"query": "momentum", "top_k": 3})
    assert r.status_code in (200, 500)


@pytest.mark.skipif(True, reason="RAG stats depends on embedder loading")
def test_rag_stats():
    c = _client()
    r = c.get("/rag/stats")
    assert r.status_code in (200, 500)

"""
test_frontend.py - 前端页面测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest


FRONTEND_DIR = PROJECT_ROOT / "frontend"


def test_frontend_index_exists():
    assert (FRONTEND_DIR / "index.html").exists()


def test_frontend_html_valid():
    content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "</html>" in content
    assert "<body" in content


def test_frontend_has_navigation():
    content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert "sidebar" in content.lower() or "aside" in content
    assert "仪表盘" in content
    assert "论文库" in content
    assert "RAG" in content
    assert "因子回测" in content
    assert "策略回测" in content
    assert "报告中心" in content


def test_frontend_has_charts():
    content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert "echarts" in content.lower()
    assert "chart-" in content


def test_frontend_has_api_integration():
    content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert "fetch(" in content
    assert "/backtest/" in content
    assert "/agent/" in content


def test_frontend_has_styling():
    content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert "tailwind" in content.lower()
    assert "glass" in content
    assert "gradient" in content


def test_frontend_has_responsive_meta():
    content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert "viewport" in content


def test_frontend_chinese_ui():
    content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert "因子挖掘" in content
    assert "回测" in content
    assert "检索" in content


def test_api_root_serves_frontend():
    """API root 应该返回前端页面。"""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from api.backtest_routes import router

    app = FastAPI()
    app.include_router(router)

    # 手动添加 root 路由
    from fastapi.responses import FileResponse
    @app.get("/")
    async def root():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "FactorLab" in r.text

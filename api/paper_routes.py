"""
api/paper_routes.py - 论文输入 API

POST /papers/upload    — PDF 上传
POST /papers/search    — arXiv 搜索
GET  /papers           — 论文列表
GET  /papers/stats     — 统计
GET  /papers/{id}      — 论文详情
POST /papers/{id}/extract — 提取因子
POST /papers/{id}/run     — 一键回测（产品闭环）
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

router = APIRouter(prefix="/papers", tags=["papers"])


class SearchRequest(BaseModel):
    query: str
    max_results: int = 10
    days: int = 0


class ExtractRequest(BaseModel):
    use_rag: bool = False


class RunRequest(BaseModel):
    provider: str = "synthetic"
    universe: str = "csi500"
    start_date: str = "20230101"
    end_date: str = "20241231"
    n_stocks: int = 50
    use_rag: bool = True
    writeback: bool = True


@router.get("")
async def list_papers(status: str = "", q: str = "", limit: int = 50, offset: int = 0):
    from core.paper_library import PaperLibrary
    lib = PaperLibrary()
    papers = lib.list(status=status, q=q, limit=limit, offset=offset)
    return {"papers": papers, "total": len(lib.papers)}


@router.get("/stats")
async def paper_stats():
    from core.paper_library import PaperLibrary
    from core.factor_db import FactorDatabase
    lib = PaperLibrary()
    db = FactorDatabase()
    return {
        "papers": lib.stats(),
        "factors": db.get_product_summary(),
    }


@router.get("/{paper_id}")
async def get_paper(paper_id: str):
    from core.paper_library import PaperLibrary
    from core.factor_db import FactorDatabase
    lib = PaperLibrary()
    paper = lib.get(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="paper not found")
    db = FactorDatabase()
    factors = db.get_by_paper(paper_id)
    return {"paper": paper, "factors": factors}


@router.post("/upload")
async def upload_paper(file: UploadFile = File(...), title: str = Form("")):
    """上传 PDF 论文。"""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only PDF files accepted")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="file too large (max 20MB)")

    # 保存到 data/papers/uploads/
    upload_dir = Path("data/papers/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    pid = "manual:" + hashlib.md5(content).hexdigest()[:16]
    pdf_path = upload_dir / f"{pid}.pdf"
    pdf_path.write_bytes(content)

    # 提取 PDF 文本
    try:
        from harness.rag.pdf_loader import pdf_to_paper_dict
        paper = pdf_to_paper_dict(pdf_path, max_pages=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {e}")

    if title:
        paper["title"] = title

    # 加入论文库
    from core.paper_library import PaperLibrary
    lib = PaperLibrary()
    record = lib.add(paper, source="pdf_upload", filename=file.filename)

    return {"paper": record, "message": "uploaded and parsed"}


@router.post("/search")
async def search_papers(req: SearchRequest):
    """arXiv 搜索论文（不持久化，返回结果供选择导入）。"""
    try:
        from core.collector import PaperCollector
        collector = PaperCollector()
        results = collector.search_arxiv(req.query, max_results=req.max_results, days=req.days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed: {e}")
    return {"results": results, "query": req.query}


@router.post("/import")
async def import_papers(papers: List[Dict]):
    """导入论文列表到论文库。"""
    from core.paper_library import PaperLibrary
    lib = PaperLibrary()
    added = []
    for p in papers:
        rec = lib.add(p, source=p.get("source", "arxiv"))
        added.append(rec)
    return {"imported": len(added), "papers": added}


@router.post("/{paper_id}/extract")
async def extract_factors(paper_id: str, req: ExtractRequest = ExtractRequest()):
    """从论文提取因子。"""
    from core.paper_library import PaperLibrary
    from core.factor_db import FactorDatabase
    lib = PaperLibrary()
    paper = lib.get(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="paper not found")

    try:
        from core.analyzer import FactorAnalyzer
        analyzer = FactorAnalyzer()
        result = analyzer.extract_factor(paper)
        if not result.is_factor:
            return {"extracted": 0, "message": "no factors found in paper"}

        db = FactorDatabase()
        factor_data = result.factor_data
        factor_data["paper_id"] = paper_id
        db.add_factor(factor_data)

        n = len(db.get_by_paper(paper_id))
        lib.set_factor_count(paper_id, n)
        return {"extracted": 1, "factor_name": factor_data.get("factor_name", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"extraction failed: {e}")


@router.post("/{paper_id}/run")
async def run_paper(paper_id: str, req: RunRequest = RunRequest()):
    """一键产品闭环: 提取 → 编译 → 回测 → 写回。"""
    try:
        from core.product_loop import run_paper_to_decision
        result = run_paper_to_decision(
            paper_id=paper_id,
            provider=req.provider,
            universe=req.universe,
            start_date=req.start_date,
            end_date=req.end_date,
            n_stocks=req.n_stocks,
            writeback=req.writeback,
        )
        return {
            "status": result.status,
            "paper_id": result.paper_id,
            "factors": len(result.factors),
            "backtests": len(result.backtests),
            "run_ids": result.run_ids,
            "errors": result.errors,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pipeline failed: {e}")

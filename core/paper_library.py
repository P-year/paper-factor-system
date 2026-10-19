"""
core/paper_library.py - 论文库统一管理

替代散落的 data/papers/papers_*.json，提供统一的论文索引。
支持 PDF 上传、arXiv 搜索、本地导入。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PaperRecord:
    """论文记录。"""
    paper_id: str = ""
    title: str = ""
    summary: str = ""
    link: str = ""
    source: str = "arxiv"        # arxiv | pdf_upload | local | injected
    filename: str = ""
    status: str = "new"           # new | indexed | extracted | failed
    factor_count: int = 0
    added_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: Dict) -> "PaperRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PaperLibrary:
    """论文库索引（JSON 持久化）。"""

    def __init__(self, index_path: Optional[str] = None):
        if index_path:
            self.index_path = Path(index_path)
        else:
            from config import DATA_DIR
            self.index_path = DATA_DIR / "paper_library.json"
        self.papers: List[Dict] = []
        self.load()

    def load(self):
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                self.papers = data.get("papers", [])
            except Exception:
                self.papers = []

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps({"papers": self.papers, "updated_at": datetime.now().isoformat()},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, paper: Dict, *, source: str = "arxiv", filename: str = "") -> Dict:
        """添加论文（去重）。返回 PaperRecord dict。"""
        pid = paper.get("arxiv_id") or paper.get("paper_id") or self._hash_id(paper.get("title", ""))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 去重
        for p in self.papers:
            if p.get("paper_id") == pid:
                return p

        record = {
            "paper_id": pid,
            "title": paper.get("title", ""),
            "summary": (paper.get("summary") or paper.get("abstract", ""))[:2000],
            "link": paper.get("link", ""),
            "source": source,
            "filename": filename or paper.get("filename", ""),
            "status": "new",
            "factor_count": 0,
            "added_at": now,
            "updated_at": now,
        }
        self.papers.append(record)
        self.save()
        return record

    def import_legacy_papers(self) -> int:
        """一次性导入 data/papers/papers_*.json 中的论文。"""
        from config import PAPER_DIR
        count = 0
        paper_dir = Path(PAPER_DIR)
        if not paper_dir.exists():
            return 0
        for fp in sorted(paper_dir.glob("papers_*.json")):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            papers = data.get("papers", data) if isinstance(data, dict) else data
            if not papers or not isinstance(papers, list):
                continue
            for p in papers:
                if isinstance(p, dict) and p.get("title"):
                    self.add(p, source="arxiv")
                    count += 1
        return count

    def get(self, paper_id: str) -> Optional[Dict]:
        for p in self.papers:
            if p.get("paper_id") == paper_id:
                return p
        return None

    def list(self, *, status: str = "", q: str = "", limit: int = 50, offset: int = 0) -> List[Dict]:
        result = self.papers
        if status:
            result = [p for p in result if p.get("status") == status]
        if q:
            ql = q.lower()
            result = [p for p in result if ql in (p.get("title", "") + p.get("summary", "")).lower()]
        return result[offset:offset + limit]

    def update_status(self, paper_id: str, status: str, **extra) -> bool:
        for p in self.papers:
            if p.get("paper_id") == paper_id:
                p["status"] = status
                p["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                p.update(extra)
                self.save()
                return True
        return False

    def set_factor_count(self, paper_id: str, n: int) -> bool:
        return self.update_status(paper_id, "extracted", factor_count=n)

    def stats(self) -> Dict:
        status_counts = {}
        source_counts = {}
        total_factors = 0
        for p in self.papers:
            s = p.get("status", "new")
            status_counts[s] = status_counts.get(s, 0) + 1
            src = p.get("source", "arxiv")
            source_counts[src] = source_counts.get(src, 0) + 1
            total_factors += p.get("factor_count", 0)
        return {
            "total": len(self.papers),
            "status": status_counts,
            "by_source": source_counts,
            "total_factors": total_factors,
        }

    @staticmethod
    def _hash_id(text: str) -> str:
        return "manual:" + hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

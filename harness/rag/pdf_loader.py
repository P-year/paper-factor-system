"""
harness/rag/pdf_loader.py - PDF 文本提取（基于 pdfplumber）

设计：
- 限制前 N 页（默认 5，避免 PDF太长，references/acknowledgments 不在因子讨论范围）
- 按页提取，每页构造一个 section marker（如 "## Page 3"）
- 返回标准 paper dict（含 title + summary + arxiv_id + link）

用法：
    from harness.rag.pdf_loader import pdf_to_paper_dict
    paper = pdf_to_paper_dict(Path("paper.pdf"), max_pages=5)
    # paper = {"arxiv_id": "manual:md5...", "title": "...", "summary": "## Page 1\n...", "link": "..."}
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    """规范化 PDF 文本：合并空白行、去除 control chars。"""
    if not text:
        return ""
    # 合并多空白为单空格
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _pdf_id(path: Path) -> str:
    """从 PDF 路径构造稳定 source_id（无 arxiv_id 时用）。"""
    return hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _first_meaningful_line(pages_text: List[str]) -> str:
    """从首页提取首句非空内容作为 title（避免取到 "NBER WORKING PAPER SERIES" 等元信息）。"""
    if not pages_text:
        return ""
    first = pages_text[0]
    # 跳过常见的元信息行
    skip_patterns = [
        r"^NBER\b", r"^Working Paper", r"^Working paper", r"^arXiv:",
        r"^Submitted", r"^Preprint", r"^JEL No\.",
        r"^\d+$",  # 页码
        r"^©",
    ]
    skip_re = re.compile("|".join(skip_patterns), re.IGNORECASE)
    lines = first.split("\n")
    for line in lines:
        line = line.strip()
        if not line or len(line) < 5 or len(line) > 200:
            continue
        if skip_re.match(line):
            continue
        return line
    return lines[0].strip() if lines else ""


def extract_pdf_text(
    pdf_path: Path,
    *,
    max_pages: int = 5,
) -> List[Dict[str, Any]]:
    """提取 PDF 文本（按页）。

    Returns:
        List of {"page_num": int, "text": str, "title_guess": str}
        title_guess 只在 page 1 填，后续页为空字符串。

    Raises:
        FileNotFoundError: PDF 不存在
        ImportError: pdfplumber 未装
        Exception: pdfplumber 解析失败（密码 / 损坏）
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF extraction. Install with: pip install pdfplumber"
        )

    pages: List[Dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for i in range(min(max_pages, total_pages)):
            page = pdf.pages[i]
            text = page.extract_text() or ""
            text = _clean_text(text)
            pages.append({
                "page_num": i + 1,
                "text": text,
                "title_guess": _first_meaningful_line([text]) if i == 0 else "",
            })
    return pages


def pdf_to_paper_dict(
    pdf_path: Path,
    *,
    max_pages: int = 5,
    paper_id: Optional[str] = None,
) -> Dict[str, Any]:
    """把 PDF 转成标准 paper dict（与 harness.eval.paper_loader 输出兼容）。

    summary 字段拼接多页文本（用 '## Page N' 标记分页），
    让 chunker 按页 boundary 切分。
    """
    pdf_path = Path(pdf_path)
    pages = extract_pdf_text(pdf_path, max_pages=max_pages)

    # 拼接
    parts: List[str] = []
    title_guess = ""
    for p in pages:
        title_guess = title_guess or p["title_guess"]
        if p["text"]:
            parts.append(f"## Page {p['page_num']}\n{p['text']}")
    summary = "\n\n".join(parts)

    sid = paper_id or _pdf_id(pdf_path)

    # link 用文件名
    return {
        "arxiv_id": sid,
        "title": title_guess or pdf_path.stem,
        "summary": summary,
        "link": f"file://{pdf_path.resolve().as_posix()}",
        "published": "",
        "authors": "",
        "categories": "",
        "pdf_link": f"file://{pdf_path.resolve().as_posix()}",
        "source": "local_pdf",
        "filename": pdf_path.name,
    }


def pdf_dir_to_papers(
    pdf_dir: Path,
    *,
    max_pages: int = 5,
    pattern: str = "*.pdf",
) -> List[Dict[str, Any]]:
    """扫描 pdf_dir 把所有 PDF 转成 paper dicts。"""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for fp in sorted(pdf_dir.glob(pattern)):
        try:
            out.append(pdf_to_paper_dict(fp, max_pages=max_pages))
        except Exception as e:
            # 单 PDF 失败不阻塞
            import warnings
            warnings.warn(f"failed to extract {fp.name}: {e}")
            continue
    return out
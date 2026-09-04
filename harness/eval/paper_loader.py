"""
harness/eval/paper_loader.py - 论文输入标准化

把各种形式的 paper 输入（dict / arxiv_id / json_file）转换成标准 paper dict。

支持的 source：
    dict       直接用 dict 字段（title / summary / link）
    arxiv_id   从 arxiv API 抓（仅摘要，不下 PDF）
    json_file  读 data/papers/papers_*.json 格式的本地文件

输出 paper dict 必有字段：arxiv_id / title / summary / link
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_paper_input(item: Any) -> Dict[str, Any]:
    """单条 paper 输入 → 标准 paper dict。

    支持：
        - dict: {source: "dict", title, summary, link, arxiv_id?}
        - dict: {source: "arxiv_id", id: "2502.12345"}
        - dict: {source: "json_file", path: "data/papers/papers_xxx.json"}
        - str: "2502.12345"  ← 便捷写法（视为 arxiv_id）
        - Path: 本地文件路径  ← 视为 json_file
        - 已标准化的 paper dict: 直接返回
    """
    # str → arxiv_id 便捷写法
    if isinstance(item, str):
        item = {"source": "arxiv_id", "id": item}

    # Path → json_file
    if isinstance(item, Path):
        item = {"source": "json_file", "path": str(item)}

    if not isinstance(item, dict):
        raise ValueError(f"paper_input must be dict/str/Path, got {type(item).__name__}")

    source = item.get("source", "dict")

    if source == "dict":
        title = item.get("title", "")
        summary = item.get("summary", "")
        link = item.get("link", "")
        arxiv_id = item.get("arxiv_id") or link or f"manual:{_hash_title(title)}"
        return _standard_paper(arxiv_id=arxiv_id, title=title, summary=summary, link=link)

    if source == "arxiv_id":
        arxiv_id = item.get("id") or item.get("arxiv_id")
        if not arxiv_id:
            raise ValueError(f"arxiv_id source requires 'id' field: {item}")
        return _fetch_from_arxiv(arxiv_id)

    if source == "json_file":
        path = item.get("path")
        if not path:
            raise ValueError(f"json_file source requires 'path' field: {item}")
        return _load_paper_from_json(Path(path), item.get("index", 0))

    if source == "already_standard":
        # 直接是标准化 paper dict（用于复用已有 collect 结果）
        return _validate_standard(item)

    raise ValueError(f"Unknown paper source: {source!r}")


def normalize_papers(items: List[Any]) -> List[Dict[str, Any]]:
    """批量标准化。"""
    return [normalize_paper_input(i) for i in items]


def write_papers_to_collection(papers: List[Dict[str, Any]]) -> Path:
    """把 papers 写到 data/papers/papers_<ts>_injected.json。

    analyze_node 通过 PAPER_DIR 读 papers_*.json，加载时只看 arxiv_id 命中 state.collected_paper_ids。
    写盘是为了让 analyze_node 的 _load_papers() 找得到。
    """
    from harness.paths import PAPER_DIR

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]
    fp = PAPER_DIR / f"papers_{ts}_injected.json"
    fp.write_text(
        json.dumps(
            {"papers": papers, "stats": {"total": len(papers), "source": "injected"}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return fp


# === 内部工具 ===

def _standard_paper(*, arxiv_id: str, title: str, summary: str, link: str) -> Dict[str, Any]:
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "summary": summary,
        "link": link or f"https://arxiv.org/abs/{arxiv_id}",
    }


def _validate_standard(paper: Dict[str, Any]) -> Dict[str, Any]:
    """检查 paper 必含字段，补默认。"""
    return _standard_paper(
        arxiv_id=paper.get("arxiv_id") or paper.get("link") or "",
        title=paper.get("title", ""),
        summary=paper.get("summary", ""),
        link=paper.get("link", ""),
    )


def _hash_title(title: str) -> str:
    import hashlib
    return hashlib.md5((title or "untitled").encode("utf-8")).hexdigest()[:10]


def _fetch_from_arxiv(arxiv_id: str) -> Dict[str, Any]:
    """从 arxiv API 抓单篇论文摘要。"""
    import re
    import requests

    # arxiv_id 标准化（如 "2502.12345" 或 "2502.12345v1"）
    arxiv_id = arxiv_id.strip()
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    headers = {"User-Agent": "paper-factor-system/1.0 (research; +https://github.com)"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"arxiv fetch failed for {arxiv_id}: {e}")

    # 解析 atom XML
    text = resp.text
    # 简单正则提取（避免 lxml 依赖）
    title_m = re.search(r"<title>(.*?)</title>", text, re.DOTALL)
    summary_m = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
    link_m = re.search(rf'<id>http://arxiv.org/abs/{arxiv_id}.*?</id>', text, re.DOTALL)

    title = (title_m.group(1).strip() if title_m else arxiv_id).replace("\n", " ")
    summary = (summary_m.group(1).strip() if summary_m else "").replace("\n", " ")
    link = f"https://arxiv.org/abs/{arxiv_id}"

    return _standard_paper(arxiv_id=arxiv_id, title=title, summary=summary, link=link)


def _load_paper_from_json(path: Path, index: int = 0) -> Dict[str, Any]:
    """从 papers_*.json 加载第 index 篇。"""
    if not path.exists():
        raise FileNotFoundError(f"paper file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    papers = data.get("papers", [])
    if not papers:
        raise ValueError(f"no papers in {path}")
    if index >= len(papers):
        raise IndexError(f"index {index} out of range ({len(papers)} papers)")
    return _validate_standard(papers[index])
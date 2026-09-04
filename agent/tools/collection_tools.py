"""
collection_tools - 论文采集相关
复用：PaperCollector (collector.py) / LocalPaperCollector (local_collector.py)
v3-3：search_arxiv 加 arxiv_search 速率限制
"""
from typing import Dict, List, Optional

from collector import PaperCollector
from local_collector import LocalPaperCollector
from agent.paths import PAPER_DIR
from harness.ratelimit import get_default_limiter, RateLimitExceeded


def search_arxiv(
    days: int = 90,
    force: bool = False,
    keywords: Optional[List[str]] = None,
    max_results_per_keyword: int = 30,
) -> Dict:
    """
    从 arXiv 采集因子相关论文（含金融过滤 + 去重）。

    Args:
        days: 搜索近 N 天内的论文
        force: 是否强制重新采集（否则尝试加载已有数据）
        keywords: 自定义关键词列表（None=使用 config.py 默认）
        max_results_per_keyword: 每个关键词最大结果数

    Returns:
        {"papers": [...], "count": N, "from_cache": bool, "rate_limited": bool (可选)}
    """
    # v3-3：rate-limit check（不阻塞，超限直接返回错误）
    try:
        get_default_limiter().check("arxiv_search")
    except RateLimitExceeded as e:
        return {
            "papers": [],
            "count": 0,
            "error": str(e),
            "rate_limited": True,
            "retry_after": round(e.retry_after, 2),
        }

    try:
        collector = PaperCollector()

        # 检查断点续传
        if not force:
            existing_files = sorted(PAPER_DIR.glob("papers_*.json"), reverse=True)
            if existing_files:
                papers = collector.load_papers(str(existing_files[0]))
                return {"papers": papers, "count": len(papers), "from_cache": True}

        # 采集
        papers = collector.collect_factor_papers(days=days, force=force)

        # 自定义关键词（如果提供）
        if keywords:
            for kw in keywords:
                extra = collector._search_by_keyword(kw, max_results=max_results_per_keyword)
                for p in extra:
                    papers.append(p)

        papers = collector.filter_finance_papers(papers)

        if papers:
            collector.save_papers(papers)

        return {
            "papers": papers[:50],
            "count": len(papers),
            "from_cache": False,
        }
    except Exception as e:
        return {"papers": [], "count": 0, "error": str(e)}


def read_paper(paper_id: str, local_dir: Optional[str] = None) -> Dict:
    """
    读取单篇论文的完整文本（仅本地 PDF 用，arxiv ID 走 search_arxiv 拿摘要）。

    Args:
        paper_id: PDF 文件名（含或不含 .pdf）或 arxiv_id
        local_dir: 本地 PDF 目录（None=不读本地）

    Returns:
        {"text": str, "meta": {filename, n_pages}, "source": "local"|"arxiv"}
    """
    if not local_dir:
        return {"text": "", "meta": {}, "error": "no local_dir provided"}

    try:
        collector = LocalPaperCollector(local_dir)
        text, meta = collector.read_pdf(paper_id) if hasattr(collector, "read_pdf") else ("", {})
        return {"text": text[:5000], "meta": meta, "source": "local"}
    except Exception as e:
        return {"text": "", "meta": {}, "error": str(e)}
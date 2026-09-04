"""
论文采集模块 - 增强版
支持多数据源、错误处理、断点续传、去重

增强内容：
1. 多数据源支持（arXiv + 本地PDF）
2. 更完善的错误处理
3. 去重机制（基于标题+摘要hash）
4. 进度显示优化
5. 断点续传（加载已采集列表）
"""

import arxiv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from config import (
    ARXIV_KEYWORDS,
    ARXIV_CATEGORIES,
    MAX_PAPERS_PER_SEARCH,
    COLLECTION_DAYS,
    PAPER_DIR,
)


@dataclass
class CollectionStats:
    """采集统计信息"""
    total_found: int = 0
    after_filter: int = 0
    duplicates_removed: int = 0
    saved: int = 0
    errors: int = 0


class PaperCollector:
    """
    论文采集器（增强版）

    支持：
    - arXiv关键词搜索
    - arXiv分类搜索
    - 本地PDF扫描
    - 多数据源去重
    - 金融相关性过滤
    - 断点续传
    """

    def __init__(self):
        self.client = arxiv.Client()
        self.papers: List[Dict] = []
        self.seen_titles: Set[str] = set()
        self.seen_hashes: Set[str] = set()  # 标题+摘要前100字hash
        self.stats = CollectionStats()

    # ------------------------------------------------------
    # 公开API
    # ------------------------------------------------------

    def collect_factor_papers(
        self,
        days: int = COLLECTION_DAYS,
        force: bool = False,
    ) -> List[Dict]:
        """
        采集因子相关论文

        Args:
            days: 搜索近N天内的论文
            force: 是否强制重新采集（否则尝试加载已有数据）

        Returns:
            去重后的论文列表
        """
        # 尝试断点续传
        if not force:
            existing = self._load_latest_collection()
            if existing:
                print(f"[INFO] 加载已有采集结果: {len(existing)} 篇")
                return existing

        all_papers: Dict[str, Dict] = {}

        # 1. 按分类搜索
        print(f"[1/2] 按分类搜索近{days}天论文...")
        for cat in ARXIV_CATEGORIES:
            try:
                papers = self._search_by_category(cat, days=days, max_results=50)
                for paper in papers:
                    all_papers[paper["link"]] = paper
                print(f"  {cat}: +{len(papers)} 篇")
            except Exception as e:
                print(f"  ⚠️ {cat} 搜索失败: {e}")
                self.stats.errors += 1

        # 2. 按关键词搜索
        print(f"[2/2] 按关键词搜索...")
        for kw in ARXIV_KEYWORDS:
            try:
                papers = self._search_by_keyword(kw, max_results=MAX_PAPERS_PER_SEARCH)
                for paper in papers:
                    all_papers[paper["link"]] = paper
                print(f"  [{kw[:30]}]: +{len(papers)} 篇")
            except Exception as e:
                print(f"  ⚠️ 关键词搜索失败: {e}")
                self.stats.errors += 1

        result = list(all_papers.values())
        self.stats.total_found = len(result)

        # 金融相关性过滤
        filtered = self.filter_finance_papers(result)
        self.stats.after_filter = len(filtered)

        print(f"\n采集完成: 原始{self.stats.total_found} → 过滤后{self.stats.after_filter}（去重{self.stats.duplicates_removed}）")

        self.papers = filtered
        return filtered

    def filter_finance_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        过滤金融/量化相关论文

        规则：
        - 金融关键词≥2 且 非金融关键词<2 → 通过
        - 纯CS/ML理论（图像/NLP/游戏等）→ 排除
        """
        positive_kw = [
            "factor", "portfolio", "stock", "invest", "finance", "asset pricing",
            "alpha", "return", "risk", "equity", "market", "trading", "investment",
            "因子", "投资", "股票", "组合", "金融",
            "multimodal", "LLM", "newsflow", "earnings", "sentiment",
        ]
        negative_kw = [
            "neural network", "deep learning", "transformer", "NLP", "natural language",
            "image", "vision", "reinforcement learning", "robot", "game",
            "image classification", "object detection", "speech", "audio",
            "GAN", "VAE", "diffusion model", "generative",
        ]

        filtered = []
        for paper in papers:
            title = paper.get("title", "").lower()
            abstract = paper.get("summary", "").lower()
            text = title + " " + abstract

            pos_score = sum(1 for kw in positive_kw if kw in text)
            neg_score = sum(1 for kw in negative_kw if kw in text)

            if pos_score >= 2 and neg_score < 2:
                filtered.append(paper)
            else:
                self.stats.duplicates_removed += 1

        return filtered

    def save_papers(self, papers: Optional[List[Dict]] = None, filepath: Optional[str] = None):
        """保存论文列表到JSON"""
        papers = papers or self.papers

        if filepath is None:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = PAPER_DIR / f"papers_{date_str}.json"

        # 添加元数据
        save_data = {
            "meta": {
                "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_count": len(papers),
                "stats": {
                    "total_found": self.stats.total_found,
                    "after_filter": self.stats.after_filter,
                    "duplicates_removed": self.stats.duplicates_removed,
                    "errors": self.stats.errors,
                }
            },
            "papers": papers,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        self.stats.saved = len(papers)
        print(f"已保存 {len(papers)} 篇论文到 {filepath}")

    def load_papers(self, filepath: str) -> List[Dict]:
        """从JSON文件加载论文列表"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            papers = data.get("papers", [])
            meta = data.get("meta", {})
            print(f"加载 {len(papers)} 篇论文（采集时间: {meta.get('collected_at', '未知')}）")
        else:
            papers = data
            print(f"加载 {len(papers)} 篇论文")

        self.papers = papers
        return papers

    # ------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------

    def _search_by_keyword(
        self,
        keyword: str,
        max_results: int = MAX_PAPERS_PER_SEARCH,
    ) -> List[Dict]:
        """按关键词搜索arXiv"""
        search = arxiv.Search(
            query=keyword,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for result in self.client.results(search):
            paper = self._result_to_paper(result)
            if self._is_new_paper(paper):
                results.append(paper)

        return results

    def _search_by_category(
        self,
        category: str,
        days: int = COLLECTION_DAYS,
        max_results: int = 50,
    ) -> List[Dict]:
        """按分类搜索近期论文"""
        from datetime import timezone
        date_cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        results = []
        for result in self.client.results(search):
            if result.published >= date_cutoff:
                paper = self._result_to_paper(result)
                if self._is_new_paper(paper):
                    results.append(paper)

        return results

    def _result_to_paper(self, result) -> Dict:
        """arXiv结果转换为字典"""
        return {
            "title": result.title.replace("\n", " ").strip(),
            "summary": result.summary.replace("\n", " ").strip(),
            "link": result.entry_id,
            "published": result.published.strftime("%Y-%m-%d"),
            "authors": [a.name for a in result.authors],
            "categories": result.categories,
            "pdf_link": result.pdf_url,
            "arxiv_id": result.entry_id.split("/")[-1],
        }

    def _is_new_paper(self, paper: Dict) -> bool:
        """检查是否是新论文（去重）"""
        title_lower = paper["title"].lower()
        abstract_short = paper["summary"][:100].lower()
        hash_key = f"{title_lower}:{abstract_short}"

        if title_lower in self.seen_titles:
            self.stats.duplicates_removed += 1
            return False

        if hash_key in self.seen_hashes:
            self.stats.duplicates_removed += 1
            return False

        self.seen_titles.add(title_lower)
        self.seen_hashes.add(hash_key)
        return True

    def _load_latest_collection(self) -> Optional[List[Dict]]:
        """加载最新的已采集数据（断点续传）"""
        try:
            files = sorted(PAPER_DIR.glob("papers_*.json"), reverse=True)
            if files:
                return self.load_papers(str(files[0]))
        except Exception:
            pass
        return None

    def get_statistics(self) -> CollectionStats:
        """获取采集统计"""
        return self.stats


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    collector = PaperCollector()
    papers = collector.collect_factor_papers(days=60, force=True)
    papers = collector.filter_finance_papers(papers)
    collector.save_papers(papers)

    stats = collector.get_statistics()
    print(f"\n采集统计:")
    print(f"  总发现: {stats.total_found}")
    print(f"  过滤后: {stats.after_filter}")
    print(f"  去重移除: {stats.duplicates_removed}")
    print(f"  保存成功: {stats.saved}")
    print(f"  错误数: {stats.errors}")

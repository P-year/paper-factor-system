"""
本地PDF论文采集模块
从指定目录读取PDF论文并提取文本
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import pdfplumber


class LocalPaperCollector:
    """从本地目录读取PDF论文"""

    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.papers = []

    def scan_pdfs(self) -> List[str]:
        """扫描目录下的所有PDF文件"""
        pdf_files = list(self.directory.glob("*.pdf"))
        print(f"发现 {len(pdf_files)} 个PDF文件")
        return [str(f) for f in pdf_files]

    def extract_text_from_pdf(self, pdf_path: str, max_pages: int = 5) -> Dict:
        """
        从PDF提取文本（取前N页的摘要部分）

        Args:
            pdf_path: PDF文件路径
            max_pages: 最大读取页数（用于提取摘要）

        Returns:
            包含 title/summary/pages 的字典
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                title = ""

                # 提取标题（通常在前几行）
                full_text = []
                for i, page in enumerate(pdf.pages[:max_pages]):
                    text = page.extract_text() or ""
                    if text.strip():
                        full_text.append(text)

                        # 从第一页提取标题（假设在前几行）
                        if i == 0 and not title:
                            lines = [l.strip() for l in text.split("\n") if l.strip()]
                            # 跳过明显不是标题的行
                            for line in lines[:10]:
                                if len(line) > 20 and len(line) < 200:
                                    title = line
                                    break

                combined_text = "\n".join(full_text)

                # 提取摘要（寻找abstract/introduction部分）
                summary = self._extract_summary(combined_text)

                return {
                    "title": title or Path(pdf_path).stem,
                    "summary": summary or combined_text[:2000],
                    "full_text": combined_text,
                    "total_pages": total_pages,
                    "path": pdf_path,
                    "filename": Path(pdf_path).name,
                }

        except Exception as e:
            print(f"读取PDF失败 {pdf_path}: {e}")
            return {
                "title": Path(pdf_path).stem,
                "summary": "",
                "full_text": "",
                "total_pages": 0,
                "path": pdf_path,
                "error": str(e),
            }

    def _extract_summary(self, text: str) -> str:
        """从正文中提取摘要部分"""
        text_lower = text.lower()

        # 寻找摘要关键词
        markers = [
            "abstract",
            "summary",
            "overview",
            "introduction",  # 有时直接就是introduction
        ]

        for marker in markers:
            idx = text_lower.find(marker)
            if idx != -1:
                # 提取摘要后的文字
                abstract_text = text[idx + len(marker):]
                # 通常摘要到第一个双换行或固定长度结束
                paras = abstract_text.split("\n\n")
                if paras:
                    # 取第一个段落，或前500字符
                    summary = paras[0].strip()[:1500]
                    if len(summary) > 50:
                        return summary

        # 如果没找到摘要，返回前1000字符
        return text[:1500].strip()

    def collect_papers(self, max_papers: int = 10) -> List[Dict]:
        """
        采集目录下所有论文

        Args:
            max_papers: 最大处理论文数（用于测试）

        Returns:
            论文列表
        """
        pdf_files = self.scan_pdfs()
        papers = []

        for i, pdf_path in enumerate(pdf_files[:max_papers]):
            print(f"处理 [{i+1}/{min(len(pdf_files), max_papers)}]: {Path(pdf_path).name}")

            paper = self.extract_text_from_pdf(pdf_path)
            papers.append(paper)

            if (i + 1) % 5 == 0:
                print(f"  已处理 {i+1} 篇论文")

        self.papers = papers
        print(f"\n共采集 {len(papers)} 篇论文")
        return papers

    def save_papers(self, filepath: Optional[str] = None):
        """保存采集的论文到JSON"""
        if filepath is None:
            from config import PAPER_DIR
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d")
            filepath = PAPER_DIR / f"local_papers_{date_str}.json"

        # 只保存标题和摘要，不保存full_text（太大）
        save_data = []
        for paper in self.papers:
            save_data.append({
                "title": paper.get("title", ""),
                "summary": paper.get("summary", ""),
                "total_pages": paper.get("total_pages", 0),
                "path": paper.get("path", ""),
                "filename": paper.get("filename", ""),
            })

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        print(f"已保存 {len(save_data)} 篇论文到 {filepath}")

    def load_papers(self, filepath: str) -> List[Dict]:
        """从JSON加载论文"""
        with open(filepath, "r", encoding="utf-8") as f:
            self.papers = json.load(f)
        print(f"已加载 {len(self.papers)} 篇论文")
        return self.papers


if __name__ == "__main__":
    # 测试
    paper_dir = r"D:\Programming\QuantLearning\量化论文\文献\Agostino Capponi"

    collector = LocalPaperCollector(paper_dir)
    papers = collector.collect_papers(max_papers=3)

    for p in papers:
        print(f"\n标题: {p['title'][:80]}")
        print(f"摘要: {p['summary'][:200]}...")
        print("-" * 50)
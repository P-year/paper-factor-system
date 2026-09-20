"""
harness/rag/chunker.py - 论文文本切块

目的：把论文摘要切成适合 embedding 的 chunks（段落 + 滑动窗口）。

设计：
- 默认 max_chunk_chars=800（~200 tokens，中文友好）
- 默认 overlap_chars=200（保留跨 chunk 上下文）
- 段落切分优先（按句末符号 + 段落分隔）
- 单段超长走滑动窗口
- chunk_id 唯一 = source_id + 段序号
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List


# 段落分隔正则：句末标点 + 换行；或空行
_PARAGRAPH_SPLIT_RE = re.compile(
    r"(?<=[。.!?！？])\s*\n"   # 句末标点 + 换行
    r"|"
    r"\n\s*\n"               # 空行
)

# 句末标点（用于滑动窗口边界对齐）
_SENTENCE_END_RE = re.compile(r"[。.!?！？]")


def _safe_id(paper: Dict[str, Any]) -> str:
    """从 paper 构造稳定 source_id。

    优先级：arxiv_id > link > title 前 32 字符 + hash
    """
    arxiv_id = paper.get("arxiv_id")
    if arxiv_id:
        return str(arxiv_id)
    link = paper.get("link", "")
    if link:
        return hashlib.md5(link.encode("utf-8")).hexdigest()[:16]
    title = paper.get("title", "")
    return hashlib.md5(title[:32].encode("utf-8")).hexdigest()[:16]


class Chunker:
    """段落 + 滑动窗口 chunk 切分。"""

    def __init__(
        self,
        *,
        max_chunk_chars: int = 800,
        overlap_chars: int = 200,
    ):
        if overlap_chars >= max_chunk_chars:
            raise ValueError(
                f"overlap_chars ({overlap_chars}) must be < max_chunk_chars ({max_chunk_chars})"
            )
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

    def split_text(self, text: str, *, source_id: str = "") -> List[Dict[str, Any]]:
        """切分纯文本 → chunks。"""
        if not text or not text.strip():
            return []
        text = text.strip()
        paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        out: List[Dict[str, Any]] = []
        chunk_idx = 0
        cursor = 0
        for para in paragraphs:
            # 在原文中找位置（用于 start/end）
            start = text.find(para, cursor)
            if start < 0:
                start = cursor
            cursor = start + len(para)

            if len(para) <= self.max_chunk_chars:
                out.append(self._make_chunk(para, source_id, chunk_idx, start, start + len(para)))
                chunk_idx += 1
            else:
                # 滑动窗口
                window_chunks = self._sliding_window(para, start)
                for wc_text, wc_start, wc_end in window_chunks:
                    out.append(self._make_chunk(wc_text, source_id, chunk_idx, wc_start, wc_end))
                    chunk_idx += 1
        return out

    def split_paper(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """切分单篇论文 → chunks。

        paper 期望含 title + summary；标题作为 chunk 0 的前缀以保留主题。
        """
        source_id = _safe_id(paper)
        title = paper.get("title", "")
        summary = paper.get("summary", "")
        if title and summary:
            text = f"标题: {title}\n\n{summary}"
        else:
            text = title or summary
        return self.split_text(text, source_id=source_id)

    def split_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量切分。"""
        out: List[Dict[str, Any]] = []
        for p in papers:
            out.extend(self.split_paper(p))
        return out

    # === 内部 ===

    def _make_chunk(
        self,
        text: str,
        source_id: str,
        idx: int,
        start: int,
        end: int,
    ) -> Dict[str, Any]:
        return {
            "chunk_id": f"{source_id}:{idx}",
            "source_id": source_id,
            "text": text,
            "start": start,
            "end": end,
        }

    def _sliding_window(
        self,
        text: str,
        base_offset: int,
    ) -> List[tuple]:
        """滑动窗口切分（句末 + 数学块边界对齐）。

        step = max_chunk_chars - overlap_chars（保证 overlap）
        每窗口末尾 100 字符内优先找：
        1. 数学块结尾（$、]、)）（v6-4）
        2. 句末标点（。.!?！？）
        3. 硬切
        """
        step = self.max_chunk_chars - self.overlap_chars
        out = []
        i = 0
        n = len(text)
        last_end = 0
        while i < n:
            end = min(i + self.max_chunk_chars, n)
            if end < n:
                boundary = self._safe_boundary(text, end, look_back=100)
                # 边界必须至少前进 step
                if boundary > i + step:
                    end = boundary
            chunk_text = text[i:end].strip()
            if chunk_text:
                out.append((chunk_text, base_offset + i, base_offset + end))
                last_end = end
            if end >= n:
                break
            i = i + step
            if i <= last_end - 1 and i + step <= last_end:
                # i 没真前进（边界对齐卡住），强制 step 推进
                i = last_end
        return out

    def _find_sentence_end(self, text: str, pos: int, *, look_back: int = 100) -> int:
        """在 pos 往前 look_back 字符内找最近的句末边界。"""
        lo = max(0, pos - look_back)
        for i in range(pos - 1, lo - 1, -1):
            if _SENTENCE_END_RE.match(text[i]):
                return i + 1
        return pos  # 没找到，返回原 pos

    def _safe_boundary(self, text: str, pos: int, *, look_back: int = 100) -> int:
        """v6-4：找边界（数学块 > 句末 > 硬切）。

        优先在 pos 往前 look_back 字符内找数学块结尾字符（$、]、)），
        没有再找句末标点。
        返回的位置是"切片终点"（exclusive）。
        """
        if not text:
            return pos
        lo = max(0, pos - look_back)
        # 优先级 1：数学块结尾字符
        # 包含：$（公式围栏）、]/）（括号闭）、+ - =（运算符边界）
        math_chars = {"$", "]", ")", "+", "-", "="}
        for i in range(pos - 1, lo - 1, -1):
            if text[i] in math_chars:
                return i + 1
        # 优先级 2：句末标点
        for i in range(pos - 1, lo - 1, -1):
            if _SENTENCE_END_RE.match(text[i]):
                return i + 1
        return pos
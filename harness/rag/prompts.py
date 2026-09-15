"""
harness/rag/prompts.py - RAG 检索上下文 prompt 模板

用法：
    from harness.rag.prompts import SIMILAR_PAPERS_CONTEXT_PROMPT

    block = SIMILAR_PAPERS_CONTEXT_PROMPT.format(
        top_k=5,
        similar_papers_block=retriever.format_context(chunks),
    )
"""


SIMILAR_PAPERS_CONTEXT_PROMPT = """## 相关论文参考（本地 RAG 检索 Top {top_k}）

以下是与你当前研究主题最相似的论文摘要片段，仅供灵感参考，请勿直接复用其因子名称：

{similar_papers_block}

## 参考规则
1. 仅在论文主题/方法与上述片段高度相关时借鉴其因子思路
2. 不要复述上述内容；它们是上下文，不是答案
3. 如果参考论文讨论的因子与当前论文无关，忽略即可
""".strip()
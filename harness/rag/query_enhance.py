"""
harness/rag/query_enhance.py - query 预处理

- 跨语言轻量增强：中文 query 去停用词 + 简单同义词扩展
- 不引入翻译模型，零额外依赖

设计：
- detect_chinese(query) → 是否有中文字符
- expand_query(query) → 加同义词（中英映射，常见金融/ML 词）
- tokenize_for_bm25(query) → 中英混合分词（单字 + 词）

用法：
    from harness.rag.query_enhance import expand_query, is_chinese
    if is_chinese(q):
        q_expanded = expand_query(q)
        # 用 q_expanded 跑检索
"""
from __future__ import annotations

import re
from typing import List, Set


# 简化的金融/ML 中英同义词表（覆盖 ground-truth 高频词）
_SYNONYMS: dict = {
    # 因子/金融
    "因子": ["factor", "alpha", "signal"],
    "动量": ["momentum", "trend"],
    "反转": ["reversal", "revert"],
    "价值": ["value"],
    "质量": ["quality"],
    "波动": ["volatility"],
    "套利": ["arbitrage"],
    "高频": ["high frequency", "hft"],
    "低频": ["low frequency"],
    "量化": ["quantitative", "quant"],
    "回测": ["backtest", "backtesting"],
    # 时序/ML
    "时序": ["time series", "temporal", "sequential"],
    "预测": ["forecast", "prediction", "forecasting"],
    "预训练": ["pretrained", "pre-training", "pretraining"],
    "基础模型": ["foundation model", "base model"],
    "生成": ["generative", "generation"],
    "扩散": ["diffusion"],
    "推荐": ["recommendation", "recommender", "recsys"],
    "协同": ["collaborative"],
    "过滤": ["filtering"],
    "强化学习": ["reinforcement learning", "rl"],
    "深度学习": ["deep learning", "dl"],
    "注意力": ["attention"],
    "训练": ["training", "train"],
    "测试": ["test", "testing"],
    # 通用
    "数据": ["data", "dataset"],
    "模型": ["model"],
    "方法": ["method", "approach"],
    "论文": ["paper"],
}

# 停用词
_STOPWORDS: Set[str] = {
    "的", "了", "和", "是", "在", "与", "或", "为", "对", "把",
    "我", "你", "他", "她", "它", "我们", "他们",
    "什么", "怎么", "如何", "哪些", "是否",
    "中", "上", "下", "里", "外", "前", "后",
    "请", "可以", "建议", "需要", "想要",
}


def is_chinese(text: str) -> bool:
    """query 是否含中文字符（用于决定是否走中文增强）。"""
    return any("一" <= c <= "鿿" for c in text)


def expand_query(query: str) -> str:
    """给中文 query 加同义词（不引入翻译模型）。

    例子: "动量因子" → "动量 momentum trend 因子 factor alpha signal"
    """
    if not is_chinese(query):
        return query  # 英文不处理（embedder 已是英文模型）

    # 分词（中文 2字短语 / 单字 + 英文整体保留）
    import re as _re
    tokens: List[str] = []
    # 先抽英文 token（连续 A-Za-z）
    en_tokens: List[str] = []
    for m in _re.finditer(r"[A-Za-z]+", query):
        en_tokens.append(m.group())
        # 用 0 长度占位避免再次匹配
        query = query.replace(m.group(), " ", 1)
    # 剩下的是中文 + 标点
    i = 0
    while i < len(query):
        # 跳过空白
        if query[i].isspace():
            i += 1
            continue
        # 2 字中文短语
        if i + 1 < len(query) and "一" <= query[i] <= "鿿" and "一" <= query[i + 1] <= "鿿":
            two = query[i:i + 2]
            if two in _SYNONYMS and two not in _STOPWORDS:
                tokens.append(two)
                i += 2
                continue
            # 检查 next 两字（"预训练" 滑窗）
            if i + 2 < len(query) and "一" <= query[i + 2] <= "鿿":
                next_two = query[i + 1:i + 3]
                if next_two in _SYNONYMS and next_two not in _STOPWORDS:
                    ch = query[i]
                    if ch not in _STOPWORDS:
                        tokens.append(ch)
                    i += 1
                    continue
        # 单字
        ch = query[i]
        if "一" <= ch <= "鿿" and ch not in _STOPWORDS:
            tokens.append(ch)
        # 非中文（标点）跳过
        i += 1
    # 合并
    all_tokens = en_tokens + tokens

    # 展开同义词（仅中文 token 展开；英文 token 直接保留）
    expanded_tokens: List[str] = []
    for tok in all_tokens:
        expanded_tokens.append(tok)
        if tok in _SYNONYMS:
            for syn in _SYNONYMS[tok]:
                if syn not in expanded_tokens:
                    expanded_tokens.append(syn)

    # 原始 token（精确匹配）+ 扩展 token（语义）混合
    return " ".join(expanded_tokens)


__all__ = [
    "is_chinese",
    "expand_query",
]
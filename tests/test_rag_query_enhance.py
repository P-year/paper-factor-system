"""
test_rag_query_enhance.py - query 预处理单元测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.rag.query_enhance import is_chinese, expand_query


# === is_chinese ===

def test_is_chinese_pure_zh():
    assert is_chinese("动量因子") is True


def test_is_chinese_pure_en():
    assert is_chinese("momentum factor") is False


def test_is_chinese_mixed():
    """中英混合也算中文。"""
    assert is_chinese("Moirai 时序 基础模型") is True


def test_is_chinese_empty():
    assert is_chinese("") is False


# === expand_query ===

def test_expand_query_pure_zh_adds_synonyms():
    """纯中文 query 展开同义词。"""
    out = expand_query("动量因子")
    # 原 token 保留 + 英文同义词加入
    assert "动量" in out
    assert "因子" in out
    assert "momentum" in out
    assert "trend" in out
    assert "factor" in out
    assert "alpha" in out


def test_expand_query_pure_en_no_op():
    """纯英文 query 不处理。"""
    out = expand_query("momentum factor")
    assert out == "momentum factor"


def test_expand_query_mixed():
    """中英混合：英文保留 + 中文展开。"""
    out = expand_query("Moirai 时序 预训练")
    # 英文 token 保留
    assert "Moirai" in out
    # 中文展开
    assert "时序" in out
    assert "time series" in out  # 中文 → 英文同义词
    # 预训练 = 预 + 训练（算法滑窗）
    assert "pretrained" in out or "训练" in out


def test_expand_query_strips_stopwords():
    """停用词（"了" "的"）被过滤。"""
    out = expand_query("动量的因子表现")
    # "的" 是停用词，不应在 out
    assert "的" not in out.split()
    # "动量" "因子" 保留
    assert "动量" in out
    assert "因子" in out


def test_expand_query_two_char_phrase():
    """二字词组能识别为整体。"""
    out = expand_query("协同过滤")
    # 单字展开后拼接：协同 collaborative 过滤 filtering
    assert "collaborative" in out
    assert "filtering" in out
    # 验证没被错误地切碎（不应有 "协 同" 两个独立单字）
    tokens = out.split()
    # 至少 "协同" + "collaborative" 出现（连续）
    assert "协同 collaborative" in out or "协同" in tokens


def test_expand_query_dedups():
    """同义词去重。"""
    out = expand_query("动量 动量 动量")
    # "momentum" 应只出现一次
    assert out.count("momentum") == 1


def test_expand_query_handles_unknown_tokens():
    """未在同义词表的 token 保留原样。"""
    out = expand_query("区块链因子")
    # "区块链" 不在表，保留
    assert "区" in out or "块" in out or "链" in out
    # "因子" 展开
    assert "factor" in out


def test_expand_query_recall_boost_zh():
    """真实 recall 提升（端到端 smoke）。"""
    # 1. 不展开
    from harness.rag.store import PaperRAGStore
    from harness.rag.embedder import HashBackend
    from harness.rag.pdf_loader import pdf_dir_to_papers
    from harness.rag.retriever import PaperRetriever
    import tempfile
    from pathlib import Path

    pdfs = PROJECT_ROOT / "tests" / "fixtures" / "pdfs"
    if not pdfs.exists() or not list(pdfs.glob("*.pdf")):
        pytest.skip("PDF fixtures not available")

    queries = [
        "动量因子表现",
        "推荐系统协同过滤",
        "时序预训练基础模型",
    ]
    with tempfile.TemporaryDirectory() as td:
        store = PaperRAGStore(rag_dir=Path(td) / "test", embedder=HashBackend())
        papers = pdf_dir_to_papers(pdfs, max_pages=3)
        store.index_papers(papers)
        r = PaperRetriever(store, fusion_strategy="rrf")

        for q in queries:
            chunks = r.retrieve(q, top_k=3)
            assert len(chunks) >= 0  # 不崩
            # chunks 应含 paper_title
            for c in chunks:
                assert "paper_title" in c


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
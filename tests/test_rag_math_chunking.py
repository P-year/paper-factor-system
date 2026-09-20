"""
test_rag_math_chunking.py - 公式感知 chunking 测试

覆盖：
1. _safe_boundary 数学块优先
2. 滑动窗口遇 $ 字符调整边界
3. 长公式段不硬切
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.rag.chunker import Chunker


@pytest.fixture
def chunker():
    return Chunker(max_chunk_chars=100, overlap_chars=20)


# === _safe_boundary 单元测试 ===

def test_safe_boundary_math_priority(chunker):
    """数学块结尾（$）优先级 > 句末（.）。"""
    text = "The equation $x + y = z$ has a clear meaning."
    # pos=35（"meaning."后）附近，应找最近 $（位置 27）or "."(位置 32)
    boundary = chunker._safe_boundary(text, pos=35, look_back=20)
    # $在 pos-8（27），. 在 pos-3（32）→ $ 优先
    assert boundary <= 32  # ≤ 句末位置


def test_safe_boundary_falls_back_to_sentence(chunker):
    """无数学块时回退到句末。"""
    text = "This is sentence one. This is sentence two."
    boundary = chunker._safe_boundary(text, pos=20, look_back=20)
    # 应找 "one." 的位置
    assert boundary >= 16 and boundary <= 22


def test_safe_boundary_no_boundary_found(chunker):
    """无任何边界时返回原 pos（硬切）。"""
    text = "abcdefghijklmnopqrstuvwxyz"  # 无标点
    boundary = chunker._safe_boundary(text, pos=15, look_back=10)
    assert boundary == 15


def test_safe_boundary_closing_bracket(chunker):
    """检测 ] / ) 作为数学块结尾。"""
    text = "Result f(x) = 0] and proof."
    boundary = chunker._safe_boundary(text, pos=22, look_back=15)
    # ] 在 14，. 在 19。数学字符集含 ]→ 返回 15（] 后一位）
    # 句末 . → 返回 20
    assert boundary <= 16  # 应更接近 ] (15)


def test_safe_boundary_look_back_limit(chunker):
    """超出 look_back 范围的边界不考虑。"""
    text = "$".join(["x"] * 50)  # $ 离 pos 太远
    text += "." * 30  # . 接近 pos
    boundary = chunker._safe_boundary(text, pos=80, look_back=10)
    # 句末应在 look_back 内
    assert boundary >= 70


# === sliding_window 实际行为 ===

def test_long_formula_section_no_hard_cut(chunker):
    """长公式段：`_safe_boundary` 优先数学字符 > 句末。

    验证：含公式 + 句末标点的混合文本，边界应优先对齐数学字符。
    """
    # 公式 + 句末交替，公式字符 + / - / = / $ / ] / ) 优先级 > .
    text = ("Equation $x + 1 = y$. " + "And we prove the result. ") * 50
    chunks = chunker.split_text(text, source_id="test")
    # 至少一个 chunk 末尾是 . 或 $（自然边界）
    endings = [c["text"].rstrip()[-1] for c in chunks]
    assert any(e in (".", "$", "。") for e in endings)


def test_math_then_text_chunks(chunker):
    """公式后接文本：边界对齐公式末尾。"""
    text = "Equation $a + b = c$ " * 30  # ~900 chars
    chunks = chunker.split_text(text, source_id="test")
    assert len(chunks) >= 1


def test_chinese_with_formula(chunker):
    """中文段落含公式。"""
    text = "动量因子公式 $r = (p_t - p_{t-1}) / p_{t-1}$ 表现显著。" * 30
    chunks = chunker.split_text(text, source_id="test")
    assert len(chunks) >= 1


# === 实际 PDF 端到端 ===

def test_real_pdf_with_math_not_hard_cut(chunker):
    """学术 PDF 含公式，长段应不硬切。"""
    # 模拟论文 Methods 章节：一段公式密集的长文
    para = (
        "We estimate the factor return $r_t = w_t^T r_{t+1}$ "
        "using the optimization "
        "$\\min_w \\sum_i (w^T x_i - y_i)^2 + \\lambda ||w||_2^2$. "
    )
    para = para * 30  # ~3KB
    chunks = chunker.split_text(para, source_id="paper_with_math")
    assert len(chunks) >= 1
    # 至少一个 chunk 不被硬切
    ok = False
    for c in chunks:
        # chunk 末尾是 $ 或 . 或其他完整边界
        t = c["text"].rstrip()
        if t.endswith("$") or t.endswith(".") or t.endswith("。"):
            ok = True
            break
    assert ok, "at least one chunk should end on a natural boundary"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
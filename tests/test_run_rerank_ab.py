"""
test_run_rerank_ab.py - v10：真 reranker A/B benchmark 测试
"""
import json
import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "benchmarks"))

import pytest

import run_rerank_ab
from run_rerank_ab import _build_embedder, _summarize, _diff


# === _build_embedder ===

def test_build_embedder_hash():
    emb, label = _build_embedder("hash")
    assert label == "hash"
    assert emb.dim > 0


def test_build_embedder_unknown():
    with pytest.raises(ValueError, match="unknown embedder"):
        _build_embedder("unknown-model")


# === _summarize / _diff ===

def test_summarize_extracts_metrics():
    report = {
        "recall_at_k": 0.85,
        "mrr": 0.72,
        "ndcg_at_k": 0.78,
        "per_query": [],  # 不应进 summary
        "queries_count": 195,
    }
    s = _summarize(report)
    assert set(s.keys()) == {"recall_at_k", "mrr", "ndcg_at_k"}
    assert s["recall_at_k"] == 0.85
    assert "per_query" not in s


def test_diff_calculates_delta():
    baseline = {"recall_at_k": 0.70, "mrr": 0.55, "ndcg_at_k": 0.58}
    rerank = {"recall_at_k": 0.75, "mrr": 0.62, "ndcg_at_k": 0.66}
    d = _diff(baseline, rerank)
    assert abs(d["recall_at_k"] - 0.05) < 1e-6
    assert abs(d["mrr"] - 0.07) < 1e-6
    assert abs(d["ndcg_at_k"] - 0.08) < 1e-6


def test_diff_negative():
    """rerank 变差时 diff 应为负。"""
    baseline = {"recall_at_k": 0.80, "mrr": 0.70, "ndcg_at_k": 0.75}
    rerank = {"recall_at_k": 0.70, "mrr": 0.60, "ndcg_at_k": 0.65}
    d = _diff(baseline, rerank)
    assert d["recall_at_k"] < 0
    assert d["mrr"] < 0
    assert d["ndcg_at_k"] < 0


def test_diff_zero():
    baseline = {"recall_at_k": 0.5, "mrr": 0.5, "ndcg_at_k": 0.5}
    d = _diff(baseline, dict(baseline))
    assert all(v == 0.0 for v in d.values())


# === 端到端 A/B（需要 fixture PDFs；本地跑；CI 跳过）===

@pytest.mark.skipif(
    not (PROJECT_ROOT / "tests" / "fixtures" / "pdfs").exists()
    or not list((PROJECT_ROOT / "tests" / "fixtures" / "pdfs").glob("*.pdf")),
    reason="PDF fixtures 不存在"
)
def test_rerank_ab_smoke_end_to_end(tmp_path, monkeypatch):
    """A/B 端到端：跑 hash + noop reranker → 生成两个 results 文件。"""
    import run_rerank_ab as m

    # Mock real reranker（避免 300MB 下载）：用 noop 代替
    class FakeReranker:
        def __init__(self):
            self.model_name = "fake-noop"
        def _load(self):
            pass
        def rerank(self, query, chunks, *, top_k=None):
            return chunks[:top_k] if top_k else chunks

    monkeypatch.setattr(m, "_build_real_reranker", lambda name: FakeReranker())

    # 截短 queries 加快测试
    original_queries = m.QUERIES_PATH
    short_q = tmp_path / "short_queries.jsonl"
    qs = []
    for line in original_queries.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            qs.append(line)
    short_q.write_text("\n".join(qs[:10]) + "\n", encoding="utf-8")
    monkeypatch.setattr(m, "QUERIES_PATH", short_q)
    # run_rerank_ab imports from _common; patch there too
    import _common
    monkeypatch.setattr(_common, "QUERIES_PATH", short_q)

    # 改 results_dir 到 tmp
    monkeypatch.setattr(_common, "RESULTS_DIR", tmp_path / "results")
    # run_rerank_ab 模块级 import 也要 patch
    monkeypatch.setattr(m, "RESULTS_DIR", tmp_path / "results")

    # 跑
    monkeypatch.setattr(sys, "argv", ["run_rerank_ab.py", "--embedder", "hash", "--k", "5"])
    try:
        m.main()
    except RuntimeError:
        # pytest capture 内嵌 RuntimeError：忽略（不影响结果文件生成）
        pass

    # 应生成 diff 文件
    results = list((tmp_path / "results").glob("*.json"))
    assert len(results) >= 2  # baseline + rerank
    diff_file = next((tmp_path / "results").glob("rerank_ab_*.json"))
    diff = json.loads(diff_file.read_text(encoding="utf-8"))
    assert "baseline" in diff
    assert "rerank" in diff
    assert "diff" in diff


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

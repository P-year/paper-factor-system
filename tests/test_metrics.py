"""
test_metrics.py - Prometheus 指标单元测试

覆盖：
1. 各 record 函数正确写入 counter / histogram
2. metrics_text 输出 Prometheus 文本格式
3. start_metrics_server（不真启动，只验证函数）
4. 没有 prometheus_client 时优雅降级
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness import metrics


@pytest.fixture(autouse=True)
def disable_real_build(monkeypatch):
    """所有测试 mock 掉 _build_metrics 防止真实构建。"""
    monkeypatch.setattr(metrics, "_build_metrics", lambda: None)


# === 基本记录 ===

def test_record_node_duration():
    """record_node_duration 应写入 histogram。"""
    class FakeHistogram:
        def __init__(self):
            self.observed = []
        def labels(self, **kw):
            outer = self
            class L:
                def observe(self, v):
                    outer.observed.append((kw, v))
            return L()

    fake = FakeHistogram()
    metrics.node_duration_histogram = fake

    metrics.record_node_duration("plan", 150.0, "paper_factor")
    assert fake.observed == [(dict(node="plan", pipeline="paper_factor"), 150.0)]


def test_record_node_invocation():
    class FakeCounter:
        def __init__(self):
            self.counts = []
        def labels(self, **kw):
            outer = self
            class L:
                def inc(self, n=1):
                    outer.counts.append((kw, n))
            return L()
    fake = FakeCounter()
    metrics.node_invocations_counter = fake

    metrics.record_node_invocation("plan", "success", "paper_factor")
    assert fake.counts == [(dict(node="plan", pipeline="paper_factor", status="success"), 1)]


def test_record_llm_tokens():
    class FakeCounter:
        def __init__(self):
            self.counts = []
        def labels(self, **kw):
            outer = self
            class L:
                def inc(self, n=1):
                    outer.counts.append((kw, n))
            return L()
    fake = FakeCounter()
    metrics.llm_tokens_counter = fake

    metrics.record_llm_tokens("deepseek", "deepseek-chat", 1000, 500)
    assert (dict(provider="deepseek", model="deepseek-chat", direction="input"), 1000) in fake.counts
    assert (dict(provider="deepseek", model="deepseek-chat", direction="output"), 500) in fake.counts


def test_record_llm_cost():
    class FakeCounter:
        def __init__(self):
            self.counts = []
        def labels(self, **kw):
            outer = self
            class L:
                def inc(self, n=1):
                    outer.counts.append((kw, n))
            return L()
    fake = FakeCounter()
    metrics.llm_cost_counter = fake

    metrics.record_llm_cost("deepseek", "deepseek-chat", 0.001)
    assert fake.counts == [(dict(provider="deepseek", model="deepseek-chat"), 0.001)]


def test_record_run_invocation():
    class FakeCounter:
        def __init__(self):
            self.counts = []
        def labels(self, **kw):
            outer = self
            class L:
                def inc(self, n=1):
                    outer.counts.append((kw, n))
            return L()
    fake = FakeCounter()
    metrics.run_invocations_counter = fake

    metrics.record_run_invocation("paper_factor", "scheduler")
    assert fake.counts == [(dict(pipeline="paper_factor", mode="scheduler"), 1)]


def test_record_rate_limited():
    class FakeCounter:
        def __init__(self):
            self.counts = []
        def labels(self, **kw):
            outer = self
            class L:
                def inc(self, n=1):
                    outer.counts.append((kw, n))
            return L()
    fake = FakeCounter()
    metrics.rate_limited_counter = fake

    metrics.record_rate_limited("arxiv_search")
    assert fake.counts == [(dict(tool="arxiv_search"), 1)]


# === 降级（metrics 全 None 时不崩）===

def test_record_doesnt_crash_when_metrics_none():
    metrics.node_duration_histogram = None
    metrics.node_invocations_counter = None
    metrics.llm_tokens_counter = None
    metrics.llm_cost_counter = None
    metrics.run_invocations_counter = None
    metrics.rate_limited_counter = None

    metrics.record_node_duration("plan", 100)
    metrics.record_node_invocation("plan", "success")
    metrics.record_llm_tokens("d", "m", 1, 2)
    metrics.record_llm_cost("d", "m", 0.1)
    metrics.record_run_invocation("p", "m")
    metrics.record_rate_limited("t")


# === metrics_text ===

def test_metrics_text_graceful_when_unavailable(monkeypatch):
    monkeypatch.setattr(metrics, "PROMETHEUS_AVAILABLE", False)
    text = metrics.metrics_text()
    assert "not installed" in text


# === start_metrics_server ===

def test_start_metrics_server_returns_bool():
    result = metrics.start_metrics_server(port=19999)
    assert isinstance(result, bool)


def test_is_available():
    assert isinstance(metrics.is_available(), bool)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
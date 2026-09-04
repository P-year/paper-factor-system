"""
harness/metrics.py - Prometheus 指标

目的：暴露节点延迟 p50/p95、LLM token 速率、cost/run、错误率。

指标：
    harness_node_duration_ms       Histogram    每个节点耗时
    harness_node_invocations_total Counter      每个节点调用次数（status=success/error 标签）
    harness_llm_tokens_total       Counter      LLM token 用量（direction=input/output）
    harness_llm_cost_usd_total     Counter      LLM USD cost
    harness_run_invocations_total  Counter      run 启动次数（pipeline 标签）
    harness_rate_limited_total     Counter      rate limit 触发次数

暴露：
    start_metrics_server(port=9090) → 启动 HTTP server 暴露 /metrics
    metrics_text()                   → Prometheus 文本格式（用于测试 / 自定义暴露）

用法：
    from harness.metrics import (
        node_duration_histogram, llm_tokens_counter, llm_cost_counter,
        start_metrics_server,
    )

    node_duration_histogram.labels(node="plan").observe(123)
    llm_tokens_counter.labels(provider="deepseek", model="deepseek-chat", direction="input").inc(100)
    llm_cost_counter.labels(provider="deepseek", model="deepseek-chat").inc(0.001)
"""
import threading
from typing import Any, Dict, Optional

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        CollectorRegistry,
        start_http_server,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# === 默认 registry（避免污染全局） ===

_DEFAULT_REGISTRY = None


def _get_registry():
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        if PROMETHEUS_AVAILABLE:
            _DEFAULT_REGISTRY = CollectorRegistry()
        else:
            _DEFAULT_REGISTRY = None
    return _DEFAULT_REGISTRY


# === 指标定义 ===

# 节点延迟（毫秒）
node_duration_histogram = None
# 节点调用次数
node_invocations_counter = None
# LLM tokens
llm_tokens_counter = None
# LLM cost
llm_cost_counter = None
# run 启动次数
run_invocations_counter = None
# rate limit 触发
rate_limited_counter = None


def _build_metrics():
    """懒构建指标（首次访问时）。"""
    global node_duration_histogram, node_invocations_counter, llm_tokens_counter
    global llm_cost_counter, run_invocations_counter, rate_limited_counter, _metrics_built

    if not PROMETHEUS_AVAILABLE:
        return

    if _metrics_built:
        return  # 已构建过：跳过

    reg = _get_registry()
    if reg is None:
        return

    # Histogram buckets：覆盖 1ms ~ 30s
    buckets = (1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 30000)

    node_duration_histogram = Histogram(
        "harness_node_duration_ms",
        "Node execution duration in milliseconds",
        labelnames=("node", "pipeline"),
        buckets=buckets,
        registry=reg,
    )
    node_invocations_counter = Counter(
        "harness_node_invocations_total",
        "Total node invocations",
        labelnames=("node", "pipeline", "status"),
        registry=reg,
    )
    llm_tokens_counter = Counter(
        "harness_llm_tokens_total",
        "Total LLM tokens used",
        labelnames=("provider", "model", "direction"),
        registry=reg,
    )
    llm_cost_counter = Counter(
        "harness_llm_cost_usd_total",
        "Total LLM cost in USD",
        labelnames=("provider", "model"),
        registry=reg,
    )
    run_invocations_counter = Counter(
        "harness_run_invocations_total",
        "Total run invocations",
        labelnames=("pipeline", "mode"),
        registry=reg,
    )
    rate_limited_counter = Counter(
        "harness_rate_limited_total",
        "Total rate limit triggers",
        labelnames=("tool",),
        registry=reg,
    )
    _metrics_built = True


# === 便捷函数 ===

def record_node_duration(node_name: str, duration_ms: float, pipeline: str = "unknown") -> None:
    """记录节点耗时。"""
    _build_metrics()
    if node_duration_histogram is None:
        return
    try:
        node_duration_histogram.labels(node=node_name, pipeline=pipeline).observe(duration_ms)
    except Exception:
        pass


def record_node_invocation(node_name: str, status: str, pipeline: str = "unknown") -> None:
    """status = "success" | "error"。"""
    _build_metrics()
    if node_invocations_counter is None:
        return
    try:
        node_invocations_counter.labels(node=node_name, pipeline=pipeline, status=status).inc()
    except Exception:
        pass


def record_llm_tokens(provider: str, model: str, prompt: int, completion: int) -> None:
    """记录 LLM tokens。"""
    _build_metrics()
    if llm_tokens_counter is None:
        return
    try:
        llm_tokens_counter.labels(provider=provider, model=model, direction="input").inc(prompt)
        llm_tokens_counter.labels(provider=provider, model=model, direction="output").inc(completion)
    except Exception:
        pass


def record_llm_cost(provider: str, model: str, cost_usd: float) -> None:
    """记录 LLM USD cost。"""
    _build_metrics()
    if llm_cost_counter is None:
        return
    try:
        llm_cost_counter.labels(provider=provider, model=model).inc(cost_usd)
    except Exception:
        pass


def record_run_invocation(pipeline: str, mode: str) -> None:
    """mode = "repl" | "scheduler" | "api" | "eval"。"""
    _build_metrics()
    if run_invocations_counter is None:
        return
    try:
        run_invocations_counter.labels(pipeline=pipeline, mode=mode).inc()
    except Exception:
        pass


def record_rate_limited(tool_name: str) -> None:
    """rate limit 触发次数。"""
    _build_metrics()
    if rate_limited_counter is None:
        return
    try:
        rate_limited_counter.labels(tool=tool_name).inc()
    except Exception:
        pass


# === 服务端 ===

_metrics_server_started = False
_metrics_built = False
_metrics_lock = threading.Lock()


def start_metrics_server(port: int = 9090) -> bool:
    """启动 Prometheus metrics HTTP server（/metrics）。

    Returns:
        True 启动成功，False 失败（或 prometheus_client 未安装）。
    """
    global _metrics_server_started
    if not PROMETHEUS_AVAILABLE:
        return False

    with _metrics_lock:
        if _metrics_server_started:
            return True
        _build_metrics()
        try:
            start_http_server(port, registry=_get_registry())
            _metrics_server_started = True
            return True
        except Exception:
            return False


def metrics_text() -> str:
    """返回 Prometheus 文本格式的 metrics 输出。"""
    if not PROMETHEUS_AVAILABLE:
        return "# prometheus_client not installed\n"
    _build_metrics()
    try:
        return generate_latest(_get_registry()).decode("utf-8")
    except Exception as e:
        return f"# error: {e}\n"


def is_available() -> bool:
    """prometheus_client 是否可用。"""
    return PROMETHEUS_AVAILABLE
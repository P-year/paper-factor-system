"""
harness/ratelimit.py - Per-tool 速率限制

目的：保护外部 API（arXiv / AKShare）不被 ban / 触发限流。

实现：滑动窗口（sliding window）计数器。

默认限制：
    arxiv_search    5 次 / 60s   (arXiv 官方建议：每 IP 4 req/s 但实际更严)
    akshare_*       30 次 / 60s  (AKShare 没有官方限制，但防止误用)
    web_*           10 次 / 60s  (通用 web 抓取)

用法：
    from harness.ratelimit import get_default_limiter, RateLimitExceeded

    limiter = get_default_limiter()
    limiter.check("arxiv_search")  # 超限时抛 RateLimitExceeded
    # 或：
    limiter.wait("arxiv_search")  # 超限时 sleep 到允许

配置（环境变量，JSON）：
    HARNESS_RATE_LIMITS='{"arxiv_search": {"max": 10, "window_s": 60}}'
"""
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional


class RateLimitExceeded(RuntimeError):
    """超过 per-tool 速率限制。"""
    def __init__(self, tool_name: str, max_calls: int, window_s: float, retry_after: float):
        self.tool_name = tool_name
        self.max_calls = max_calls
        self.window_s = window_s
        self.retry_after = retry_after
        super().__init__(
            f"rate limit exceeded for '{tool_name}': "
            f"{max_calls} calls / {window_s}s. retry in {retry_after:.1f}s"
        )


@dataclass
class Limit:
    max_calls: int
    window_s: float


DEFAULT_LIMITS: Dict[str, Limit] = {
    # arXiv 官方建议：4 req/s 但实际触发 429 时窗口更严
    "arxiv_search": Limit(max_calls=5, window_s=60.0),
    "arxiv_fetch": Limit(max_calls=10, window_s=60.0),
    # AKShare 没官方限制，防止误用
    "akshare_*": Limit(max_calls=30, window_s=60.0),
    # 通用 web 抓取
    "web_*": Limit(max_calls=10, window_s=60.0),
    # LLM API（防触发 rate limit / cost spike）
    "llm_call": Limit(max_calls=60, window_s=60.0),
}


def _load_limits() -> Dict[str, Limit]:
    """读 env 覆盖默认表。"""
    raw = os.getenv("HARNESS_RATE_LIMITS")
    if not raw:
        return dict(DEFAULT_LIMITS)
    try:
        cfg = json.loads(raw)
        merged = dict(DEFAULT_LIMITS)
        for name, spec in cfg.items():
            merged[name] = Limit(max_calls=int(spec["max"]), window_s=float(spec["window_s"]))
        return merged
    except Exception:
        return dict(DEFAULT_LIMITS)


def _match_limit(tool_name: str, limits: Dict[str, Limit]) -> Optional[Limit]:
    """精确匹配 + 通配符匹配（tool_name 前缀 vs prefix_*）。"""
    if tool_name in limits:
        return limits[tool_name]
    for pattern, limit in limits.items():
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            if tool_name.startswith(prefix):
                return limit
    return None


class RateLimiter:
    """线程安全的滑动窗口速率限制器。

    每个 tool 维护一个时间戳 deque，新调用 check() 时：
        - 若 deque 长度 >= max_calls，删掉超出 window 的旧条目
        - 若仍 >= max_calls，超限（抛或 wait）
        - 否则 append 当前时间戳，放行
    """

    def __init__(self, limits: Optional[Dict[str, Limit]] = None, *, default_limit: Optional[Limit] = None):
        self._limits = limits if limits is not None else _load_limits()
        self._default_limit = default_limit  # 没匹配规则的工具用
        self._calls: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def set_limit(self, tool_name: str, max_calls: int, window_s: float) -> None:
        """运行时设置/覆盖单 tool 的限制。"""
        with self._lock:
            self._limits[tool_name] = Limit(max_calls=max_calls, window_s=window_s)

    def _get_limit(self, tool_name: str) -> Optional[Limit]:
        return _match_limit(tool_name, self._limits) or self._default_limit

    def _trim(self, dq: Deque[float], window_s: float, now: float) -> None:
        """删掉超出 window 的旧条目。"""
        cutoff = now - window_s
        while dq and dq[0] < cutoff:
            dq.popleft()

    def check(self, tool_name: str) -> None:
        """检查是否允许调用。超限抛 RateLimitExceeded。

        不会自动 wait——调用方决定要不要 sleep 重试。
        """
        limit = self._get_limit(tool_name)
        if limit is None:
            return  # 没限制

        now = time.time()
        with self._lock:
            dq = self._calls.setdefault(tool_name, deque())
            self._trim(dq, limit.window_s, now)

            if len(dq) >= limit.max_calls:
                # 超限：retry_after = (最早条目 + window) - now
                retry_after = dq[0] + limit.window_s - now
                raise RateLimitExceeded(
                    tool_name=tool_name,
                    max_calls=limit.max_calls,
                    window_s=limit.window_s,
                    retry_after=max(retry_after, 0.01),
                )
            dq.append(now)

    def wait(self, tool_name: str, *, timeout: float = 60.0) -> None:
        """超限时 sleep 直到允许（或 timeout）。

        用法：limiter.wait('arxiv_search')  # 会阻塞直到配额可用
        """
        limit = self._get_limit(tool_name)
        if limit is None:
            return

        start = time.time()
        while True:
            try:
                self.check(tool_name)
                return
            except RateLimitExceeded as e:
                if time.time() - start > timeout:
                    raise
                time.sleep(min(e.retry_after, 1.0))

    def stats(self, tool_name: str) -> Dict[str, Any]:
        """查当前窗口内的调用数（用于 observability）。"""
        limit = self._get_limit(tool_name)
        if limit is None:
            return {"limited": False}
        with self._lock:
            dq = self._calls.get(tool_name, deque())
            now = time.time()
            self._trim(dq, limit.window_s, now)
            return {
                "limited": True,
                "limit_max": limit.max_calls,
                "limit_window_s": limit.window_s,
                "current_calls": len(dq),
                "remaining": max(0, limit.max_calls - len(dq)),
            }


# === 全局 limiter ===

_global_limiter: Optional[RateLimiter] = None
_global_lock = threading.Lock()


def get_default_limiter() -> RateLimiter:
    """拿全局 limiter。首次访问时构造。"""
    global _global_limiter
    if _global_limiter is None:
        with _global_lock:
            if _global_limiter is None:
                _global_limiter = RateLimiter()
    return _global_limiter


def reset_limiter() -> None:
    """清空全局 limiter（测试用）。"""
    global _global_limiter
    with _global_lock:
        _global_limiter = None
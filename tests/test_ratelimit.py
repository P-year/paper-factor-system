"""
test_ratelimit.py - 速率限制单元测试

覆盖：
1. 基本检查：未超限放行
2. 超限抛 RateLimitExceeded
3. wait() 自动 sleep 重试
4. 滑动窗口：旧条目过期后允许
5. 通配符匹配（akshare_*）
6. env 覆盖默认
7. 线程安全
8. stats() 报告
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.ratelimit import (
    RateLimiter,
    RateLimitExceeded,
    Limit,
    DEFAULT_LIMITS,
    _match_limit,
    get_default_limiter,
    reset_limiter,
)


# === 基本 ===

def test_check_passes_under_limit():
    lim = RateLimiter({"t": Limit(5, 60.0)})
    for _ in range(5):
        lim.check("t")  # 不抛


def test_check_raises_when_exceeded():
    lim = RateLimiter({"t": Limit(2, 60.0)})
    lim.check("t")
    lim.check("t")
    with pytest.raises(RateLimitExceeded) as exc:
        lim.check("t")
    assert exc.value.tool_name == "t"
    assert exc.value.max_calls == 2
    assert exc.value.retry_after > 0


def test_check_no_limit_unlimited():
    lim = RateLimiter({})  # 没规则
    for _ in range(100):
        lim.check("anything")  # 不抛


# === wait() ===

def test_wait_blocks_until_window_clears(monkeypatch):
    lim = RateLimiter({"t": Limit(2, 0.5)})  # 0.5s 窗口
    lim.check("t")
    lim.check("t")
    # 立即 wait：应 sleep 到窗口清空
    started = time.time()
    lim.wait("t")
    elapsed = time.time() - started
    assert elapsed >= 0.4  # 至少等了窗口的时间


def test_wait_returns_immediately_if_available():
    lim = RateLimiter({"t": Limit(5, 60.0)})
    lim.check("t")
    started = time.time()
    lim.wait("t")
    elapsed = time.time() - started
    assert elapsed < 0.1


def test_wait_timeout_raises():
    lim = RateLimiter({"t": Limit(1, 60.0)})
    lim.check("t")
    with pytest.raises(RateLimitExceeded):
        lim.wait("t", timeout=0.1)


# === 滑动窗口 ===

def test_window_slides():
    """窗口期内超限 → 等窗口外的条目过期 → 允许。"""
    lim = RateLimiter({"t": Limit(2, 0.2)})
    lim.check("t")
    lim.check("t")
    with pytest.raises(RateLimitExceeded):
        lim.check("t")
    # 等窗口清空
    time.sleep(0.25)
    lim.check("t")  # 应放行（之前的 2 条已过期）


# === 通配符匹配 ===

def test_match_limit_exact():
    assert _match_limit("arxiv_search", {"arxiv_search": Limit(5, 60)}) == Limit(5, 60)


def test_match_limit_wildcard():
    limits = {"akshare_*": Limit(30, 60)}
    assert _match_limit("akshare_stock_zh_a_hist", limits) == Limit(30, 60)


def test_match_limit_no_match():
    assert _match_limit("unknown_tool", {"arxiv_search": Limit(5, 60)}) is None


# === default_limit ===

def test_default_limit_applies_to_unmatched():
    lim = RateLimiter({}, default_limit=Limit(3, 60.0))
    for _ in range(3):
        lim.check("any_tool")
    with pytest.raises(RateLimitExceeded):
        lim.check("any_tool")


# === env 覆盖 ===

def test_env_overrides_default(monkeypatch):
    import json
    monkeypatch.setenv("HARNESS_RATE_LIMITS", json.dumps({
        "arxiv_search": {"max": 100, "window_s": 60},
    }))
    # 不通过 reset_limiter 因为 _load_limits 是调用时读 env
    lim = RateLimiter()
    assert lim._get_limit("arxiv_search").max_calls == 100


# === 线程安全 ===

def test_concurrent_check_thread_safe():
    lim = RateLimiter({"t": Limit(100, 60.0)})
    import threading

    def worker(n):
        for _ in range(n):
            try:
                lim.check("t")
            except RateLimitExceeded:
                pass

    threads = [threading.Thread(target=worker, args=(50,)) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    stats = lim.stats("t")
    assert stats["current_calls"] == 100  # 精确 100 次（受限于 max=100）


# === stats ===

def test_stats_reports_current():
    lim = RateLimiter({"t": Limit(5, 60.0)})
    lim.check("t")
    lim.check("t")
    s = lim.stats("t")
    assert s["limited"] is True
    assert s["current_calls"] == 2
    assert s["remaining"] == 3
    assert s["limit_max"] == 5


def test_stats_no_limit():
    lim = RateLimiter({})
    s = lim.stats("anything")
    assert s == {"limited": False}


# === global limiter ===

def test_global_limiter_persists(monkeypatch):
    reset_limiter()
    l1 = get_default_limiter()
    l2 = get_default_limiter()
    assert l1 is l2
    reset_limiter()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
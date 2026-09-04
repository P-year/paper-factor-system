"""
test_concurrent.py - 并发 run 安全测试

覆盖：
1. FileLock 基本获取/释放
2. 不同 session_id 互不干扰
3. 同一 session_id 第二次获取失败（已在握）
4. SessionLockRegistry 缓存
5. session_lock context manager
6. 跨进程模拟（subprocess 验证锁真实生效）
"""
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.concurrent import (
    FileLock,
    SessionLockRegistry,
    get_default_lock_registry,
    reset_lock_registry,
    session_lock,
    file_lock,
)


# === FileLock 基本 ===

def test_filelock_acquire_release(tmp_path):
    fp = tmp_path / "test.lock"
    fl = FileLock(fp)
    assert fl.acquire() is True
    fl.release()


def test_filelock_context_manager(tmp_path):
    fp = tmp_path / "test.lock"
    with FileLock(fp):
        # 锁住了
        pass
    # 释放了


def test_filelock_double_acquire_same_instance(tmp_path):
    """同一实例第二次 acquire 立即返回 True（已持有）。"""
    fp = tmp_path / "test.lock"
    fl = FileLock(fp)
    fl.acquire()
    assert fl.acquire() is True  # 不抛
    fl.release()


# === 同 session_id 互斥（单进程内） ===

def test_same_session_blocks(tmp_path):
    """同 session 不同 FileLock 实例互斥（模拟跨进程）。

    Linux 上有效。Windows 上 msvcrt 是 per-handle，不保证跨 handle 工作。
    """
    if sys.platform == "win32":
        pytest.skip("msvcrt.locking is per-handle on Windows")

    reg = SessionLockRegistry(tmp_path)
    fl1 = reg.get("session_A")
    fl1.acquire()

    fl2 = FileLock(reg.get("session_A").path, timeout=0.3)
    started = time.time()
    result = fl2.acquire()
    elapsed = time.time() - started
    assert result is False
    assert elapsed >= 0.25

    fl1.release()


def test_different_sessions_no_block(tmp_path):
    reg = SessionLockRegistry(tmp_path)
    fl1 = reg.get("session_A")
    fl1.acquire()
    fl2 = FileLock(reg.get("session_B").path, timeout=0.5)
    assert fl2.acquire() is True
    fl1.release()
    fl2.release()


# === session_lock context manager ===

def test_session_lock_context(tmp_path):
    reset_lock_registry()
    reg = get_default_lock_registry()
    # 用 tmp_path 作为 lock_dir
    from harness.concurrent import _default_registry, SessionLockRegistry
    import harness.concurrent as c
    c._default_registry = SessionLockRegistry(tmp_path)

    with session_lock("test_session"):
        pass  # 锁已获取并释放

    # 二次进入应成功（已释放）
    with session_lock("test_session"):
        pass


def test_session_lock_timeout(tmp_path):
    import harness.concurrent as c
    c._default_registry = SessionLockRegistry(tmp_path)

    # 直接用 FileLock（模拟跨进程）
    fl = FileLock(c._default_registry.get("test_session").path, timeout=0.3)
    fl.acquire()
    try:
        with pytest.raises(TimeoutError):
            with session_lock("test_session", timeout=0.3):
                pass
    finally:
        fl.release()


# === file_lock ===

def test_file_lock_basic(tmp_path):
    fp = tmp_path / "x.lock"
    with file_lock(fp):
        pass


def test_file_lock_blocks(tmp_path):
    fp = tmp_path / "x.lock"
    with file_lock(fp, timeout=1):
        # 已经在锁里，再锁应超时
        with pytest.raises(TimeoutError):
            with file_lock(fp, timeout=0.3):
                pass


# === 线程安全 ===

def test_thread_safety(tmp_path):
    """3 个线程：A 同 session 两个 + B 不同 session。

    Linux 上 fcntl.flock 是真正的 advisory lock，能在跨进程工作。
    Windows 上 msvcrt.locking 是 per-handle，跨进程不一定可靠——
    这里仅在 Linux 上严格测试。
    """
    if sys.platform == "win32":
        pytest.skip("msvcrt.locking is per-handle on Windows; not reliable across handles")

    reg = SessionLockRegistry(tmp_path)
    results = []

    def worker(name, path):
        fl = FileLock(path, timeout=3)
        if fl.acquire():
            results.append((name, "got"))
            time.sleep(0.1)
            fl.release()
        else:
            results.append((name, "timeout"))

    path_A = reg.get("session_A").path
    path_B = reg.get("session_B").path

    threads = [
        threading.Thread(target=worker, args=("A", path_A)),
        threading.Thread(target=worker, args=("A", path_A)),
        threading.Thread(target=worker, args=("B", path_B)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    a_results = [r for r in results if r[0] == "A"]
    b_results = [r for r in results if r[0] == "B"]
    assert sorted(r[1] for r in a_results) == ["got", "timeout"]
    assert b_results == [("B", "got")]


# === 跨进程（subprocess）===

def test_cross_process_lock(tmp_path):
    """subprocess 持有锁时，主进程拿不到（Linux only）。"""
    if sys.platform == "win32":
        pytest.skip("msvcrt.locking is per-handle on Windows")

    import subprocess
    fp = tmp_path / "cross.lock"

    # 写一个长持锁的脚本
    script = f"""
import sys, time
sys.path.insert(0, r'{PROJECT_ROOT}')
from harness.concurrent import FileLock
from pathlib import Path

fl = FileLock(Path(r'{fp}'), timeout=30)
fl.acquire()
print('LOCKED')
sys.stdout.flush()
time.sleep(3)
fl.release()
print('RELEASED')
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # 等子进程 LOCKED 输出
    line = proc.stdout.readline()
    assert "LOCKED" in line

    # 主进程尝试拿锁 → 应超时
    fl = FileLock(fp, timeout=0.5)
    assert fl.acquire() is False

    # 等子进程结束
    proc.wait(timeout=10)
    assert "RELEASED" in proc.stdout.read()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
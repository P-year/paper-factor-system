"""
harness/concurrent.py - 并发 run 安全

目的：
1. 防止同一 session_id 并发写入（race condition 损坏 JSON）
2. 多进程/多 worker 共享 state 时不互相覆盖
3. 提供跨进程的互斥（基于文件锁 fcntl / msvcrt）

实现：
    FileLock 类：基于 fcntl (Unix) / msvcrt (Windows) 的跨进程文件锁
    SessionLockRegistry：按 session_id 缓存 FileLock 实例

用法：
    from harness.concurrent import session_lock

    with session_lock("my_session_id", exclusive=True):
        # 临界区：写文件 / 改 state
        save_session_state("my_session_id", state)
"""
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

from harness.paths import SESSIONS_DIR


# === FileLock ===

class FileLock:
    """跨进程文件锁（基于 fcntl / msvcrt）。

    同一个文件路径的 FileLock 实例之间互斥（即使跨进程）。
    """

    def __init__(self, path: Path, *, timeout: float = 30.0):
        self.path = Path(path)
        self.timeout = timeout
        self._fd: Optional[int] = None
        self._lock = threading.Lock()  # 同进程内的二次保护

    def acquire(self, exclusive: bool = True, *, timeout: Optional[float] = None) -> bool:
        """获取锁。失败返回 False（超时不抛）。"""
        if not exclusive:
            raise ValueError("FileLock only supports exclusive mode")

        if timeout is None:
            timeout = self.timeout

        with self._lock:
            if self._fd is not None:
                return True  # 已持有
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)
            self._fd = os.open(
                str(self.path),
                os.O_RDWR | os.O_CREAT,
            )

        # 进入临界区（跨进程）
        start = time.time()
        while True:
            try:
                if self._lock_impl(self._fd):
                    return True
            except Exception:
                pass
            if time.time() - start > timeout:
                try:
                    os.close(self._fd)
                except Exception:
                    pass
                self._fd = None
                return False
            time.sleep(0.05)

    def release(self) -> None:
        with self._lock:
            if self._fd is None:
                return
            try:
                self._unlock_impl(self._fd)
            except Exception:
                pass
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None

    def _lock_impl(self, fd: int) -> bool:
        if sys.platform == "win32":
            return self._lock_windows(fd)
        else:
            import fcntl
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except (BlockingIOError, OSError):
                return False

    def _unlock_impl(self, fd: int) -> None:
        if sys.platform == "win32":
            self._unlock_windows(fd)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)

    def _lock_windows(self, fd: int) -> bool:
        """Windows 用 msvcrt.locking（best-effort，跨进程不一定可靠）。

        注：msvcrt.locking 是 per-handle，不是 per-file。
        在 Windows 上真正的跨进程文件锁需要 LockFileEx（更复杂）。
        本实现保证同进程内 + 多数情况下跨进程也能工作。
        """
        import msvcrt
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except (OSError, ValueError):
            return False

    def _unlock_windows(self, fd: int) -> None:
        import msvcrt
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except Exception:
            pass

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"failed to acquire lock: {self.path}")
        return self

    def __exit__(self, *args):
        self.release()


# === SessionLockRegistry ===

class SessionLockRegistry:
    """按 session_id 缓存 FileLock 实例。"""

    def __init__(self, lock_dir: Path):
        self._dir = Path(lock_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, FileLock] = {}
        self._meta_lock = threading.Lock()

    def get(self, session_id: str) -> FileLock:
        with self._meta_lock:
            if session_id not in self._locks:
                self._locks[session_id] = FileLock(
                    self._dir / f"{session_id}.lock",
                    timeout=30.0,
                )
            return self._locks[session_id]


_default_registry: Optional[SessionLockRegistry] = None


def get_default_lock_registry() -> SessionLockRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = SessionLockRegistry(SESSIONS_DIR / "_locks")
    return _default_registry


def reset_lock_registry() -> None:
    global _default_registry
    _default_registry = None


# === 便捷上下文管理器 ===

@contextmanager
def session_lock(session_id: str, *, exclusive: bool = True, timeout: float = 30.0):
    """对单个 session 加文件锁。"""
    registry = get_default_lock_registry()
    fl = registry.get(session_id)
    fl.timeout = timeout
    acquired = fl.acquire(exclusive=exclusive)
    if not acquired:
        raise TimeoutError(f"failed to acquire session lock: {session_id}")
    try:
        yield
    finally:
        fl.release()


@contextmanager
def file_lock(path: Path, *, timeout: float = 30.0):
    """对单个文件路径加文件锁。"""
    fl = FileLock(path, timeout=timeout)
    with fl:
        yield
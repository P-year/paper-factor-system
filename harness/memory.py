"""
harness/memory.py - Cold/Warm 记忆持久化层

Cold tier：JSONL 追加 + FileLock 跨进程锁 + 按 session_id/topic/time/keyword 检索。
Warm tier：复用 harness.checkpoints（save_session_state/load_session_state）。

设计：
- 每行 JSON 一个 record，schema = {ts, session_id, role, content, topic, tokens, source, **extra}
- append 用 FileLock 重试 3 次（fcntl/msvcrt 跨平台，v3-8 已验证）
- query 全表扫描（JSONL 没索引；规模小时 OK，过大用 compact_older_than 归档）
- search_text：content 子串匹配（lowercase + 含中文）
- v4-5：retention + size_limit 自动归档，避免文件无限增长
- v4-5：自动 archive 到 cold/archive/YYYY-MM.jsonl

用法：
    from harness.memory import ColdMemoryStore
    store = ColdMemoryStore(Memory/cold/all.jsonl)
    store.append({"ts": ..., "session_id": ..., "role": "user", "content": "...", "topic": "..."})
    rows = store.query(session_id="abc", topic="plan_decision", limit=10)
    matches = store.search_text("动量")
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from harness.concurrent import file_lock
from harness.paths import MEMORY_DIR


COLD_DIR = MEMORY_DIR / "cold"
COLD_ARCHIVE_DIR = COLD_DIR / "archive"

# 默认：30 天之前的记录自动归档；文件超 50MB 强制归档
DEFAULT_RETENTION_DAYS = 30
DEFAULT_SIZE_LIMIT_MB = 50
ARCHIVE_CHECK_INTERVAL = 100  # 每 N 次 append 触发一次归档检查


# === Schema 常量 ===

_VALID_KEYS: Set[str] = {"ts", "session_id", "role", "content", "topic", "tokens", "source"}


def _now_iso() -> str:
    return datetime.now().isoformat()


def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """确保 record 含必需字段。"""
    out = dict(record)
    out.setdefault("ts", _now_iso())
    out.setdefault("session_id", "")
    out.setdefault("role", "")
    out.setdefault("content", "")
    out.setdefault("topic", "general")
    out.setdefault("tokens", 0)
    out.setdefault("source", "hot")
    return out


def _matches_filters(rec: Dict[str, Any], *,
                     session_id: Optional[str],
                     topic: Optional[str],
                     since: Optional[str],
                     until: Optional[str]) -> bool:
    if session_id is not None and rec.get("session_id") != session_id:
        return False
    if topic is not None and rec.get("topic") != topic:
        return False
    if since is not None and rec.get("ts", "") < since:
        return False
    if until is not None and rec.get("ts", "") > until:
        return False
    return True


class ColdMemoryStore:
    """JSONL 持久化的 cold memory。

    append 用 FileLock 防并发覆盖（v3-8 已实测）。
    query/search_text 全表扫描，O(N)；N 大时用 compact_older_than 归档老数据。
    """

    def __init__(self, jsonl_path: Path, *, lock_dir: Optional[Path] = None,
                 retention_days: Optional[int] = None,
                 size_limit_mb: Optional[float] = None,
                 archive_dir: Optional[Path] = None):
        self.path = Path(jsonl_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        # FileLock 用 .lock 后缀
        self.lock_dir = lock_dir or (self.path.parent / "_locks")
        self.lock_dir.mkdir(parents=True, exist_ok=True)

        # v4-5：retention + 自动归档
        # None 表示"禁用"（不归档），不替换为默认值
        self.retention_days = retention_days
        self.size_limit_mb = size_limit_mb
        self.archive_dir = Path(archive_dir) if archive_dir else COLD_ARCHIVE_DIR
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._append_count = 0  # 用于触发归档检查的节流

    # === write ===

    def append(self, record: Dict[str, Any], *, max_retries: int = 3,
               auto_archive: bool = True) -> None:
        """追加一条 record 到 JSONL。

        FileLock 失败时重试（应对 fcntl 偶发 EAGAIN）。
        v4-5：auto_archive=True 时每 ARCHIVE_CHECK_INTERVAL 次 append 后
        检查 retention/size，超阈值自动归档。
        """
        rec = _normalize_record(record)
        line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
        lock_path = self.lock_dir / (self.path.stem + ".lock")

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                with file_lock(lock_path, timeout=5):
                    with self.path.open("a", encoding="utf-8") as f:
                        f.write(line)
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except OSError:
                            pass
                    self._append_count += 1
                # 自动归档（FileLock 释放后做，不阻塞下一次 append）
                if auto_archive and self._append_count % ARCHIVE_CHECK_INTERVAL == 0:
                    try:
                        self.maybe_archive()
                    except Exception:
                        pass  # 归档失败不阻塞主流程
                return
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    time.sleep(0.05 * (2 ** attempt))
        if last_exc:
            raise RuntimeError(
                f"failed to append to {self.path} after {max_retries} retries: {last_exc}"
            )

    def append_many(self, records: Iterable[Dict[str, Any]]) -> int:
        """批量追加。返回写入条数。"""
        n = 0
        for r in records:
            self.append(r)
            n += 1
        return n

    # === read ===

    def _iter_lines(self) -> Iterable[Dict[str, Any]]:
        """逐行解析 JSONL，容错（坏行跳过）。"""
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return

    def count(self) -> int:
        """总条数。"""
        if not self.path.exists():
            return 0
        n = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    def query(self, *,
              session_id: Optional[str] = None,
              topic: Optional[str] = None,
              since: Optional[str] = None,
              until: Optional[str] = None,
              limit: int = 50) -> List[Dict[str, Any]]:
        """按 session_id/topic/time 过滤，返回最近 limit 条（按 ts 倒序）。"""
        out: List[Dict[str, Any]] = []
        for rec in self._iter_lines():
            if not _matches_filters(rec, session_id=session_id, topic=topic,
                                    since=since, until=until):
                continue
            out.append(rec)
        out.sort(key=lambda r: r.get("ts", ""), reverse=True)
        return out[:limit]

    def search_text(self, keyword: str, *, limit: int = 20,
                    session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """content 子串匹配（大小写不敏感，中文也适用）。"""
        if not keyword:
            return []
        kw_lower = keyword.lower()
        out: List[Dict[str, Any]] = []
        for rec in self._iter_lines():
            if session_id is not None and rec.get("session_id") != session_id:
                continue
            content = rec.get("content", "")
            if not isinstance(content, str):
                continue
            if kw_lower in content.lower():
                out.append(rec)
        out.sort(key=lambda r: r.get("ts", ""), reverse=True)
        return out[:limit]

    # === maintenance ===

    def maybe_archive(self) -> int:
        """如需归档则触发。返回归档条数（0 = 未触发）。

        触发条件（任一满足）：
        1. 文件大小 > size_limit_mb
        2. 有超过 retention_days 天的记录

        归档到 archive_dir/YYYY-MM.jsonl（按月分文件）。
        """
        archived = 0
        # 检查文件大小（size_limit_mb=None 表示禁用 size 触发）
        if self.size_limit_mb is not None and self.size_limit_mb > 0:
            size_mb = self.path.stat().st_size / (1024 * 1024) if self.path.exists() else 0
            if size_mb > self.size_limit_mb:
                archived += self._archive_all()
                return archived

        # 检查 retention
        if self.retention_days is not None and self.retention_days > 0:
            cutoff = (datetime.now() - timedelta(days=self.retention_days)).isoformat()
            archived += self.archive_old_records(cutoff)
        return archived

    def archive_old_records(self, cutoff_iso: str) -> int:
        """把 ts < cutoff_iso 的记录归档到 archive_dir/YYYY-MM.jsonl。返回归档条数。"""
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        kept: List[Dict[str, Any]] = []
        archived_groups: Dict[str, List[Dict[str, Any]]] = {}

        for rec in self._iter_lines():
            ts = rec.get("ts", "")
            if ts < cutoff_iso:
                # 按月分文件（YYYY-MM）
                month_key = ts[:7] if len(ts) >= 7 else "unknown"
                archived_groups.setdefault(month_key, []).append(rec)
            else:
                kept.append(rec)

        if not archived_groups:
            return 0

        # 写入归档文件（每组一个）
        total_moved = 0
        for month_key, recs in archived_groups.items():
            archive_path = self.archive_dir / f"{month_key}.jsonl"
            with archive_path.open("a", encoding="utf-8") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    total_moved += 1

        # 原子重写原文件（保留新记录）
        if kept:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for rec in kept:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            os.replace(tmp, self.path)
        else:
            # 全部归档，原文件清空
            self.clear()
        return total_moved

    def _archive_all(self) -> int:
        """size 触发：全部归档（最老一组）。"""
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        recs = list(self._iter_lines())
        if not recs:
            return 0
        # 按月分
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for rec in recs:
            ts = rec.get("ts", "")
            month_key = ts[:7] if len(ts) >= 7 else "unknown"
            groups.setdefault(month_key, []).append(rec)
        for month_key, recs_in_month in groups.items():
            archive_path = self.archive_dir / f"{month_key}.jsonl"
            with archive_path.open("a", encoding="utf-8") as f:
                for rec in recs_in_month:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self.clear()
        return len(recs)

    def list_archives(self) -> List[Path]:
        """列出所有归档文件，按月排序。"""
        if not self.archive_dir.exists():
            return []
        return sorted(self.archive_dir.glob("*.jsonl"))

    def load_archive(self, month: str) -> List[Dict[str, Any]]:
        """读某月归档（YYYY-MM 格式）。返回 records。"""
        path = self.archive_dir / f"{month}.jsonl"
        if not path.exists():
            return []
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def compact_older_than(self, ts: str, dest: Path) -> int:
        """把 ts 之前的记录移到 dest（归档），从原文件删除。返回移动条数。"""
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.touch()

        kept: List[str] = []
        moved = 0
        for rec in self._iter_lines():
            if rec.get("ts", "") < ts:
                with dest.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                moved += 1
            else:
                kept.append(json.dumps(rec, ensure_ascii=False, default=str))

        # 原子重写
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        os.replace(tmp, self.path)
        return moved

    def clear(self) -> None:
        """清空（测试用）。"""
        if self.path.exists():
            self.path.unlink()
        self.path.touch()


def default_cold_store(filename: str = "all.jsonl") -> ColdMemoryStore:
    """默认 cold store 路径：memory/cold/{filename}.jsonl"""
    return ColdMemoryStore(COLD_DIR / filename)
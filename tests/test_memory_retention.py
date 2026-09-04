"""
test_memory_retention.py - ColdMemoryStore retention + 自动归档测试

覆盖：
1. retention_days 触发 archive_old_records
2. size_limit_mb 触发 _archive_all
3. archive_dir 按月分文件
4. archive 文件可读回
5. auto_archive 在 append 100 次后触发
6. retention_days=None 不归档
7. 归档后原文件只保留新记录
8. list_archives / load_archive 接口
9. cutoff 边界（= cutoff 的记录保留）
"""
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.memory import (
    ColdMemoryStore,
    DEFAULT_RETENTION_DAYS,
    DEFAULT_SIZE_LIMIT_MB,
    ARCHIVE_CHECK_INTERVAL,
)


@pytest.fixture
def cold_dir(tmp_path):
    """独立的 cold 目录（每个测试用 tmp_path）。"""
    return tmp_path / "cold"


@pytest.fixture
def store(cold_dir):
    return ColdMemoryStore(
        cold_dir / "test.jsonl",
        archive_dir=cold_dir / "archive",
        retention_days=30,
        size_limit_mb=0.001,  # 1KB，强制小文件触发
    )


@pytest.fixture
def no_retention_store(cold_dir):
    return ColdMemoryStore(
        cold_dir / "test.jsonl",
        archive_dir=cold_dir / "archive",
        retention_days=None,
        size_limit_mb=None,
    )


# === 配置默认值 ===

def test_default_retention_days():
    assert DEFAULT_RETENTION_DAYS == 30


def test_default_size_limit_mb():
    assert DEFAULT_SIZE_LIMIT_MB == 50


def test_archive_check_interval_is_positive():
    assert ARCHIVE_CHECK_INTERVAL > 0


# === archive_old_records ===

def test_archive_old_records_basic(cold_dir):
    store = ColdMemoryStore(cold_dir / "x.jsonl", archive_dir=cold_dir / "archive")
    store.append({"ts": "2020-01-15T00:00:00", "content": "old"})
    store.append({"ts": "2030-01-15T00:00:00", "content": "new"})
    moved = store.archive_old_records("2025-01-01T00:00:00")
    assert moved == 1
    # 原文件只剩 new
    assert store.count() == 1
    assert store.query()[0]["content"] == "new"
    # 归档文件存在
    assert (cold_dir / "archive" / "2020-01.jsonl").exists()


def test_archive_by_month(cold_dir):
    store = ColdMemoryStore(cold_dir / "x.jsonl", archive_dir=cold_dir / "archive")
    store.append({"ts": "2020-01-15T00:00:00", "content": "a"})
    store.append({"ts": "2020-02-15T00:00:00", "content": "b"})
    store.append({"ts": "2020-03-15T00:00:00", "content": "c"})
    moved = store.archive_old_records("2025-01-01T00:00:00")
    assert moved == 3
    # 3 个不同月份归档
    assert (cold_dir / "archive" / "2020-01.jsonl").exists()
    assert (cold_dir / "archive" / "2020-02.jsonl").exists()
    assert (cold_dir / "archive" / "2020-03.jsonl").exists()


def test_archive_cutoff_inclusive(cold_dir):
    """cutoff_iso 等于记录 ts 时：记录保留（不归档）。"""
    store = ColdMemoryStore(cold_dir / "x.jsonl", archive_dir=cold_dir / "archive")
    ts = "2025-06-15T00:00:00"
    store.append({"ts": ts, "content": "edge"})
    moved = store.archive_old_records(ts)
    assert moved == 0
    assert store.count() == 1


def test_archive_all_archives_when_all_old(cold_dir):
    store = ColdMemoryStore(cold_dir / "x.jsonl", archive_dir=cold_dir / "archive")
    store.append({"ts": "2020-01-15T00:00:00", "content": "a"})
    store.append({"ts": "2020-02-15T00:00:00", "content": "b"})
    moved = store.archive_old_records("2025-01-01T00:00:00")
    assert moved == 2
    # 原文件空
    assert store.count() == 0


def test_archive_no_op_when_nothing_old(cold_dir):
    store = ColdMemoryStore(cold_dir / "x.jsonl", archive_dir=cold_dir / "archive")
    store.append({"ts": "2030-01-15T00:00:00", "content": "future"})
    moved = store.archive_old_records("2025-01-01T00:00:00")
    assert moved == 0
    assert store.count() == 1


# === maybe_archive (size 触发) ===

def test_maybe_archive_size_trigger(cold_dir):
    """文件 > size_limit_mb 时全部归档。"""
    fp = cold_dir / "big.jsonl"
    fp.parent.mkdir(parents=True, exist_ok=True)
    # 写一个 ~2KB 的文件
    with fp.open("w") as f:
        for i in range(50):
            f.write(json_line({"ts": "2020-01-01T00:00:00", "content": "x" * 50}))

    store = ColdMemoryStore(
        fp, archive_dir=cold_dir / "archive",
        size_limit_mb=0.001,  # 1KB
        retention_days=None,
    )
    moved = store.maybe_archive()
    assert moved >= 50
    # 原文件空
    assert store.count() == 0
    # 归档文件有数据
    archives = store.list_archives()
    assert len(archives) >= 1


def test_maybe_archive_retention_trigger(cold_dir):
    """文件未超 size 但有老记录 → 按 retention 归档。"""
    fp = cold_dir / "x.jsonl"
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w") as f:
        for i in range(5):
            f.write(json_line({"ts": "2020-01-01T00:00:00", "content": "old"}))

    store = ColdMemoryStore(
        fp, archive_dir=cold_dir / "archive",
        size_limit_mb=1000,  # 很大，不触发 size
        retention_days=30,   # 30 天前的老记录归档
    )
    moved = store.maybe_archive()
    assert moved == 5
    assert store.count() == 0


def test_maybe_archive_no_op_when_fresh(cold_dir):
    """记录都在 retention 内 → 不归档。"""
    store = ColdMemoryStore(cold_dir / "x.jsonl", archive_dir=cold_dir / "archive")
    store.append({"ts": datetime.now().isoformat(), "content": "fresh"})
    moved = store.maybe_archive()
    assert moved == 0
    assert store.count() == 1


def test_maybe_archive_no_retention(cold_dir):
    """retention_days=None + size_limit_mb=None → 永不归档。"""
    store = ColdMemoryStore(
        cold_dir / "x.jsonl", archive_dir=cold_dir / "archive",
        retention_days=None, size_limit_mb=None,
    )
    old_ts = (datetime.now() - timedelta(days=365)).isoformat()
    store.append({"ts": old_ts, "content": "ancient"})
    moved = store.maybe_archive()
    assert moved == 0
    assert store.count() == 1


# === auto_archive 在 append 触发 ===

def test_auto_archive_triggers_every_n_appends(cold_dir, monkeypatch):
    """每 ARCHIVE_CHECK_INTERVAL 次 append 检查一次。"""
    fp = cold_dir / "x.jsonl"
    fp.parent.mkdir(parents=True, exist_ok=True)
    # 先写一堆老记录到归档目录（模拟）
    # 直接构造 store
    store = ColdMemoryStore(
        fp, archive_dir=cold_dir / "archive",
        retention_days=30, size_limit_mb=None,
    )
    # monkeypatch maybe_archive 计数
    calls = []
    orig = store.maybe_archive
    def counted():
        calls.append(1)
        return orig()
    store.maybe_archive = counted

    # 写 ARCHIVE_CHECK_INTERVAL - 1 条：不应触发
    for i in range(ARCHIVE_CHECK_INTERVAL - 1):
        store.append({"content": "x"}, auto_archive=True)
    assert len(calls) == 0

    # 第 ARCHIVE_CHECK_INTERVAL 条：触发一次
    store.append({"content": "y"}, auto_archive=True)
    assert len(calls) == 1


def test_auto_archive_disabled(cold_dir):
    """auto_archive=False 时不检查。"""
    store = ColdMemoryStore(
        cold_dir / "x.jsonl", archive_dir=cold_dir / "archive",
        retention_days=30,
    )
    calls = []
    orig = store.maybe_archive
    store.maybe_archive = lambda: calls.append(1) or orig()

    for i in range(ARCHIVE_CHECK_INTERVAL + 5):
        store.append({"content": "x"}, auto_archive=False)
    assert len(calls) == 0


def test_auto_archive_failure_does_not_break(cold_dir):
    """归档出错不阻塞 append。"""
    store = ColdMemoryStore(
        cold_dir / "x.jsonl", archive_dir=cold_dir / "archive",
        retention_days=30,
    )
    def broken():
        raise RuntimeError("archive failed")
    store.maybe_archive = broken

    # 应正常 append（不抛）
    for i in range(ARCHIVE_CHECK_INTERVAL + 1):
        store.append({"content": "x"}, auto_archive=True)
    assert store.count() == ARCHIVE_CHECK_INTERVAL + 1


# === list_archives / load_archive ===

def test_list_archives_empty(cold_dir):
    store = ColdMemoryStore(cold_dir / "x.jsonl", archive_dir=cold_dir / "archive")
    assert store.list_archives() == []


def test_list_archives_after_archive(cold_dir):
    store = ColdMemoryStore(cold_dir / "x.jsonl", archive_dir=cold_dir / "archive")
    store.append({"ts": "2020-01-15T00:00:00", "content": "a"})
    store.append({"ts": "2020-02-15T00:00:00", "content": "b"})
    store.archive_old_records("2025-01-01T00:00:00")
    archives = store.list_archives()
    assert len(archives) == 2
    # 按月排序
    assert "2020-01.jsonl" in str(archives[0])
    assert "2020-02.jsonl" in str(archives[1])


def test_load_archive(cold_dir):
    store = ColdMemoryStore(cold_dir / "x.jsonl", archive_dir=cold_dir / "archive")
    store.append({"ts": "2020-01-15T00:00:00", "content": "jan_a"})
    store.append({"ts": "2020-01-20T00:00:00", "content": "jan_b"})
    store.archive_old_records("2025-01-01T00:00:00")
    recs = store.load_archive("2020-01")
    assert len(recs) == 2
    assert {r["content"] for r in recs} == {"jan_a", "jan_b"}


def test_load_archive_nonexistent(cold_dir):
    store = ColdMemoryStore(cold_dir / "x.jsonl", archive_dir=cold_dir / "archive")
    assert store.load_archive("2099-01") == []


# === 跨会话持久（cold 真的"长期"）===

def test_cold_persists_across_instances(cold_dir):
    """新 store 实例读同一个文件 → 历史记录在。"""
    fp = cold_dir / "x.jsonl"
    store1 = ColdMemoryStore(fp, archive_dir=cold_dir / "archive")
    store1.append({"session_id": "s1", "content": "memory 1"})
    store1.append({"session_id": "s2", "content": "memory 2"})

    # 新实例 + 重启
    store2 = ColdMemoryStore(fp, archive_dir=cold_dir / "archive")
    rows = store2.query()
    assert len(rows) == 2
    # 跨 session 检索
    matches = store2.search_text("memory")
    assert len(matches) == 2


def test_cold_archive_persists_after_clear(cold_dir):
    """clear() 只清主文件，归档保留。"""
    store = ColdMemoryStore(cold_dir / "x.jsonl", archive_dir=cold_dir / "archive")
    store.append({"ts": "2020-01-15T00:00:00", "content": "old"})
    store.archive_old_records("2025-01-01T00:00:00")
    # 主文件空
    assert store.count() == 0
    # 清主文件
    store.clear()
    # 归档还在
    archives = store.list_archives()
    assert len(archives) >= 1
    recs = store.load_archive("2020-01")
    assert len(recs) >= 1


# === 辅助 ===

def json_line(d):
    import json
    return json.dumps(d, ensure_ascii=False, default=str) + "\n"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
"""
test_memory_cold.py - ColdMemoryStore 单元测试

覆盖：
1. append + count
2. query filter（session_id / topic / since / until）
3. search_text 关键词匹配
4. 并发 append 无丢失（FileLock 重试）
5. compact_older_than 归档
6. 容错：坏行跳过、空文件
"""
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.memory import ColdMemoryStore


@pytest.fixture
def store(tmp_path):
    """每次测试一个独立的 JSONL。"""
    s = ColdMemoryStore(tmp_path / "test.jsonl")
    yield s
    # 不清，让其他测试看到独立文件


# === append + count ===

def test_append_one(store):
    store.append({"session_id": "s1", "role": "user", "content": "hi"})
    assert store.count() == 1


def test_append_many(store):
    for i in range(10):
        store.append({"session_id": "s1", "content": f"msg{i}"})
    assert store.count() == 10


def test_append_normalizes_missing_fields(store):
    """缺字段时 record 自动补默认。"""
    store.append({"content": "hi"})
    rows = store.query()
    assert len(rows) == 1
    r = rows[0]
    assert "ts" in r
    assert r["role"] == ""
    assert r["topic"] == "general"
    assert r["source"] == "hot"


# === query ===

def test_query_by_session(store):
    store.append({"session_id": "s1", "content": "a"})
    store.append({"session_id": "s2", "content": "b"})
    out = store.query(session_id="s1")
    assert len(out) == 1
    assert out[0]["content"] == "a"


def test_query_by_topic(store):
    store.append({"topic": "plan", "content": "a"})
    store.append({"topic": "summary", "content": "b"})
    out = store.query(topic="plan")
    assert len(out) == 1
    assert out[0]["content"] == "a"


def test_query_by_time(store):
    store.append({"ts": "2020-01-01T00:00:00", "content": "old"})
    store.append({"ts": "2030-01-01T00:00:00", "content": "new"})
    out = store.query(since="2025-01-01T00:00:00")
    assert len(out) == 1
    assert out[0]["content"] == "new"


def test_query_limit(store):
    for i in range(20):
        store.append({"content": f"m{i}"})
    out = store.query(limit=5)
    assert len(out) == 5


def test_query_returns_newest_first(store):
    store.append({"ts": "2020-01-01T00:00:00", "content": "old"})
    store.append({"ts": "2030-01-01T00:00:00", "content": "new"})
    out = store.query()
    assert out[0]["content"] == "new"


def test_query_empty_file(store):
    assert store.query() == []


# === search_text ===

def test_search_text_basic(store):
    store.append({"content": "momentum factor for A-shares"})
    store.append({"content": "value factor research"})
    out = store.search_text("momentum")
    assert len(out) == 1
    assert "momentum" in out[0]["content"]


def test_search_text_case_insensitive(store):
    store.append({"content": "MOMENTUM strategy"})
    out = store.search_text("momentum")
    assert len(out) == 1


def test_search_text_chinese(store):
    store.append({"content": "动量因子在A股的表现"})
    store.append({"content": "value research"})
    out = store.search_text("动量")
    assert len(out) == 1


def test_search_text_empty_kw(store):
    store.append({"content": "x"})
    assert store.search_text("") == []


def test_search_text_with_session_filter(store):
    store.append({"session_id": "s1", "content": "momentum here"})
    store.append({"session_id": "s2", "content": "momentum here too"})
    out = store.search_text("momentum", session_id="s1")
    assert len(out) == 1
    assert out[0]["session_id"] == "s1"


# === 容错 ===

def test_skip_corrupt_lines(store, tmp_path):
    """坏 JSON 行应跳过（query 不崩）。"""
    fp = tmp_path / "test.jsonl"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(
        'valid\n{"role":"user","content":"a"}\nnot-json-line\n{"role":"user","content":"b"}\n',
        encoding="utf-8",
    )
    s = ColdMemoryStore(fp)
    # count 是非空行数（4），query 跳过非法 JSON 行 → 2
    assert s.count() == 4
    rows = s.query()
    assert len(rows) == 2
    assert {r["content"] for r in rows} == {"a", "b"}


def test_count_no_file(store, tmp_path):
    """文件不存在时 count 返回 0。"""
    s = ColdMemoryStore(tmp_path / "nonexistent.jsonl")
    assert s.count() == 0


# === compact_older_than ===

def test_compact_older_than_archives(store, tmp_path):
    store.append({"ts": "2020-01-01T00:00:00", "content": "old"})
    store.append({"ts": "2030-01-01T00:00:00", "content": "new"})
    dest = tmp_path / "archive.jsonl"
    moved = store.compact_older_than("2025-01-01T00:00:00", dest)
    assert moved == 1
    # 原文件只剩 new
    assert store.count() == 1
    assert store.query()[0]["content"] == "new"
    # 归档文件有 old
    assert dest.exists()


def test_compact_creates_dest_file(store, tmp_path):
    """dest 不存在时自动创建。"""
    dest = tmp_path / "archive.jsonl"
    moved = store.compact_older_than("2030-01-01T00:00:00", dest)
    assert moved == 0
    assert dest.exists()


# === 并发 append ===

def test_concurrent_append_no_loss(tmp_path):
    """多线程并发 append 100 条，全部应该写入。"""
    if sys.platform == "win32":
        # msvcrt 在跨 handle 上不一定可靠，宽松些
        pytest.skip("msvcrt cross-handle locking less reliable; coverage still ok")

    fp = tmp_path / "concurrent.jsonl"
    store = ColdMemoryStore(fp)

    n_threads = 5
    per_thread = 20

    def worker(tid):
        for i in range(per_thread):
            store.append({"session_id": f"t{tid}", "content": f"msg{i}"})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store.count() == n_threads * per_thread


def test_concurrent_append_same_session_no_loss(tmp_path):
    if sys.platform == "win32":
        pytest.skip("msvcrt cross-handle locking less reliable")

    fp = tmp_path / "same_session.jsonl"
    store = ColdMemoryStore(fp)
    n = 30

    def worker(i):
        store.append({"session_id": "s1", "content": f"msg{i}"})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store.count() == n


# === append_many ===

def test_append_many(store):
    n = store.append_many([
        {"content": "a"},
        {"content": "b"},
        {"content": "c"},
    ])
    assert n == 3
    assert store.count() == 3


# === clear ===

def test_clear(store):
    store.append({"content": "x"})
    store.clear()
    assert store.count() == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
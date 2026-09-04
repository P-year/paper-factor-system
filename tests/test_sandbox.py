"""
test_sandbox.py - 文件路径沙箱单元测试

覆盖：
1. allowed_roots 内的路径允许
2. allowed_roots 外的路径拒绝
3. 跨路径遍历攻击（../）阻止
4. 符号链接解析
5. safe_write_text / safe_read_text 实际写入检查
6. 默认 sandbox 配置正确
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.sandbox import (
    PathSandbox,
    SandboxViolation,
    get_default_sandbox,
    reset_sandbox,
    safe_write_text,
    safe_write_json,
    safe_read_text,
    safe_append_text,
)


# === 基本检查 ===

def test_allowed_path_passes(tmp_path):
    sb = PathSandbox([tmp_path])
    assert sb.is_safe(tmp_path / "subdir" / "file.txt") is True


def test_disallowed_path_rejected(tmp_path):
    sb = PathSandbox([tmp_path / "data"])
    assert sb.is_safe(Path("/etc/passwd")) is False


def test_assert_safe_write_raises_outside():
    sb = PathSandbox([Path("/tmp/allowed")])
    with pytest.raises(SandboxViolation) as exc:
        sb.assert_safe_write(Path("/etc/passwd"))
    assert exc.value.mode == "write"


def test_assert_safe_read_raises_outside():
    sb = PathSandbox([Path("/tmp/allowed")])
    with pytest.raises(SandboxViolation) as exc:
        sb.assert_safe_read(Path("/etc/passwd"))
    assert exc.value.mode == "read"


# === 路径遍历攻击 ===

def test_path_traversal_blocked(tmp_path):
    """../../../etc/passwd 应被阻止。"""
    sb = PathSandbox([tmp_path])
    evil = tmp_path / ".." / ".." / "etc" / "passwd"
    assert sb.is_safe(evil) is False


def test_symlink_resolution(tmp_path):
    """符号链接指向 allowed 之外的位置应被阻止。"""
    import os
    sb = PathSandbox([tmp_path / "allowed"])

    # 创建符号链接：allowed/symlink -> /tmp/evil
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    evil_dir = tmp_path / "evil"
    evil_dir.mkdir()
    evil_file = evil_dir / "secret.txt"
    evil_file.write_text("secret")

    symlink = allowed_dir / "symlink"
    if os.name != "nt":
        symlink.symlink_to(evil_file)
        assert sb.is_safe(symlink) is False
    else:
        # Windows 上 symlink 要权限，这里跳过
        pytest.skip("symlink test skipped on Windows")


# === 默认 sandbox ===

def test_default_sandbox_allows_data():
    sb = get_default_sandbox()
    from harness.paths import DATA_DIR, MEMORY_DIR, OUTPUT_DIR
    assert sb.is_safe(DATA_DIR / "papers" / "test.json")
    assert sb.is_safe(MEMORY_DIR / "sessions" / "x.json")
    assert sb.is_safe(OUTPUT_DIR / "report.md")


def test_default_sandbox_blocks_system():
    sb = get_default_sandbox()
    assert sb.is_safe(Path("/etc/passwd")) is False
    assert sb.is_safe(Path("C:/Windows/System32/drivers/etc/hosts")) is False


def test_reset_sandbox():
    reset_sandbox()
    sb1 = get_default_sandbox()
    reset_sandbox()
    sb2 = get_default_sandbox()
    assert sb1 is not sb2


# === 便捷 wrapper ===

def test_safe_write_text(tmp_path):
    """safe_write_text 能写入 allowed 内的路径。"""
    sb = PathSandbox([tmp_path])
    # monkey-patch
    import harness.sandbox
    harness.sandbox._default_sandbox = sb

    fp = tmp_path / "x.txt"
    safe_write_text(fp, "hello")
    assert fp.read_text() == "hello"


def test_safe_write_text_violation_raises(tmp_path):
    sb = PathSandbox([tmp_path])
    import harness.sandbox
    harness.sandbox._default_sandbox = sb

    with pytest.raises(SandboxViolation):
        safe_write_text(Path("/etc/passwd"), "evil")


def test_safe_write_json(tmp_path):
    sb = PathSandbox([tmp_path])
    import harness.sandbox
    harness.sandbox._default_sandbox = sb

    fp = tmp_path / "x.json"
    safe_write_json(fp, {"a": 1, "b": [2, 3]})
    import json
    data = json.loads(fp.read_text())
    assert data == {"a": 1, "b": [2, 3]}


def test_safe_append_text(tmp_path):
    sb = PathSandbox([tmp_path])
    import harness.sandbox
    harness.sandbox._default_sandbox = sb

    fp = tmp_path / "log.txt"
    safe_append_text(fp, "line1\n")
    safe_append_text(fp, "line2\n")
    assert fp.read_text() == "line1\nline2\n"


def test_safe_read_text(tmp_path):
    sb = PathSandbox([tmp_path])
    import harness.sandbox
    harness.sandbox._default_sandbox = sb

    fp = tmp_path / "x.txt"
    fp.write_text("content")
    assert safe_read_text(fp) == "content"


def test_safe_read_text_violation_raises(tmp_path):
    sb = PathSandbox([tmp_path])
    import harness.sandbox
    harness.sandbox._default_sandbox = sb

    with pytest.raises(SandboxViolation):
        safe_read_text(Path("/etc/shadow"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
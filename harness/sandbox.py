"""
harness/sandbox.py - 文件路径沙箱

目的：限制 tool 写入路径必须在 allowed_roots 内（默认 data/ + memory/ + output/）。
防止 tool bug 或 prompt injection 把文件写到不该写的位置。

用法：
    from harness.sandbox import get_default_sandbox, SandboxViolation

    sb = get_default_sandbox()
    sb.assert_safe_write(path)     # 写入前检查
    sb.assert_safe_read(path)      # 读取前检查（可选）

    # 直接用 wrapper：
    from harness.sandbox import safe_write_text
    safe_write_text(path, content)
"""
import os
from pathlib import Path
from typing import Iterable, Optional

from harness.paths import PROJECT_ROOT, DATA_DIR, MEMORY_DIR, OUTPUT_DIR


class SandboxViolation(RuntimeError):
    """文件路径不在 allowed_roots 内。"""
    def __init__(self, path: Path, allowed_roots: list, mode: str):
        self.path = Path(path)
        self.allowed_roots = allowed_roots
        self.mode = mode  # "read" | "write"
        super().__init__(
            f"sandbox {mode} violation: {path} not under any of "
            f"{[str(r) for r in allowed_roots]}"
        )


class PathSandbox:
    """路径白名单。allowed_roots 下的路径（含子目录）允许读写。"""

    def __init__(self, allowed_roots: Iterable[Path], *, allow_project_root: bool = True):
        self._allowed = [Path.resolve(p) for p in allowed_roots]
        if allow_project_root:
            self._allowed.append(PROJECT_ROOT.resolve())

    def is_safe(self, path: Path, *, mode: str = "write") -> bool:
        """检查路径是否在 allowed_roots 下。"""
        try:
            resolved = Path(path).resolve()
        except Exception:
            return False
        return any(
            self._is_under(resolved, root) for root in self._allowed
        )

    def _is_under(self, path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def assert_safe_write(self, path: Path) -> None:
        if not self.is_safe(path, mode="write"):
            raise SandboxViolation(path, self._allowed, "write")

    def assert_safe_read(self, path: Path) -> None:
        if not self.is_safe(path, mode="read"):
            raise SandboxViolation(path, self._allowed, "read")


# === 默认 sandbox ===

_default_sandbox: Optional[PathSandbox] = None


def get_default_sandbox() -> PathSandbox:
    """默认 sandbox：允许 data/ + memory/ + output/ + 项目根（限 env 配置覆盖）。"""
    global _default_sandbox
    if _default_sandbox is None:
        _default_sandbox = PathSandbox(
            allowed_roots=[DATA_DIR, MEMORY_DIR, OUTPUT_DIR],
            allow_project_root=True,
        )
    return _default_sandbox


def reset_sandbox() -> None:
    """清空（测试用）。"""
    global _default_sandbox
    _default_sandbox = None


# === 便捷函数 ===

def safe_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """写入文件前检查路径合法。"""
    get_default_sandbox().assert_safe_write(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding=encoding)


def safe_write_json(path: Path, obj, *, encoding: str = "utf-8", indent: int = 2) -> None:
    """写 JSON 前检查。"""
    import json
    safe_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent, default=str), encoding=encoding)


def safe_read_text(path: Path, *, encoding: str = "utf-8") -> str:
    """读取前检查。"""
    get_default_sandbox().assert_safe_read(path)
    return Path(path).read_text(encoding=encoding)


def safe_append_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """追加前检查。"""
    get_default_sandbox().assert_safe_write(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding=encoding) as f:
        f.write(content)
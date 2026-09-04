"""DEPRECATED: use harness.paths directly.

Phase 0 兼容垫片。新代码请 from harness.paths import ...
"""
from harness.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    PAPER_DIR,
    FACTOR_DB_PATH,
    MEMORY_DIR,
    SESSIONS_DIR,
    RUNS_DIR,
    AUDIT_DIR,
    OUTPUT_DIR,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "PAPER_DIR",
    "FACTOR_DB_PATH",
    "MEMORY_DIR",
    "SESSIONS_DIR",
    "RUNS_DIR",
    "AUDIT_DIR",
    "OUTPUT_DIR",
]
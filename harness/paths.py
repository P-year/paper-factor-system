"""
harness/paths.py - 共享的项目根路径工具

所有 harness/* 与 pipelines/* 文件用 from harness.paths import PROJECT_ROOT
不再各自计算 __file__.parent.parent...

Phase 0：原 agent/paths.py 搬运而来。Phase 1 之后，agent/paths.py 改为 re-export。
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 便捷子路径
DATA_DIR = PROJECT_ROOT / "data"
PAPER_DIR = DATA_DIR / "papers"
FACTOR_DB_PATH = DATA_DIR / "factor_candidates.json"
MEMORY_DIR = PROJECT_ROOT / "memory"
SESSIONS_DIR = MEMORY_DIR / "sessions"
RUNS_DIR = MEMORY_DIR / "runs"
AUDIT_DIR = MEMORY_DIR / "audit"
RAG_DIR = MEMORY_DIR / "rag"            # v5：本地论文 RAG 向量库
OUTPUT_DIR = PROJECT_ROOT / "output"

for d in (DATA_DIR, PAPER_DIR, MEMORY_DIR, SESSIONS_DIR, RUNS_DIR, AUDIT_DIR, RAG_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)
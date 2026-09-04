"""
conftest.py - pytest 配置

加项目根到 sys.path，使 `from harness.x import ...` 可用。
避免 pytest 把项目根的 __init__.py 当成 test module 收集。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
"""
harness/eval/ - 通用 eval 框架

目标：让"测一个 pipeline"变成"在 json 里写 case + type 字段"。

公共 API：
    from harness.eval import run_case, run_cases, run_cases_from_file
    from harness.eval import judge_output
    from harness.eval.case_schema import normalize_case, CaseType
"""

from harness.eval.case_schema import (
    CaseType,
    normalize_case,
    normalize_cases,
)
from harness.eval.runner import (
    run_case,
    run_cases,
    run_cases_from_file,
    generate_report,
)
from harness.eval.judge import judge_output
from harness.eval.paper_loader import (
    normalize_paper_input,
    normalize_papers,
    write_papers_to_collection,
)

__all__ = [
    "CaseType",
    "normalize_case",
    "normalize_cases",
    "run_case",
    "run_cases",
    "run_cases_from_file",
    "generate_report",
    "judge_output",
    "normalize_paper_input",
    "normalize_papers",
    "write_papers_to_collection",
]
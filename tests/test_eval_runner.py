"""
test_eval_runner.py - eval runner 类型分发测试

覆盖：
1. normalize_case 校验必填字段
2. full type 默认
3. inject_papers type 必须有 input_papers
4. runner.run_case / run_cases_from_file 能加载并跑 case 文件
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from harness.eval.case_schema import normalize_case, normalize_cases, CaseType


# === normalize_case ===

def test_normalize_old_case_defaults_to_full():
    """旧 case 无 type 字段时默认 full。"""
    case = {
        "id": "test",
        "goal": "test goal",
        "judge_criteria": ["c1"],
    }
    nc = normalize_case(case)
    assert nc["type"] == CaseType.FULL.value
    assert nc["pipeline"] == "paper_factor"
    assert nc["weight"] == 1.0


def test_normalize_inject_papers_requires_input():
    """inject_papers 必须有 input_papers。"""
    case = {
        "id": "test",
        "type": "inject_papers",
        "goal": "test",
        "judge_criteria": ["c1"],
    }
    with pytest.raises(ValueError, match="input_papers"):
        normalize_case(case)


def test_normalize_inject_papers_default_skip_nodes():
    """inject_papers 默认 skip_nodes=['collect']。"""
    case = {
        "id": "test",
        "type": "inject_papers",
        "goal": "test",
        "judge_criteria": ["c1"],
        "input_papers": [{"source": "dict", "title": "T", "summary": "S"}],
    }
    nc = normalize_case(case)
    assert nc["skip_nodes"] == ["collect"]


def test_normalize_inject_papers_validates_paper_format():
    """inject_papers 的 input_papers 每条必须有 source。"""
    case = {
        "id": "test",
        "type": "inject_papers",
        "goal": "test",
        "judge_criteria": ["c1"],
        "input_papers": [{"title": "T", "summary": "S"}],  # 没 source
    }
    with pytest.raises(ValueError, match="source"):
        normalize_case(case)


def test_normalize_unknown_type_raises():
    case = {"id": "test", "type": "weird", "goal": "g", "judge_criteria": ["c"]}
    with pytest.raises(ValueError, match="Unknown case type"):
        normalize_case(case)


def test_normalize_missing_id_raises():
    case = {"goal": "g", "judge_criteria": ["c"]}
    with pytest.raises(ValueError, match="id"):
        normalize_case(case)


def test_normalize_missing_judge_criteria_raises():
    case = {"id": "test", "goal": "g"}
    with pytest.raises(ValueError, match="judge_criteria"):
        normalize_case(case)


def test_normalize_skip_approve_sets_flag():
    case = {
        "id": "test",
        "type": "skip_approve",
        "goal": "g",
        "judge_criteria": ["c"],
    }
    nc = normalize_case(case)
    assert nc["skip_decide"] is True


# === normalize_cases ===

def test_normalize_cases_list():
    cases = normalize_cases([
        {"id": "a", "goal": "ga", "judge_criteria": ["c"]},
        {"id": "b", "goal": "gb", "judge_criteria": ["c"]},
    ])
    assert len(cases) == 2
    assert all(c["type"] == "full" for c in cases)


# === case 文件加载 ===

def test_load_full_cases_file():
    """harness/eval/cases/full.json 加载正确。"""
    fp = PROJECT_ROOT / "harness" / "eval" / "cases" / "full.json"
    assert fp.exists(), f"full.json missing: {fp}"
    data = json.loads(fp.read_text(encoding="utf-8"))
    cases = normalize_cases(data["test_cases"])
    assert len(cases) == 5
    assert all(c["type"] == "full" for c in cases)


def test_load_inject_papers_cases_file():
    """harness/eval/cases/inject_papers.json 加载正确。"""
    fp = PROJECT_ROOT / "harness" / "eval" / "cases" / "inject_papers.json"
    assert fp.exists(), f"inject_papers.json missing: {fp}"
    data = json.loads(fp.read_text(encoding="utf-8"))
    cases = normalize_cases(data["test_cases"])
    assert len(cases) >= 3
    assert all(c["type"] == "inject_papers" for c in cases)


# === runner 接口 ===

def test_runner_imports():
    """runner 模块导入无错。"""
    from harness.eval.runner import run_case, run_cases, run_cases_from_file, generate_report
    assert callable(run_case)
    assert callable(run_cases)
    assert callable(run_cases_from_file)
    assert callable(generate_report)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
"""
harness/eval/case_schema.py - eval case schema + normalize

支持的 case type：
    full           从 collect 开始的全流程（默认，兼容旧版）
    inject_papers  注入 paper 到 state，跳过 skip_nodes（默认含 collect）
    skip_approve   全流程但不挂 decide（scheduler 模式）

字段：
    id             必填，用例 ID
    type           可选，默认 full
    pipeline       可选，默认 paper_factor
    goal           必填，目标描述
    input_papers   inject_papers 时必填，paper 列表
    skip_nodes     可选，inject_papers 时默认 ["collect"]
    expected_steps 可选，期望访问的节点列表（用于 trace 校验）
    min_papers     可选，期望最少 paper 数
    min_factors    可选，期望最少 factor 数
    judge_criteria 必填，评判标准列表
    weight         可选，默认 1.0
"""
from enum import Enum
from typing import Any, Dict, List


class CaseType(str, Enum):
    FULL = "full"
    INJECT_PAPERS = "inject_papers"
    SKIP_APPROVE = "skip_approve"


VALID_TYPES = {t.value for t in CaseType}


def normalize_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """补全默认值 + 校验必填字段。返回新 dict（不修改入参）。"""
    out = dict(case)

    # type 默认 full
    ctype = out.get("type", CaseType.FULL.value)
    if ctype not in VALID_TYPES:
        raise ValueError(f"Unknown case type: {ctype!r}. Valid: {VALID_TYPES}")
    out["type"] = ctype

    # pipeline 默认 paper_factor
    out.setdefault("pipeline", "paper_factor")

    # id 必填
    if "id" not in out:
        raise ValueError(f"case missing 'id': {out}")

    # goal 必填
    if "goal" not in out:
        raise ValueError(f"case {out['id']!r} missing 'goal'")

    # judge_criteria 必填且非空
    criteria = out.get("judge_criteria")
    if not criteria or not isinstance(criteria, list):
        raise ValueError(f"case {out['id']!r} missing 'judge_criteria' (list)")

    # inject_papers 必须有 input_papers
    if ctype == CaseType.INJECT_PAPERS.value:
        papers = out.get("input_papers")
        if not papers:
            raise ValueError(f"case {out['id']!r} type=inject_papers requires 'input_papers'")
        # 校验每条 paper 都有 source 字段
        for i, p in enumerate(papers):
            if not isinstance(p, dict) or "source" not in p:
                raise ValueError(
                    f"case {out['id']!r} input_papers[{i}] invalid: "
                    f"need dict with 'source' field"
                )
        # skip_nodes 默认含 collect
        if "skip_nodes" not in out:
            out["skip_nodes"] = ["collect"]

    # skip_approve 默认跳过 decide（让 graph 走 scheduler 模式）
    if ctype == CaseType.SKIP_APPROVE.value:
        out.setdefault("skip_decide", True)

    # weight 默认 1.0
    out.setdefault("weight", 1.0)

    # min_papers / min_factors 默认 0
    out.setdefault("min_papers", 0)
    out.setdefault("min_factors", 0)

    # expected_steps 默认空
    out.setdefault("expected_steps", [])

    return out


def normalize_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_case(c) for c in cases]
"""
harness/lineage.py - 因子 lineage 追踪

目的：每个 factor / quality / backtest 都打 metadata 标签，建立从 input paper → factor → 回测 的因果链。

字段（写入 factor["_lineage"] / quality_result["_lineage"] / backtest_result["_lineage"]）：
    source_paper_id       str   来源 arxiv_id（或 paper_link）
    source_paper_title    str
    extract_iteration     int   提取时的 iteration_count
    extract_timestamp     ISO 时间戳
    quality_iteration     int   quality_check 时的 iteration_count
    quality_method        str   "heuristic" / "llm" / "llm_failed"
    backtest_iteration    int   回测时的 iteration_count
    backtest_timestamp    ISO 时间戳

用法：
    from harness.lineage import tag_factor_lineage, tag_quality_lineage, tag_backtest_lineage

    tag_factor_lineage(factor, state, source_paper={"arxiv_id": "x", "title": "T"})
    tag_quality_lineage(quality_result, state)
    tag_backtest_lineage(backtest_result, state)
"""
from datetime import datetime
from typing import Any, Dict, Optional


def _now_iso() -> str:
    return datetime.now().isoformat()


def tag_factor_lineage(
    factor: Dict[str, Any],
    state: Dict[str, Any],
    source_paper: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """给 factor 加 lineage metadata。"""
    factor["_lineage"] = {
        "source_paper_id": (source_paper or {}).get("arxiv_id") or factor.get("paper_link", ""),
        "source_paper_title": (source_paper or {}).get("title") or factor.get("paper_title", ""),
        "extract_iteration": state.get("iteration_count", 0),
        "extract_timestamp": _now_iso(),
        "pipeline_name": state.get("pipeline_name", ""),
        "session_id": state.get("session_id", ""),
    }
    return factor


def tag_quality_lineage(
    quality_result: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """给 quality_result 加 lineage metadata。"""
    quality_result["_lineage"] = {
        "quality_iteration": state.get("iteration_count", 0),
        "quality_method": quality_result.get("method", "unknown"),
        "quality_timestamp": _now_iso(),
        "pipeline_name": state.get("pipeline_name", ""),
    }
    return quality_result


def tag_backtest_lineage(
    backtest_result: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """给 backtest_result 加 lineage metadata。"""
    backtest_result["_lineage"] = {
        "backtest_iteration": state.get("iteration_count", 0),
        "backtest_timestamp": _now_iso(),
        "pipeline_name": state.get("pipeline_name", ""),
        "backtest_status": "success" if backtest_result.get("IC") is not None else "failed",
    }
    return backtest_result


def factor_lineage_summary(factor: Dict[str, Any]) -> Dict[str, Any]:
    """从 factor 提取 lineage 摘要（用于报告 / debug）。"""
    lineage = factor.get("_lineage", {})
    return {
        "source_paper_id": lineage.get("source_paper_id", "?"),
        "source_paper_title": lineage.get("source_paper_title", "?"),
        "extract_iteration": lineage.get("extract_iteration", "?"),
        "extract_timestamp": lineage.get("extract_timestamp", "?"),
    }


def full_lineage_chain(factor: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """汇总单个 factor 的完整 lineage（factor + quality + backtest）。"""
    fname = factor.get("factor_name", "?")
    quality = next(
        (r for r in state.get("quality_results", []) if r.get("factor_name") == fname),
        None,
    )
    backtest = next(
        (r for r in state.get("backtest_results", []) if r.get("original_factor_name") == fname),
        None,
    )

    chain = {
        "factor_name": fname,
        "factor_lineage": factor.get("_lineage", {}),
        "quality": quality.get("_lineage", {}) if quality else None,
        "backtest": backtest.get("_lineage", {}) if backtest else None,
    }
    return chain
"""
analysis_tools - 因子提取 + 数据可行性
复用：FactorAnalyzer (analyzer.py) / FACTOR_CONFIGS (data_fetcher.py) / FEASIBILITY_PROMPT (yaml)
"""
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from analyzer import FactorAnalyzer
from data_fetcher import FACTOR_CONFIGS


def _load_feasibility_prompt() -> str:
    """从 paper_factor/config.yaml 读取 FEASIBILITY_PROMPT。

    Phase 1：直接读 yaml，绕开 deprecated 的 agent.prompts.decision。
    """
    import yaml as _yaml
    yaml_path = Path(__file__).resolve().parent.parent.parent / "pipelines" / "paper_factor" / "config.yaml"
    with yaml_path.open(encoding="utf-8") as f:
        return _yaml.safe_load(f)["prompts"]["feasibility"]


FEASIBILITY_PROMPT = _load_feasibility_prompt()


def extract_factor_tool(paper: Dict[str, Any]) -> Dict[str, Any]:
    """
    用 LLM（GLM/DeepSeek）从单篇论文提取因子。

    Args:
        paper: 论文 dict（需含 title, summary, link 字段）

    Returns:
        {
          "is_factor": bool,
          "factor": {...} | None,
          "reason": str | None,
          "api_provider": "glm"|"deepseek",
          "duration_ms": int,
          "error": str | None
        }
    """
    try:
        analyzer = FactorAnalyzer(provider="auto")
        result = analyzer.extract_factor(paper)

        if result.error:
            return {
                "is_factor": False,
                "factor": None,
                "reason": None,
                "error": result.error,
                "api_provider": result.api_provider,
                "duration_ms": result.duration_ms,
            }

        if result.is_factor:
            factor = result.factor_data
            factor["paper_title"] = result.paper_title
            factor["paper_link"] = paper.get("link", "")
            factor["paper_abstract"] = paper.get("summary", "")
            return {
                "is_factor": True,
                "factor": factor,
                "reason": None,
                "api_provider": result.api_provider,
                "duration_ms": result.duration_ms,
                "error": None,
            }

        return {
            "is_factor": False,
            "factor": None,
            "reason": result.factor_data.get("reason", "not a factor paper"),
            "api_provider": result.api_provider,
            "duration_ms": result.duration_ms,
            "error": None,
        }
    except Exception as e:
        return {
            "is_factor": False,
            "factor": None,
            "reason": None,
            "error": str(e),
            "api_provider": "",
            "duration_ms": 0,
        }


def check_data_feasibility(factor: Dict[str, Any]) -> Dict[str, Any]:
    """
    检查候选因子在当前 AKShare 数据下是否可真实计算（启发式）。

    Args:
        factor: 候选因子 dict（含 factor_name, data_needed, definition）

    Returns:
        {
          "feasible": bool,
          "matching_factor": str | None,  # FACTOR_CONFIGS 中匹配的键
          "missing": List[str],
          "alternatives": List[str]
        }
    """
    factor_name = factor.get("factor_name", "").lower()
    data_needed = (factor.get("data_needed") or "").lower()
    definition = (factor.get("definition") or "").lower()

    # 1. 按名称匹配（FACTOR_CONFIGS 实际键：return_1m/return_3m/return_6m/return_1y/turnover_1m）
    matching = None
    if factor_name:  # 防御空字符串（"" in "x" 永远为 True）
        for key in FACTOR_CONFIGS.keys():
            if key in factor_name or factor_name in key:
                matching = key
                break

    # 2. 按数据需求关键词匹配（中文别名 → FACTOR_CONFIGS 键）
    # 注意：只对 factor_name 匹配（data_needed/definition 太长会误触发）
    if not matching:
        alias_map = {
            # 动量（factor_name 专用，data_needed 里的"动量指标"是概念引用）
            "1月收益": "return_1m",
            "1个月收益": "return_1m",
            "近1月": "return_1m",
            "3月收益": "return_3m",
            "近3月": "return_3m",
            "6月收益": "return_6m",
            "近6月": "return_6m",
            "1年收益": "return_1y",
            "近1年": "return_1y",
            "1月换手": "turnover_1m",
            "近1月换手": "turnover_1m",
            "月均换手": "turnover_1m",
            "1月动量": "return_1m",
            "3月动量": "return_3m",
            "6月动量": "return_6m",
            "12月动量": "return_1y",
            "1年动量": "return_1y",
            "momentum": "return_1m",
            # 质量
            "roe": "ROE",
            "净资产收益率": "ROE",
            "毛利率": "gross_profit_margin",
            "销售毛利率": "gross_profit_margin",
            "净利率": "net_profit_margin",
            "销售净利率": "net_profit_margin",
            # 效率
            "总资产周转": "asset_turnover",
            "资产周转率": "asset_turnover",
            # 成长
            "营收同比": "revenue_growth_yoy",
            "营收增长": "revenue_growth_yoy",
            "营业总收入": "revenue_growth_yoy",
            "净利润同比": "profit_growth_yoy",
            "净利润增长": "profit_growth_yoy",
            "利润增速": "profit_growth_yoy",
        }
        # 短别名（≤3 字）只在 factor_name 里查；长别名三个字段都查
        for kw, key in alias_map.items():
            if len(kw) <= 3:
                if kw in factor_name:
                    matching = key
                    break
            else:
                if kw in factor_name or kw in data_needed or kw in definition:
                    matching = key
                    break

    # 3. 多模态判定
    is_multimodal = "新闻" in data_needed or "news" in data_needed or "llm" in data_needed

    if matching and matching in FACTOR_CONFIGS:
        return {
            "feasible": True,
            "matching_factor": matching,
            "missing": [],
            "alternatives": [],
            "multimodal": is_multimodal,
        }

    return {
        "feasible": False,
        "matching_factor": None,
        "missing": ["AKShare 暂未实现此因子"],
        "alternatives": [f"可参考 {k}" for k in list(FACTOR_CONFIGS.keys())[:3]],
        "multimodal": is_multimodal,
    }


def _build_factor_catalog() -> str:
    """构造给 LLM 看的因子目录（JSON 字符串）"""
    catalog = []
    for key, cfg in FACTOR_CONFIGS.items():
        catalog.append({
            "key": key,
            "name": cfg.get("name", ""),
            "category": cfg.get("category", ""),
            "data_source": cfg.get("data_source", ""),
            "formula": cfg.get("formula", ""),
        })
    return json.dumps(catalog, ensure_ascii=False, indent=2)


def llm_check_data_feasibility(factor: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM-based 可行性判断（启发式失败时升级用）。

    用途：当因子名/数据需求不在 alias 映射里，调用 LLM 决定：
    - 该因子能否用 11 个标准因子之一近似？
    - 近似用哪个？approximation 怎么算？
    - 还是真的不可行？

    Args:
        factor: 候选因子 dict

    Returns:
        {
          "feasible": bool,
          "matching_factor": str | None,   # 11 因子之一的 key
          "approximation": str,            # 怎么近似（仅 feasible=True 时有意义）
          "missing": List[str],
          "rationale": str,
          "method": "llm"
        }
    """
    try:
        from agent.nodes._llm import call_llm_json  # lazy import 避免循环

        factor_json = json.dumps(factor, ensure_ascii=False, indent=2)
        catalog_json = _build_factor_catalog()

        prompt = FEASIBILITY_PROMPT.format(
            factor_json=factor_json,
            catalog_json=catalog_json,
        )
        user_msg = "请评估这个候选因子的数据可行性，并尽量找一个最接近的标准因子来近似。严格输出 JSON。"

        result = call_llm_json(prompt, user_msg)

        matching = result.get("matching_factor")
        # 验证 LLM 说的 matching_factor 是否真实存在
        if matching and matching not in FACTOR_CONFIGS:
            matching = None

        return {
            "feasible": bool(result.get("feasible")) and matching is not None,
            "matching_factor": matching,
            "approximation": result.get("rationale", ""),  # LLM 没说怎么算，把 rationale 当说明
            "missing": result.get("missing", []),
            "alternatives": result.get("alternatives", []),
            "rationale": result.get("rationale", ""),
            "method": "llm",
        }
    except Exception as e:
        return {
            "feasible": False,
            "matching_factor": None,
            "approximation": "",
            "missing": [f"LLM check failed: {e}"],
            "alternatives": [],
            "rationale": f"LLM call failed: {e}",
            "method": "llm_failed",
        }
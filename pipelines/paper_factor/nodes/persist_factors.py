"""
persist_factors_node - 把 extracted_factors 写入 factor_candidates.json

调用时机：analyze 之后，quality_check 之前
作用：把 LLM 提取的因子入库（pending 状态），即使回测失败也保留记录
"""
from typing import Any, Dict

from pipelines.paper_factor.state import PaperFactorState
from harness.observability import observe_node


@observe_node("persist_factors_node")
def persist_factors_node(state: PaperFactorState) -> Dict[str, Any]:
    """把 extracted_factors 写入 factor 库"""
    from factor_db import FactorDatabase

    factors = state.get("extracted_factors", [])
    if not factors:
        return {"persisted_count": 0, "persist_attempted": True}

    try:
        db = FactorDatabase()
        added = db.add_factors(factors, auto_yes=True)
        db.save()
        return {
            "persisted_count": added,
            "persist_attempted": True,  # 关键：标记已尝试，不管 added 多少
            "errors": state.get("errors", []),
        }
    except Exception as e:
        return {
            "persisted_count": 0,
            "persist_attempted": True,
            "errors": state.get("errors", []) + [f"persist exception: {e}"],
        }
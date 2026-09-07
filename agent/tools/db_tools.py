"""
db_tools - 因子库 CRUD
复用：FactorDatabase (factor_db.py)
"""
from typing import Dict, List, Optional

from core.factor_db import FactorDatabase


def list_factors(
    status: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 50,
) -> Dict:
    """
    查询因子库。

    Args:
        status: pending / approved / active / rejected / deprecated / None(全部)
        keyword: 全文搜索关键词
        limit: 返回数量上限

    Returns:
        {"factors": [...], "count": N, "stats": {...}}
    """
    try:
        db = FactorDatabase()

        if keyword:
            factors = db.search(keyword)
        elif status:
            factors = db.get_by_status(status)
        else:
            factors = db.factors

        stats = db.get_statistics()

        return {
            "factors": factors[:limit],
            "count": len(factors[:limit]),
            "total_in_db": stats["total"],
            "stats": stats,
        }
    except Exception as e:
        return {"factors": [], "count": 0, "error": str(e)}


def update_factor_status(
    name: str,
    status: str,
    notes: str = "",
) -> Dict:
    """
    更新因子状态（pending → approved/rejected/active/deprecated）。

    Args:
        name: 因子名称
        status: 目标状态
        notes: 备注

    Returns:
        {"success": bool, "old_status": str|None, "new_status": str}
    """
    try:
        db = FactorDatabase()
        factor = db._find_by_name(name)
        old_status = factor.get("status") if factor else None

        if not factor:
            return {"success": False, "error": f"factor '{name}' not found"}

        success = db.update_status(name, status, notes)
        db.save()

        return {
            "success": success,
            "old_status": old_status,
            "new_status": status,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_report(
    fmt: str = "md",
    output_path: Optional[str] = None,
) -> Dict:
    """
    导出因子报告（Markdown 或 CSV）。

    Args:
        fmt: "md" | "csv"
        output_path: 自定义输出路径（None=默认 output/ 目录）

    Returns:
        {"path": str, "count": N}
    """
    try:
        db = FactorDatabase()
        if fmt == "md":
            if output_path:
                db.export_to_markdown(output_path)
                path = output_path
            else:
                from config import OUTPUT_DIR
                from datetime import datetime
                path = str(OUTPUT_DIR / f"factor_report_{datetime.now().strftime('%Y%m%d')}.md")
                db.export_to_markdown(path)
        elif fmt == "csv":
            if output_path:
                db.export_to_csv(output_path)
                path = output_path
            else:
                from config import OUTPUT_DIR
                from datetime import datetime
                path = str(OUTPUT_DIR / f"factors_{datetime.now().strftime('%Y%m%d')}.csv")
                db.export_to_csv(path)
        else:
            return {"path": "", "error": f"unknown format: {fmt}"}

        return {"path": path, "count": len(db.factors)}
    except Exception as e:
        return {"path": "", "error": str(e)}
"""
agent tools - 现有业务模块的 @tool 包装层

每个工具是一个普通 Python 函数，返回 JSON-serializable dict。
后续 LangChain @tool 装饰可以加在 TOOL_LIST 上。
"""
from .collection_tools import search_arxiv, read_paper
from .analysis_tools import extract_factor_tool, check_data_feasibility, llm_check_data_feasibility
from .backtest_tools import run_backtest
from .db_tools import list_factors, update_factor_status, generate_report

TOOL_LIST = [
    search_arxiv,
    read_paper,
    extract_factor_tool,
    check_data_feasibility,
    llm_check_data_feasibility,
    run_backtest,
    list_factors,
    update_factor_status,
    generate_report,
]

__all__ = [
    "TOOL_LIST",
    "search_arxiv",
    "read_paper",
    "extract_factor_tool",
    "check_data_feasibility",
    "llm_check_data_feasibility",
    "run_backtest",
    "list_factors",
    "update_factor_status",
    "generate_report",
]
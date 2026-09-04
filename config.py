"""
系统配置 - paper-factor-system
"""
import os
import sys
from pathlib import Path

# 加载项目级 .env（如果有）
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass

# ============================================================
# 基础路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
PAPER_DIR = DATA_DIR / "papers"
FACTOR_DB_PATH = DATA_DIR / "factor_candidates.json"

for _dir in [DATA_DIR, OUTPUT_DIR, PAPER_DIR]:
    _dir.mkdir(exist_ok=True)

# ============================================================
# API配置（支持多后端）
# ============================================================

# 智谱 GLM（主API，推荐）
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
GLM_MODEL = os.getenv("GLM_MODEL", "glm-4-flash")  # 便宜快速
GLM_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# DeepSeek（备选API，性价比高）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 回调配置
API_BACKOFF = 3  # API失败重试次数
API_TIMEOUT = 60  # API超时秒数
REQUEST_DELAY = 0.5  # 请求间隔（秒），避免触发限流

# ============================================================
# arXiv采集配置
# ============================================================
ARXIV_KEYWORDS = [
    "factor portfolio construction",
    "stock selection factor",
    "alpha factor machine learning",
    "quantitative factor investing",
    "asset pricing factor",
    "multimodal factor LLM newsflow",
    "stock return prediction factor",
]
ARXIV_CATEGORIES = ["q-fin.PR", "q-fin.ST", "q-fin.ML", "q-fin.CP"]
MAX_PAPERS_PER_SEARCH = 30
COLLECTION_DAYS = 90  # 采集近N天内的论文

# ============================================================
# Prompt模板
# ============================================================

# 标准因子提取Prompt
FACTOR_EXTRACT_PROMPT = """你是一个专业的量化因子研究员。你的任务是从学术论文中识别和提取选股因子。

论文信息：
标题：{title}
摘要：{abstract}

请判断这篇论文是否提出了新的选股因子。如果是，请提取以下信息（严格JSON格式）：
{{
  "is_factor": true,
  "factor_name": "因子名称（中文）",
  "definition": "因子定义和计算方式（详细描述）",
  "data_needed": "计算所需的数据（如：日频收益率、市值、成交量、财务指标等）",
  "calculation_formula": "计算公式（如有）",
  "market_scope": "适用市场（A股/美股/通用）",
  "paper_conclusion": "论文核心结论和回测效果",
  "confidence": "置信度（高/中/低）",
  "source": "来源论文标题",
  "paper_link": "论文链接"
}}

如果不是提出新因子，返回：
{{
  "is_factor": false,
  "reason": "不是因子的原因"
}}

输出必须是有效的JSON，不要包含其他文字。"""

# 多模态因子评估Prompt（新增）
MULTIMODAL_PROMPT = """你是一个量化多模态因子研究员。

给定候选因子和新闻流的组合信息，评估其在股票收益预测中的效果。

候选因子信息：
{factor_info}

新闻流特征摘要：
{news_summary}

请返回JSON（严格格式）：
{{
  "modality": "multimodal",
  "fusion_method": "combination",
  "news_contribution": "中",
  "expected_ic": 0.04,
  "rationale": "评估理由..."
}}

JSON之外不要输出任何文字。"""

# 因子质量评估Prompt（新增）
FACTOR_QUALITY_PROMPT = """你是一个量化因子质量评审员。

请评估以下因子的质量，判断其是否值得进入回测环节。

因子名称：{factor_name}
因子定义：{definition}
数据需求：{data_needed}
适用市场：{market_scope}
论文结论：{paper_conclusion}

请返回JSON（严格格式）：
{{
  "quality_score": 85,
  "strengths": ["优势1", "优势2"],
  "weaknesses": ["弱点1", "弱点2"],
  "recommendation": "值得回测 / 需谨慎 / 不建议"],
  "improvement_suggestions": "改进建议（如有）"
}}

JSON之外不要输出任何文字。"""

# ============================================================
# 回测配置
# ============================================================
BACKTEST_CONFIG = {
    "start_date": "20180101",
    "end_date": "20241231",
    "universe": "全A股",
    "weighting": "equal",
    "quantile_groups": 5,
    "rebalance_freq": "monthly",
    "min_icir": 0.3,          # ICIR阈值（低于此值过滤）
    "min_top_return": -20,    # 最小TOP组合年化收益率（%）
}

# ============================================================
# 多模态配置（新增）
# ============================================================
MULTIMODAL_CONFIG = {
    "enabled": True,
    "news_lookback_days": 30,    # 新闻回顾窗口（天）
    "min_news_per_stock": 3,       # 单只股票最少新闻数
    "llm_fine_tune": False,       # 是否微调LLM
    "fusion_method": "combination", # 默认融合方法
}

# ============================================================
# 工具函数
# ============================================================
def get_api_config(provider="glm"):
    """获取当前配置的API信息"""
    if provider == "glm":
        return {
            "api_key": GLM_API_KEY,
            "model": GLM_MODEL,
            "url": GLM_API_URL,
        }
    elif provider == "deepseek":
        return {
            "api_key": DEEPSEEK_API_KEY,
            "model": DEEPSEEK_MODEL,
            "url": DEEPSEEK_API_URL,
        }
    else:
        raise ValueError(f"Unknown provider: {provider}")

def check_api_keys():
    """检查已配置的API Keys"""
    configured = []
    if GLM_API_KEY:
        configured.append("GLM")
    if DEEPSEEK_API_KEY:
        configured.append("DeepSeek")
    return configured

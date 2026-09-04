"""
论文挖因子系统 - 增量输出
Paper Factor System - Incremental Output

本文档记录了针对论文 arXiv:2510.15691 的深度解读和系统对接方案

论文信息:
- 标题: Exploring the Synergy of Quantitative Factors and Newsflow Representations 
        from Large Language Models for Stock Return Prediction
- 作者: Tian Guo, Emmanuel Hauptmann (RAM Active Investments, Geneva)
- arXiv: https://arxiv.org/abs/2510.15691

对接产出文件:
- output/paper_analysis.md      → 论文深度解读（因子定义、模型架构、实验设计）
- output/code_search.md        → 开源代码搜索结果
- output/repro_design.md       → A股复现实验完整方案

本项目定位升级：
原有 Phase 1-3 规划侧重"论文→因子→回测"的单向上下文
论文 2510.15691 打开了新的能力维度：
  → 多模态融合模型（因子+LLM新闻）预测股票收益
  → 这意味着 analyzer.py 和 backtester.py 都可以扩展为多模态版本

因此在原规划基础上增加：
- Phase 1.5: 多模态因子分析（因子+新闻流联合输入）
- Phase 2.5: 增强回测（支持多模态信号的组合构建）
"""

# ============================================================
# 对接分析：现有模块 vs 论文能力映射
# ============================================================

"""
collector.py          ✅ 已有              论文采集 → 对接无障碍
analyzer.py           ⚠️ 需升级           单模态LLM分析 → 多模态融合分析
factor_db.py          ✅ 已有              因子存储 → 兼容news_representation字段
backtester.py        ⚠️ 需升级           理论估算回测 → 真实数据回测 + 多模态信号
run_pipeline.py       ✅ 已有              流程编排 → 扩展支持news相关参数
"""

# ============================================================
# 关键升级点1：analyzer.py 的多模态扩展
# ============================================================

"""
当前 analyzer.py 只接收论文摘要，提取纯因子。
论文2510.15691的核心创新是多模态融合：
  x_u = z(x_f, x_n)  ← 因子表示 + 新闻表示 → 统一表示 → 预测

升级方向（Phase 1.5）：
1. 新增因子分析场景：给定候选因子集合 + 新闻流，预测收益
2. 新增"因子+新闻"组合效果评估（不只评估单一因子）
3. 保持原有单因子分析流程不变，新增多模态分析分支

示例新增API：
analyzer.extract_multimodal_factor(factors, news_embedding) → 融合因子表示
analyzer.evaluate_news_contribution(factor, news) → 新闻增量贡献评估
"""

# ============================================================
# 关键升级点2：backtester.py 的真实数据支持
# ============================================================

"""
当前 backtester.py 只有理论估算（estimate_ic / estimate_group_return）
Phase 1 的目标是"初步筛选"，但论文证明了IC的重要性：
  - News Alone IC≈0（新闻单独没用）
  - Factors Alone IC≈0.018-0.049
  - FusionCombination IC≈0.031-0.060
  - MixtureDecoupled IC≈0.027-0.065

这意味着：
1. 必须引入真实行情数据（AKShare/Tushare）计算实际IC
2. 回测结果才是因子有效性的最终判断依据
3. 多模态信号的回测（因子+新闻）需要特殊处理

升级方向（Phase 2.5）：
1. 引入 AKShare 作为免费数据源
2. 实现真实IC计算：IC = corr(predicted_return, actual_return)
3. 多模态信号回测：news_representation作为额外的输入特征

Phase 2 的聚宽/米筐 API 作为可选升级（数据更全，但需付费）
"""

# ============================================================
# 关键升级点3：因子定义扩展（factor_candidates.json）
# ============================================================

"""
当前 factor_candidates.json 只有基础字段：
  - factor_name, definition, data_needed, calculation_formula
  - market_scope, paper_conclusion, confidence, source, paper_link

论文要求的扩展字段：
  - news_required: Boolean  # 该因子是否需要新闻流配合才有效
  - news_relevance: 高/中/低  # 新闻对因子的增量贡献度
  - modality: single/multimodal  # 单模态还是多模态因子
  - fusion_method: combination/summation/attention/mixture  # 推荐融合方法
  - llm_dependency: Boolean  # 是否依赖LLM生成的表示

示例扩展后的因子条目：
{
  "factor_name": "估值因子 + 新闻情绪融合",
  "definition": "EP_BP因子与LLM生成的新闻情绪表示拼接后过Dense层",
  "data_needed": "PE, PB等估值数据 + 公司新闻文本",
  "modality": "multimodal",
  "fusion_method": "combination",
  "news_required": true,
  "news_relevance": "中",
  "llm_dependency": true,
  "paper_conclusion": "北美市场Long-Only年化32.43%, 夏普1.0",
  "confidence": "高",
  "source": "arXiv:2510.15691"
}
"""

# ============================================================
# 实施路线图
# ============================================================

"""
【当前系统】 Phase 1 完成度较高
  ✅ collector.py    (arXiv + 本地PDF)
  ✅ analyzer.py     (GLM单因子提取)
  ✅ factor_db.py    (JSON因子库)
  ⚠️ backtester.py   (理论估算，非真实数据)
  ✅ run_pipeline.py (流程编排)

【Phase 1.5 — 多模态扩展】（论文核心能力对接）
  目标：让analyzer.py能处理"因子+新闻"组合分析
  依赖：GLM_API_KEY（已有）
  改动：
  - config.py 新增 MULTIMODAL_FACTOR_PROMPT
  - analyzer.py 新增 extract_multimodal_factor() 方法
  - output/paper_analysis.md → 作为分析参考基准

【Phase 2 — 真实数据回测】
  目标：让backtester.py能计算真实IC
  依赖：AKShare（免费，无需额外API Key）
  改动：
  - 新增 data_fetcher.py (AKShare数据获取)
  - backtester.py 扩展真实IC计算
  - run_pipeline.py 增加 --akshare 参数

【Phase 3 — 生产化】
  目标：全自动论文→因子→回测工厂
  依赖：定时任务 + 更全数据源
  改动：
  - cron定时采集arXiv新论文
  - 批量回测 + 因子池维护
  - 自动筛选 → 人工复核 → 入库
"""

# ============================================================
# 代码示例：Phase 1.5 级别的 analyzer 扩展
# ============================================================

"""
# config.py 新增
MULTIMODAL_FACTOR_PROMPT = """你是一个量化多模态因子研究员。

给定以下因子组合和新闻流表示，请评估：
1. 该组合在股票收益预测中的预期效果
2. 最优融合方法（combination/summation/attention/mixture）
3. 新闻对因子的增量贡献（高/中/低）

因子定义：
{factor_definitions}

新闻流特征：
{news_representation_summary}

请返回JSON：
{{
  "modality": "multimodal",
  "fusion_method": "combination",  // 或 summation/attention/mixture
  "news_contribution": "中",        // 高/中/低
  "expected_ic": 0.04,              // 预估IC
  "rationale": "为什么这个组合有效..."
}}

输出必须是有效JSON。"""

# analyzer.py 新增方法
def extract_multimodal_factor(self, factor: Dict, news_summary: str) -> Dict:
    prompt = MULTIMODAL_FACTOR_PROMPT.format(
        factor_definitions=factor.get("definition", ""),
        news_representation_summary=news_summary,
    )
    response = self.call_glm(prompt)
    result = json.loads(response)
    result["source_factor"] = factor.get("factor_name", "")
    return result
"""

# ============================================================
# 本次对接产出文件清单
# ============================================================

"""
output/
├── paper_analysis.md   # 论文全文解读（已生成）
├── code_search.md      # 开源代码搜索（已生成）
├── repro_design.md     # A股复现方案（已生成）
└── incremental_plan.md # 本文档 - 增量升级计划
"""

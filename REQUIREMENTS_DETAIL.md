# FactorLab — 需求颗粒度拆解与实现状态

> 详细需求清单，按功能模块逐项拆解，标注实现状态。
> 更新日期: 2026-10-21

---

## 状态图例

| 标记 | 含义 |
|------|------|
| ✅ | 已实现 |
| 🔧 | 部分实现 |
| ❌ | 未实现 |
| 🧪 | 有测试 |

---

## 1. 论文输入模块

### 1.1 PDF 上传 (FR-001)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 1.1.1 | `POST /papers/upload` API | ✅ | `api/paper_routes.py` |
| 1.1.2 | UploadFile 文件接收 | ✅ | `api/paper_routes.py` |
| 1.1.3 | PDF 格式校验 (.pdf) | ✅ | `api/paper_routes.py` |
| 1.1.4 | 大小限制 (≤20MB) | ✅ | `api/paper_routes.py` |
| 1.1.5 | pdfplumber 文本解析 (前5页) | ✅ | `harness/rag/pdf_loader.py` |
| 1.1.6 | 自动提取标题 | ✅ | `harness/rag/pdf_loader.py` |
| 1.1.7 | 存入论文库 | ✅ | `core/paper_library.py` |
| 1.1.8 | 来源标记 (pdf_upload) | ✅ | `api/paper_routes.py` |
| 1.1.9 | 前端上传按钮 | ✅ | `frontend/index.html` |
| 1.1.10 | 上传后自动解析+入库 | ✅ | `api/paper_routes.py` |
| 1.1.11 | 上传失败错误提示 | ✅ | `frontend/index.html` (Toast) |
| 1.1.12 | 批量上传 | ❌ | - |
| 1.1.13 | PDF 预览 | ❌ | - |

**实现度: 10/13 (77%)**

### 1.2 arXiv 搜索 (FR-002)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 1.2.1 | `POST /papers/search` API | ✅ | `api/paper_routes.py` |
| 1.2.2 | q-fin 分类搜索 | ✅ | `core/collector.py` |
| 1.2.3 | 关键词搜索 (7个) | ✅ | `core/collector.py` |
| 1.2.4 | 金融相关性过滤 | ✅ | `core/collector.py` |
| 1.2.5 | 标题去重 | ✅ | `core/collector.py` |
| 1.2.6 | 时间范围过滤 | ✅ | `core/collector.py` |
| 1.2.7 | 前端搜索框 | ✅ | `frontend/index.html` |
| 1.2.8 | 搜索结果展示 | ✅ | `frontend/index.html` |
| 1.2.9 | 一键导入论文库 | ✅ | `frontend/index.html` |
| 1.2.10 | 分页/排序 | 🔧 | 仅 max_results |
| 1.2.11 | SSRN/期刊搜索 | ❌ | - |
| 1.2.12 | 搜索历史 | ❌ | - |

**实现度: 9/12 (75%)**

### 1.3 论文库管理 (FR-003)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 1.3.1 | PaperLibrary 统一索引 | ✅ | `core/paper_library.py` |
| 1.3.2 | 论文去重 (arxiv_id) | ✅ | `core/paper_library.py` |
| 1.3.3 | `GET /papers` 列表 | ✅ | `api/paper_routes.py` |
| 1.3.4 | 分页 (limit/offset) | ✅ | `api/paper_routes.py` |
| 1.3.5 | 状态过滤 | ✅ | `api/paper_library.py` |
| 1.3.6 | 关键词搜索 | ✅ | `core/paper_library.py` |
| 1.3.7 | `GET /papers/{id}` 详情 | ✅ | `api/paper_routes.py` |
| 1.3.8 | 论文→因子关联查询 | ✅ | `core/factor_db.py::get_by_paper` |
| 1.3.9 | 状态更新 (new→indexed→extracted) | ✅ | `core/paper_library.py` |
| 1.3.10 | 论文统计 | ✅ | `core/paper_library.py::stats` |
| 1.3.11 | 前端论文列表 | ✅ | `frontend/index.html` |
| 1.3.12 | 前端提取因子按钮 | ✅ | `frontend/index.html` |
| 1.3.13 | 前端一键回测按钮 | ✅ | `frontend/index.html` |
| 1.3.14 | Legacy papers 导入 | ✅ | `core/paper_library.py::import_legacy_papers` |
| 1.3.15 | 论文删除 | ❌ | - |
| 1.3.16 | 论文编辑 | ❌ | - |
| 1.3.17 | 论文全文存储 | 🔧 | 仅摘要/前5页 |

**实现度: 14/17 (82%)**

---

## 2. 因子提取模块

### 2.1 LLM 因子提取 (FR-004)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 2.1.1 | DeepSeek 主调用 | ✅ | `core/analyzer.py` |
| 2.1.2 | GLM fallback | ✅ | `harness/llm_client.py` |
| 2.1.3 | FACTOR_EXTRACT_PROMPT | ✅ | `config.py` |
| 2.1.4 | 提取 factor_name | ✅ | `core/analyzer.py` |
| 2.1.5 | 提取 definition | ✅ | `core/analyzer.py` |
| 2.1.6 | 提取 calculation_formula | ✅ | `core/analyzer.py` |
| 2.1.7 | 提取 data_needed | ✅ | `core/analyzer.py` |
| 2.1.8 | 提取 confidence | ✅ | `core/analyzer.py` |
| 2.1.9 | 提取 market_scope | ✅ | `core/analyzer.py` |
| 2.1.10 | 置信度分级 (高/中/低) | ✅ | `core/analyzer.py` |
| 2.1.11 | 提取结果验证 | ✅ | `harness/verify.py` |
| 2.1.12 | 嵌套格式兼容 | ✅ | `harness/verify.py` |
| 2.1.13 | 多因子提取 (1篇多因子) | 🔧 | 仅单因子 |
| 2.1.14 | 提取失败重试 | ✅ | `harness/tool_call.py` |
| 2.1.15 | LLM 调用缓存 | 🔧 | 仅公式缓存 |

**实现度: 13/15 (87%)**

### 2.2 RAG 上下文增强 (FR-005)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 2.2.1 | paper_retrieve 节点 | ✅ | `pipelines/paper_factor/nodes/` |
| 2.2.2 | RAG 检索相似论文 | ✅ | `harness/rag/` |
| 2.2.3 | 注入提取 prompt | ✅ | `pipelines/paper_factor/nodes/analyze.py` |
| 2.2.4 | 标记 _used_rag_context | ✅ | `pipelines/paper_factor/nodes/analyze.py` |
| 2.2.5 | 标记 _rag_top_k | ✅ | `pipelines/paper_factor/nodes/analyze.py` |
| 2.2.6 | hard_rule 自动触发 | ✅ | `pipelines/paper_factor/config.yaml` |

**实现度: 6/6 (100%)**

---

## 3. 公式编译模块

### 3.1 公式归一化 (FR-006)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 3.1.1 | `normalize_formula()` 入口 | ✅ | `backtest/formula_normalize.py` |
| 3.1.2 | DSL 直接识别 | ✅ | `backtest/formula_normalize.py` |
| 3.1.3 | 关键词映射表 | ✅ | `backtest/formula_normalize.py` |
| 3.1.4 | 50+ 中文金融术语 | ✅ | `_KEYWORD_MAP` (50+) |
| 3.1.5 | 正则匹配 (N日收益/波动/均值) | ✅ | `_heuristic_normalize` |
| 3.1.6 | LLM 归一化 | ✅ | `_llm_normalize` |
| 3.1.7 | 归一化结果缓存 | ✅ | `_normalize_cache` |
| 3.1.8 | FORMULA_NORMALIZE_PROMPT | ✅ | `config.py` |
| 3.1.9 | FormulaSpec 输出 | ✅ | `formula_normalize.py` |
| 3.1.10 | 优先级: DSL→启发式→LLM | ✅ | `formula_normalize.py` |
| 3.1.11 | 术语优先级修复 | ✅ | 对数市值 > 市值 |
| 3.1.12 | 复杂公式识别失败标记 | ✅ | `formula_status=failed` |
| 3.1.13 | 归一化日志 | ❌ | - |
| 3.1.14 | 自定义术语扩展 | 🔧 | 代码级扩展 |

**实现度: 12/14 (86%)**

### 3.2 DSL 编译器 (FR-007)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 3.2.1 | `FormulaCompiler` 类 | ✅ | `backtest/formula_compiler.py` |
| 3.2.2 | `parse()` AST 解析 | ✅ | `formula_compiler.py` |
| 3.2.3 | AST 白名单验证 | ✅ | `ALLOWED_NODES` |
| 3.2.4 | 禁止 `__import__` | ✅ | AST 验证 |
| 3.2.5 | 禁止 `exec`/`eval` | ✅ | AST 验证 |
| 3.2.6 | 禁止属性访问 | ✅ | AST 验证 |
| 3.2.7 | 16 个白名单函数 | ✅ | `ALLOWED_FUNCS` |
| 3.2.8 | 8 个输入面板 | ✅ | `ALLOWED_INPUTS` |
| 3.2.9 | `compile()` 编译为函数 | ✅ | `formula_compiler.py` |
| 3.2.10 | `compute()` 计算 FactorPanel | ✅ | `formula_compiler.py` |
| 3.2.11 | `extract_inputs()` 输入提取 | ✅ | `formula_compiler.py` |
| 3.2.12 | `validate_inputs()` 校验 | ✅ | `formula_compiler.py` |
| 3.2.13 | pct_change(n) | ✅ | `_pct_change` |
| 3.2.14 | rolling_mean/std/sum(n) | ✅ | `_rolling_*` |
| 3.2.15 | ts_rank/zscore/delta(n) | ✅ | `_ts_*` |
| 3.2.16 | cs_rank/zscore() | ✅ | `_cs_*` |
| 3.2.17 | winsorize(q) | ✅ | `_winsorize` |
| 3.2.18 | log/abs/sign() | ✅ | `_log/_abs/_sign` |
| 3.2.19 | clip(lo,hi) | ✅ | `_clip` |
| 3.2.20 | where(cond,a,b) | ✅ | `_where` |
| 3.2.21 | delay(n) | ✅ | `_delay` |
| 3.2.22 | 四则运算 + 幂运算 | ✅ | AST BinOp |
| 3.2.23 | FormulaError 异常 | ✅ | `formula_compiler.py` |
| 3.2.24 | FactorCalculator 集成 | ✅ | `backtest/factors.py` |

**实现度: 24/24 (100%)**

---

## 4. 回测分析模块

### 4.1 因子回测 (FR-008)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 4.1.1 | IC 逐日截面计算 | ✅ | `backtest/ic.py::ic_series` |
| 4.1.2 | IC 均值 | ✅ | `ic_summary` |
| 4.1.3 | ICIR (IC均值/IC标准差) | ✅ | `ic_summary` |
| 4.1.4 | t 统计量 | ✅ | `ic_summary` |
| 4.1.5 | p 值 | ✅ | `ic_summary` |
| 4.1.6 | IC 胜率 (positive_ratio) | ✅ | `ic_summary` |
| 4.1.7 | Rank IC (Spearman) | ✅ | `rank_ic_series` |
| 4.1.8 | 分组收益 (Q1-Q5) | ✅ | `quantile_returns` |
| 4.1.9 | 分组收益摘要 | ✅ | `quantile_summary` |
| 4.1.10 | 多空收益 (Q5-Q1) | ✅ | `long_short_curve` |
| 4.1.11 | 因子衰减 (多持有期) | ✅ | `factor_decay` |
| 4.1.12 | 换手率时序 | ✅ | `factor_turnover` |
| 4.1.13 | 滚动 IC | ✅ | `rolling_ic` |
| 4.1.14 | FactorEvalResult 输出 | ✅ | `backtest/types.py` |
| 4.1.15 | IC 时序列表 | ✅ | `ic_series` |
| 4.1.16 | 数据点数统计 | ✅ | `data_points` |
| 4.1.17 | 状态标记 (ok/error) | ✅ | `status` |
| 4.1.18 | API `POST /backtest/factor` | ✅ | `api/backtest_routes.py` |
| 4.1.19 | 前端因子回测 UI | ✅ | `frontend/index.html` |
| 4.1.20 | IC 图表可视化 | ✅ | ECharts bar |
| 4.1.21 | IC 阈值判断 | ❌ | - |
| 4.1.22 | 因子正交化 | ❌ | - |
| 4.1.23 | 因子合成 | ❌ | - |

**实现度: 20/23 (87%)**

### 4.2 策略回测 (FR-009)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 4.2.1 | StrategyConfig | ✅ | `backtest/strategy.py` |
| 4.2.2 | 等权方案 | ✅ | `EqualWeight` |
| 4.2.3 | 市值加权方案 | ✅ | `MarketCapWeight` |
| 4.2.4 | 风险平价方案 | ✅ | `RiskParity` |
| 4.2.5 | 调仓频率 (日/周/月/季) | ✅ | `_get_rebalance_dates` |
| 4.2.6 | 交易成本 (bps) | ✅ | `transaction_cost_bps` |
| 4.2.7 | 滑点 (bps) | ✅ | `slippage_bps` |
| 4.2.8 | 多空策略 | ✅ | `long_short` |
| 4.2.9 | max_weight 约束 | ✅ | `_apply_max_weight` |
| 4.2.10 | 现金仓位支持 | ✅ | sum < 1 |
| 4.2.11 | 净值曲线 | ✅ | `equity` |
| 4.2.12 | 日收益率序列 | ✅ | `returns` |
| 4.2.13 | 回撤水下图 | ✅ | `drawdown` |
| 4.2.14 | 换手率序列 | ✅ | `turnover_series` |
| 4.2.15 | 权重历史记录 | ✅ | `weights_history` |
| 4.2.16 | 基准净值 | 🔧 | 有字段未填充 |
| 4.2.17 | API `POST /backtest/strategy` | ✅ | `api/backtest_routes.py` |
| 4.2.18 | 前端策略回测 UI | ✅ | `frontend/index.html` |
| 4.2.19 | 净值曲线图表 | ✅ | ECharts line |
| 4.2.20 | 绩效指标卡片 | ✅ | `frontend/index.html` |
| 4.2.21 | 分批建仓 | ❌ | - |
| 4.2.22 | 止损/止盈 | ❌ | - |
| 4.2.23 | 风格中性化 | ❌ | - |

**实现度: 19/23 (83%)**

### 4.3 风险绩效 (FR-010)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 4.3.1 | 年化收益率 | ✅ | `backtest/metrics.py` |
| 4.3.2 | 年化波动率 | ✅ | `annual_volatility` |
| 4.3.3 | Sharpe Ratio | ✅ | `sharpe_ratio` |
| 4.3.4 | Sortino Ratio | ✅ | `sortino_ratio` |
| 4.3.5 | Calmar Ratio | ✅ | `calmar_ratio` |
| 4.3.6 | 最大回撤 | ✅ | `max_drawdown` |
| 4.3.7 | 回撤持续时间 | ✅ | `max_drawdown` |
| 4.3.8 | 胜率 | ✅ | `win_rate` |
| 4.3.9 | 总收益率 | ✅ | `total_return` |
| 4.3.10 | 月度收益 | ✅ | `monthly_returns` |
| 4.3.11 | 月度收益矩阵 | ✅ | `monthly_return_matrix` |
| 4.3.12 | 滚动年化收益 | ✅ | `rolling_performance` |
| 4.3.13 | 滚动波动率 | ✅ | `rolling_performance` |
| 4.3.14 | 滚动 Sharpe | ✅ | `rolling_performance` |
| 4.3.15 | 信息比率 (IR) | ❌ | - |
| 4.3.16 | Omega Ratio | ❌ | - |
| 4.3.17 | Tail Risk | ❌ | - |

**实现度: 14/17 (82%)**

---

## 5. 数据层

### 5.1 本地真实数据 (FR-011)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 5.1.1 | LocalDataProvider | ✅ | `backtest/providers.py` |
| 5.1.2 | Parquet 格式存储 | ✅ | `data/market/prices.parquet` |
| 5.1.3 | 53 只 CSI500 | ✅ | 真实 A 股 |
| 5.1.4 | 1212 交易日 (2021-2025) | ✅ | 5 年 |
| 5.1.5 | 前复权收盘价 (qfq) | ✅ | AKShare qfq |
| 5.1.6 | meta.json 元数据 | ✅ | `data/market/meta.json` |
| 5.1.7 | `download_data.py` 下载脚本 | ✅ | `backtest/download_data.py` |
| 5.1.8 | 时间范围过滤 | ✅ | `get_prices()` |
| 5.1.9 | 市值数据 | 🔧 | 可选字段 |
| 5.1.10 | 成交量数据 | 🔧 | 可选字段 |
| 5.1.11 | 财务数据 (ROE等) | ❌ | - |
| 5.1.12 | 行业分类 | ❌ | - |
| 5.1.13 | 200+ 只扩展 | 🔧 | 需网络重试 |

**实现度: 9/13 (69%)**

### 5.2 合成数据 (FR-012)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 5.2.1 | SyntheticProvider | ✅ | `backtest/providers.py` |
| 5.2.2 | seed 确定性 | ✅ | `seed` 参数 |
| 5.2.3 | 随机游走生成 | ✅ | `get_prices()` |
| 5.2.4 | 因子注入 | ✅ | `factor_ic` |
| 5.2.5 | 市值数据 | ✅ | `get_market_cap()` |
| 5.2.6 | 前端可选 | ✅ | "合成数据" |

**实现度: 6/6 (100%)**

### 5.3 AKShare 数据 (FR-013)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 5.3.1 | AKShareProvider | ✅ | `backtest/providers.py` |
| 5.3.2 | 股票池查询 | ✅ | `get_universe()` |
| 5.3.3 | 日线数据下载 | ✅ | `get_prices()` |
| 5.3.4 | 限流处理 | 🔧 | 简单 sleep |
| 5.3.5 | 断点续传 | ❌ | - |
| 5.3.6 | 缓存机制 | 🔧 | `data/cache/` |

**实现度: 3/6 (50%)**

---

## 6. 产品闭环

### 6.1 一键全流程 (FR-014)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 6.1.1 | `run_paper_to_decision()` | ✅ | `core/product_loop.py` |
| 6.1.2 | `run_factor_to_decision()` | ✅ | `core/product_loop.py` |
| 6.1.3 | LoopResult 数据结构 | ✅ | `core/product_loop.py` |
| 6.1.4 | 提取→编译→回测→写回 | ✅ | `core/product_loop.py` |
| 6.1.5 | 报告自动生成 | ✅ | `core/product_loop.py` |
| 6.1.6 | 写回 factor_db | ✅ | `update_backtest_result` |
| 6.1.7 | 论文状态更新 | ✅ | `set_factor_count` |
| 6.1.8 | API `POST /papers/{id}/run` | ✅ | `api/paper_routes.py` |
| 6.1.9 | 前端一键回测按钮 | ✅ | `frontend/index.html` |
| 6.1.10 | 批量处理多论文 | 🔧 | 循环调用 |
| 6.1.11 | 进度反馈 | ❌ | - |
| 6.1.12 | 断点续跑 | ❌ | - |

**实现度: 9/12 (75%)**

### 6.2 因子审核 (FR-015)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 6.2.1 | 状态机 (pending→approved/rejected) | ✅ | `core/factor_db.py` |
| 6.2.2 | `update_status()` | ✅ | `core/factor_db.py` |
| 6.2.3 | `approve_factor()` | ✅ | `core/factor_db.py` |
| 6.2.4 | `reject_factor()` | ✅ | `core/factor_db.py` |
| 6.2.5 | `activate_factor()` | ✅ | `core/factor_db.py` |
| 6.2.6 | 批量审核 | ✅ | `batch_update_status` |
| 6.2.7 | API `POST /factors/{name}/decision` | ✅ | `api/factor_routes.py` |
| 6.2.8 | 前端批准按钮 | ✅ | `frontend/index.html` |
| 6.2.9 | 前端拒绝按钮 | ✅ | `frontend/index.html` |
| 6.2.10 | 前端激活按钮 | ✅ | `frontend/index.html` |
| 6.2.11 | 审核备注 | ✅ | `notes` |
| 6.2.12 | 审核日志 | ❌ | - |
| 6.2.13 | 审核权限 | ❌ | - |

**实现度: 11/13 (85%)**

---

## 7. Agent 自动化

### 7.1 自动论文挖掘 (FR-016)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 7.1.1 | LangGraph StateGraph | ✅ | `harness/graph_builder.py` |
| 7.1.2 | plan 节点 (LLM 决策) | ✅ | `harness/plan_node.py` |
| 7.1.3 | collect 节点 | ✅ | `nodes/collect.py` |
| 7.1.4 | paper_retrieve 节点 | ✅ | `nodes/paper_retrieve.py` |
| 7.1.5 | analyze 节点 | ✅ | `nodes/analyze.py` |
| 7.1.6 | persist_factors 节点 | ✅ | `nodes/persist_factors.py` |
| 7.1.7 | quality_check 节点 | ✅ | `nodes/quality.py` |
| 7.1.8 | backtest 节点 | ✅ | `nodes/backtest.py` |
| 7.1.9 | decide 节点 | ✅ | `nodes/decide.py` |
| 7.1.10 | respond 节点 | ✅ | `nodes/respond.py` |
| 7.1.11 | hard_rules YAML DSL | ✅ | `config.yaml` |
| 7.1.12 | 防循环 (visited≥2) | ✅ | `plan_node.py` |
| 7.1.13 | max_iter 兜底 | ✅ | `plan_node.py` |
| 7.1.14 | LLM 失败 fallback | ✅ | `_fallback_decision` |
| 7.1.15 | formula-first 回测 | ✅ | `backtest_node` |
| 7.1.16 | goal 字段传递 | ✅ | `state_base.py` |
| 7.1.17 | final_message 返回 | ✅ | `respond.py` |
| 7.1.18 | 工具调用闭环 (verify+retry) | ✅ | `harness/tool_call.py` |
| 7.1.19 | 多轮迭代 | ✅ | `iteration_count` |
| 7.1.20 | Session checkpoint | ✅ | MemorySaver |
| 7.1.21 | 中断/恢复 | 🔧 | interrupt_before_decide |
| 7.1.22 | 并行执行 | ❌ | - |

**实现度: 20/22 (91%)**

### 7.2 Agent 问答 (FR-017)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 7.2.1 | 对话式 UI | ✅ | `frontend/index.html` |
| 7.2.2 | 消息气泡 | ✅ | `frontend/index.html` |
| 7.2.3 | 加载动画 | ✅ | `frontend/index.html` |
| 7.2.4 | 快捷提问按钮 | ✅ | 3 个 |
| 7.2.5 | `POST /agent/run` | ✅ | `api/server.py` |
| 7.2.6 | 结果摘要展示 | ✅ | `frontend/index.html` |
| 7.2.7 | 统计信息 | ✅ | 论文/因子/回测数 |
| 7.2.8 | 对话历史 | ❌ | 仅当次 |
| 7.2.9 | SSE 流式 | 🔧 | API 有 UI 未接 |
| 7.2.10 | 对话导出 | ❌ | - |

**实现度: 7/10 (70%)**

---

## 8. RAG 检索

### 8.1 语义检索 (FR-018)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 8.1.1 | FAISS 向量索引 | ✅ | `harness/rag/index.py` |
| 8.1.2 | BM25 关键词索引 | ✅ | `harness/rag/bm25.py` |
| 8.1.3 | RRF 融合 | ✅ | `harness/rag/fusion.py` |
| 8.1.4 | BGE Embedding | ✅ | `harness/rag/embedder.py` |
| 8.1.5 | Cross-encoder Rerank | ✅ | `harness/rag/rerank.py` |
| 8.1.6 | 中文 query 扩展 | ✅ | `query_enhance.py` |
| 8.1.7 | LLM query 改写 | ✅ | `query_enhance.py` |
| 8.1.8 | Metadata filter | ✅ | `retriever.py` |
| 8.1.9 | HNSW 索引 | ✅ | `index.py` |
| 8.1.10 | API `POST /rag/search` | ✅ | `api/rag_routes.py` |
| 8.1.11 | 前端 RAG 检索 | ✅ | `frontend/index.html` |
| 8.1.12 | Recall@5 = 0.86 | ✅ | ground-truth 评测 |
| 8.1.13 | NDCG@5 = 0.79 | ✅ | ground-truth 评测 |
| 8.1.14 | 多向量检索 | ❌ | - |
| 8.1.15 | ColBERT | ❌ | - |

**实现度: 13/15 (87%)**

---

## 9. 报告生成

### 9.1 HTML 交互式报告 (FR-019)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 9.1.1 | `to_html()` 生成 | ✅ | `backtest/report.py` |
| 9.1.2 | 自包含 HTML | ✅ | 内嵌数据 |
| 9.1.3 | ECharts CDN | ✅ | 三级 fallback |
| 9.1.4 | 净值曲线图 | ✅ | line |
| 9.1.5 | 回撤水下图 | ✅ | area |
| 9.1.6 | IC 时序图 | ✅ | bar |
| 9.1.7 | 分组收益柱状图 | ✅ | bar |
| 9.1.8 | 因子衰减图 | ✅ | line |
| 9.1.9 | 绩效指标表格 | ✅ | HTML table |
| 9.1.10 | 中文界面 | ✅ | - |
| 9.1.11 | 响应式布局 | 🔧 | 基本响应 |
| 9.1.12 | PDF 导出 | ❌ | - |
| 9.1.13 | 月度热力图 | 🔧 | 有数据未渲染 |
| 9.1.14 | 滚动 Sharpe 图 | 🔧 | 有数据未渲染 |

**实现度: 10/14 (71%)**

### 9.2 JSON 持久化 (FR-020)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 9.2.1 | `save_json()` | ✅ | `backtest/report.py` |
| 9.2.2 | `load_json()` | ✅ | `backtest/report.py` |
| 9.2.3 | `to_dict()` | ✅ | 全部类型 |
| 9.2.4 | `output/backtests/{id}/` | ✅ | 目录结构 |
| 9.2.5 | API `GET /reports` | ✅ | `api/factor_routes.py` |
| 9.2.6 | API `GET /reports/{id}` | ✅ | `api/factor_routes.py` |
| 9.2.7 | 报告删除 | ❌ | - |
| 9.2.8 | 报告对比 | ❌ | - |

**实现度: 6/8 (75%)**

---

## 10. 前端界面

### 10.1 Dashboard (FR-021)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 10.1.1 | 因子统计卡片 | ✅ | `frontend/index.html` |
| 10.1.2 | 论文统计卡片 | ✅ | `frontend/index.html` |
| 10.1.3 | 回测统计卡片 | ✅ | `frontend/index.html` |
| 10.1.4 | Top 因子排行 | ✅ | |IC| 排序 |
| 10.1.5 | 因子状态饼图 | ✅ | ECharts pie |
| 10.1.6 | 最近报告列表 | ✅ | `frontend/index.html` |
| 10.1.7 | 实时数据 | ✅ | `GET /dashboard/stats` |
| 10.1.8 | 30s 缓存 | ✅ | `api/factor_routes.py` |
| 10.1.9 | 因子类别分布图 | 🔧 | 改为状态分布 |
| 10.1.10 | 趋势图 (时间线) | ❌ | - |

**实现度: 8/10 (80%)**

### 10.2 论文库 UI (FR-022)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 10.2.1 | 论文列表 | ✅ | `frontend/index.html` |
| 10.2.2 | arXiv 搜索框 | ✅ | `frontend/index.html` |
| 10.2.3 | PDF 上传按钮 | ✅ | `frontend/index.html` |
| 10.2.4 | 搜索结果展示 | ✅ | `frontend/index.html` |
| 10.2.5 | 一键导入按钮 | ✅ | `frontend/index.html` |
| 10.2.6 | 提取因子按钮 | ✅ | `frontend/index.html` |
| 10.2.7 | 一键回测按钮 | ✅ | `frontend/index.html` |
| 10.2.8 | 状态标签 | ✅ | tag |
| 10.2.9 | 来源标签 | ✅ | tag |
| 10.2.10 | 论文详情弹窗 | ❌ | - |
| 10.2.11 | 论文筛选 | ❌ | - |
| 10.2.12 | 批量操作 | ❌ | - |

**实现度: 9/12 (75%)**

### 10.3 因子审核 UI (FR-023)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 10.3.1 | 因子列表 | ✅ | `frontend/index.html` |
| 10.3.2 | 状态标签 | ✅ | tag |
| 10.3.3 | IC 值显示 | ✅ | tag |
| 10.3.4 | ICIR 显示 | ✅ | tag |
| 10.3.5 | compute_source 标签 | ✅ | tag |
| 10.3.6 | 批准按钮 | ✅ | `decideFactor` |
| 10.3.7 | 拒绝按钮 | ✅ | `decideFactor` |
| 10.3.8 | 激活按钮 | ✅ | `decideFactor` |
| 10.3.9 | 回测按钮 | ✅ | `backtestFactor` |
| 10.3.10 | 刷新按钮 | ✅ | `frontend/index.html` |
| 10.3.11 | 因子详情弹窗 | ❌ | - |
| 10.3.12 | IC 图表嵌入 | ❌ | - |
| 10.3.13 | 批量审核 | ❌ | - |
| 10.3.14 | 因子对比 | ❌ | - |

**实现度: 10/14 (71%)**

### 10.4 报告中心 (FR-024)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 10.4.1 | 报告列表 | ✅ | `frontend/index.html` |
| 10.4.2 | IC 标签 | ✅ | tag |
| 10.4.3 | 收益标签 | ✅ | tag |
| 10.4.4 | 回撤标签 | ✅ | tag |
| 10.4.5 | 点击查看 HTML | ✅ | `window.open` |
| 10.4.6 | 报告筛选 | 🔧 | 全部/因子/策略 |
| 10.4.7 | 报告搜索 | ❌ | - |
| 10.4.8 | 报告导出 | ❌ | - |
| 10.4.9 | 报告对比 | ❌ | - |

**实现度: 5/9 (56%)**

### 10.5 通用 UI

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 10.5.1 | 侧边栏导航 | ✅ | `frontend/index.html` |
| 10.5.2 | 页面切换动画 | ✅ | `@keyframes` |
| 10.5.3 | Toast 通知 | ✅ | `frontend/index.html` |
| 10.5.4 | 加载 spinner | ✅ | CSS |
| 10.5.5 | 响应式 meta | ✅ | viewport |
| 10.5.6 | 简洁白色主题 | ✅ | Tailwind |
| 10.5.7 | ECharts 三级 fallback | ✅ | bootcdn/unpkg/jsdelivr |
| 10.5.8 | Dark mode | ❌ | - |
| 10.5.9 | 国际化 | ❌ | 仅中文 |
| 10.5.10 | 键盘快捷键 | ❌ | - |

**实现度: 7/10 (70%)**

---

## 11. API 文档

### 11.1 Swagger (FR-025)

| 颗粒度 | 描述 | 状态 | 位置 |
|--------|------|------|------|
| 11.1.1 | FastAPI 自动生成 | ✅ | `/docs` |
| 11.1.2 | 31 个 API 端点 | ✅ | 4 个 router |
| 11.1.3 | Pydantic 模型 | ✅ | 请求/响应 |
| 11.1.4 | API 分组 (tags) | ✅ | papers/factors/backtest/rag |
| 11.1.5 | OpenAPI JSON | ✅ | `/openapi.json` |
| 11.1.6 | API 版本管理 | ❌ | - |
| 11.1.7 | API 限流 | ❌ | - |

**实现度: 5/7 (71%)**

---

## 12. 测试覆盖

| 模块 | 单元测试 | E2E | 覆盖率 |
|------|----------|-----|--------|
| 回测引擎 | 30+ | - | 高 |
| 公式编译 | 23 | - | 高 |
| 数据层 | 8 | - | 中 |
| 因子库 | 22 | - | 高 |
| 论文库 | 15 | - | 高 |
| API | 30+ | - | 高 |
| 前端 | 9 | - | 中 |
| Agent | - | 22 项全流程 | 中 |
| RAG | 40+ | - | 高 |
| **总计** | **795+** | **22** | - |

---

## 13. 总实现度汇总

| 模块 | 已实现 | 总数 | 实现度 |
|------|--------|------|--------|
| 1. 论文输入 | 33 | 42 | **79%** |
| 2. 因子提取 | 19 | 21 | **90%** |
| 3. 公式编译 | 36 | 38 | **95%** |
| 4. 回测分析 | 53 | 63 | **84%** |
| 5. 数据层 | 18 | 25 | **72%** |
| 6. 产品闭环 | 20 | 25 | **80%** |
| 7. Agent | 27 | 32 | **84%** |
| 8. RAG | 13 | 15 | **87%** |
| 9. 报告 | 16 | 22 | **73%** |
| 10. 前端 | 39 | 55 | **71%** |
| 11. API | 5 | 7 | **71%** |
| **总计** | **279** | **345** | **81%** |

---

## 14. 未实现功能（按优先级）

### P1 (高价值)

| 功能 | 模块 | 价值 |
|------|------|------|
| 论文详情弹窗 | 前端 | 用户体验 |
| 因子详情弹窗 + IC 图表 | 前端 | 决策支持 |
| 因子对比功能 | 前端 | 横向比较 |
| 月度收益热力图 | 报告 | 可视化完善 |
| 滚动 Sharpe 图 | 报告 | 风险分析 |
| 提取进度反馈 | 产品闭环 | 用户体验 |

### P2 (中等价值)

| 功能 | 模块 | 价值 |
|------|------|------|
| 财务数据集成 | 数据层 | 更多因子 |
| 行业分类 | 数据层 | 行业中性化 |
| 因子正交化 | 因子回测 | 去冗余 |
| SSE 流式输出 | Agent | 实时反馈 |
| Dark mode | 前端 | 用户体验 |
| 对话历史 | Agent | 多轮交互 |

### P3 (低优先级)

| 功能 | 模块 | 价值 |
|------|------|------|
| PDF 导出 | 报告 | 离线查看 |
| API 版本管理 | API | 长期维护 |
| 批量上传 | 论文 | 效率 |
| ColBERT | RAG | 检索质量 |
| 止损/止盈 | 策略 | 策略丰富 |

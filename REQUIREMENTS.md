# FactorLab — 论文因子挖掘与回测系统

## 需求文档 v1.0

---

## 1. 产品定位

**FactorLab** 是一个自动化量化因子挖掘平台，核心回路为：

```
论文输入 → LLM 提取因子 → 公式编译 → 回测分析 → 报告生成 → 决策审核
```

**目标用户**：量化研究员、金融工程学生、因子投资从业者

**核心价值**：将学术论文中的量化因子idea自动转化为可回测、可决策的因子，大幅缩短从论文到实盘的研究周期。

---

## 2. 系统架构

### 2.1 技术栈

| 层 | 技术 | 用途 |
|---|------|------|
| 前端 | HTML + Tailwind CSS + ECharts | 单页应用 (SPA) |
| API | FastAPI (Python 3.13) | REST API + Swagger 文档 |
| Agent | LangGraph | 多步推理 + 工具调用 |
| LLM | DeepSeek (主) / GLM (备) | 因子提取 + 公式归一化 |
| 检索 | FAISS + BM25 + BGE Embedding | RAG 论文检索 |
| 回测 | Pandas + NumPy | IC/ICIR/策略回测 |
| 数据 | AKShare / 本地 Parquet | A 股行情数据 |

### 2.2 模块结构

```
paper-factor-system/
├── backtest/           # 回测引擎
│   ├── types.py        # MarketData, FactorPanel, FactorEvalResult
│   ├── providers.py    # 数据层 (Local/Synthetic/AKShare)
│   ├── formula_compiler.py  # DSL → 可执行代码
│   ├── formula_normalize.py # 自然语言 → DSL
│   ├── factors.py      # 因子计算器
│   ├── ic.py           # IC/ICIR/分组收益
│   ├── strategy.py     # 策略引擎
│   ├── metrics.py      # 风险绩效指标
│   ├── report.py       # 报告生成
│   └── pipeline.py     # 一键编排
├── core/               # 核心业务
│   ├── analyzer.py     # LLM 因子提取
│   ├── factor_db.py    # 因子库管理
│   ├── paper_library.py # 论文库
│   ├── product_loop.py # 产品闭环编排
│   └── collector.py    # 论文采集
├── harness/            # Agent 框架
│   ├── graph_builder.py # LangGraph 构建
│   ├── tool_call.py    # 工具闭环 (verify+retry)
│   ├── verify.py       # 结果校验
│   └── rag/            # RAG 检索
├── api/                # HTTP API
│   ├── server.py       # 主服务
│   ├── paper_routes.py # 论文 API
│   ├── factor_routes.py # 因子 API
│   ├── backtest_routes.py # 回测 API
│   └── rag_routes.py   # RAG API
├── frontend/           # 前端
│   └── index.html      # 单页应用
├── pipelines/          # LangGraph Pipeline
│   └── paper_factor/   # 论文→因子 pipeline
├── data/               # 数据目录
│   ├── papers/         # 论文数据
│   ├── market/         # 行情数据
│   └── factor_candidates.json # 因子库
└── tests/              # 测试
```

---

## 3. 功能需求

### 3.1 论文输入 (Paper Input)

#### 3.1.1 PDF 上传
- **需求 ID**: FR-001
- **优先级**: P0
- **描述**: 用户可上传 PDF 格式的金融论文
- **输入**: PDF 文件 (≤20MB)
- **处理**:
  1. pdfplumber 解析文本 (前 5 页)
  2. 自动提取标题、摘要
  3. 存入论文库 (`PaperLibrary`)
  4. 可选索引到 RAG
- **API**: `POST /papers/upload`
- **验收**:
  - [x] 支持 PDF 格式
  - [x] 自动解析文本
  - [x] 非 PDF 格式拒绝
  - [x] 大小限制 20MB

#### 3.1.2 arXiv 搜索
- **需求 ID**: FR-002
- **优先级**: P0
- **描述**: 按关键词搜索 arXiv 金融论文
- **输入**: 关键词, 最大结果数
- **处理**:
  1. 按 q-fin 分类搜索
  2. 按关键词搜索
  3. 金融相关性过滤
  4. 去重
- **API**: `POST /papers/search`
- **验收**:
  - [x] 按分类搜索 (q-fin.*)
  - [x] 按关键词搜索
  - [x] 金融相关性过滤
  - [x] 结果去重

#### 3.1.3 论文库管理
- **需求 ID**: FR-003
- **优先级**: P1
- **描述**: 统一管理所有论文
- **功能**:
  - 论文列表 (分页/搜索/状态过滤)
  - 论文详情 (含提取的因子)
  - 一键提取因子
  - 一键完整回测
- **API**: `GET /papers`, `GET /papers/{id}`, `POST /papers/{id}/extract`, `POST /papers/{id}/run`

### 3.2 因子提取 (Factor Extraction)

#### 3.2.1 LLM 因子提取
- **需求 ID**: FR-004
- **优先级**: P0
- **描述**: 从论文中自动提取量化因子
- **LLM**: DeepSeek (主) / GLM (备)
- **提取内容**:
  - `factor_name`: 因子名称
  - `definition`: 因子定义
  - `calculation_formula`: 计算公式
  - `data_needed`: 所需数据
  - `confidence`: 置信度 (高/中/低)
  - `market_scope`: 适用市场
- **验收**:
  - [x] 能提取明确的因子定义
  - [x] 能提取计算公式
  - [x] 置信度分级
  - [x] LLM 失败时降级处理

#### 3.2.2 RAG 上下文增强
- **需求 ID**: FR-005
- **优先级**: P1
- **描述**: 提取因子时注入相似论文作为参考
- **处理**:
  1. RAG 检索 top-k 相似论文
  2. 注入提取 prompt
  3. 标记 `_used_rag_context`
- **验收**:
  - [x] 自动检索相关论文
  - [x] 注入提取上下文
  - [x] 标记 RAG 使用情况

### 3.3 公式编译 (Formula Compilation)

#### 3.3.1 公式归一化
- **需求 ID**: FR-006
- **优先级**: P0
- **描述**: 将自然语言公式转换为受限 DSL 表达式
- **支持的 DSL**:
  ```
  函数: pct_change, rolling_mean, rolling_std, rolling_sum,
        ts_rank, ts_zscore, ts_delta, cs_rank, cs_zscore,
        winsorize, log, abs, sign, clip, where, delay
  输入: close, returns, volume, market_cap
  运算: + - * / **
  ```
- **归一化策略**:
  1. 已有 DSL → 直接使用
  2. 启发式关键词匹配 (50+ 个中文金融术语)
  3. LLM 兜底 (DeepSeek)
- **验收**:
  - [x] 50+ 中文金融术语覆盖
  - [x] 启发式优先 + LLM 兜底
  - [x] 归一化结果缓存

#### 3.3.2 DSL 编译器
- **需求 ID**: FR-007
- **优先级**: P0
- **描述**: 安全编译 DSL 表达式为可执行函数
- **安全要求**:
  - AST 白名单验证
  - 禁止 `__import__`, `exec`, `eval`, `lambda`
  - 只允许白名单函数和输入
- **输出**: `Callable[[MarketData], pd.DataFrame]`
- **验收**:
  - [x] AST 白名单验证
  - [x] 拒绝注入攻击
  - [x] 编译为 FactorPanel

### 3.4 回测分析 (Backtesting)

#### 3.4.1 因子回测
- **需求 ID**: FR-008
- **优先级**: P0
- **描述**: 计算因子 IC/ICIR/分组收益等指标
- **指标**:
  - IC (Information Coefficient): 截面相关系数
  - ICIR (IC Information Ratio): IC 均值 / IC 标准差
  - Rank IC: Spearman 秩相关
  - 分组收益: Q1-Q5 分位数组未来收益
  - 多空收益: Top 组 - Bottom 组
  - 因子衰减: 不同持有期 IC
  - 换手率: Top 组持仓变化
- **API**: `POST /backtest/factor`
- **验收**:
  - [x] IC/ICIR/RankIC 计算
  - [x] 分组收益
  - [x] 多空收益
  - [x] 因子衰减
  - [x] 换手率

#### 3.4.2 策略回测
- **需求 ID**: FR-009
- **优先级**: P0
- **描述**: 模拟组合构建与净值表现
- **功能**:
  - 权重方案: 等权 / 市值加权 / 风险平价
  - 调仓频率: 日/周/月/季
  - 交易成本: bps 模拟
  - 多空策略: 可选
  - max_weight 约束
- **输出**: 净值曲线、回撤、换手率、成本
- **API**: `POST /backtest/strategy`
- **验收**:
  - [x] 3 种权重方案
  - [x] 交易成本模拟
  - [x] 净值曲线
  - [x] max_weight 约束

#### 3.4.3 风险绩效分析
- **需求 ID**: FR-010
- **优先级**: P1
- **描述**: 全套风险绩效指标
- **指标**:
  - 年化收益率 / 年化波动率
  - Sharpe / Sortino / Calmar
  - 最大回撤 / 回撤持续
  - 胜率 / 月度收益
  - 滚动窗口绩效
- **验收**:
  - [x] Sharpe/Sortino/Calmar
  - [x] 最大回撤
  - [x] 月度收益矩阵

### 3.5 数据层 (Data Providers)

#### 3.5.1 本地真实数据
- **需求 ID**: FR-011
- **优先级**: P0
- **描述**: 本地存储的真实 A 股行情数据
- **当前数据**: 53 只 CSI500 × 1212 天 (2021-2025)
- **格式**: Parquet (`data/market/prices.parquet`)
- **API**: `provider="local"`
- **验收**:
  - [x] 5 年真实 A 股数据
  - [x] 前复权收盘价
  - [x] 可通过脚本下载扩展

#### 3.5.2 合成数据
- **需求 ID**: FR-012
- **优先级**: P1
- **描述**: 快速生成合成测试数据
- **特点**: seed 可控、确定性、可注入已知 IC
- **API**: `provider="synthetic"`
- **验收**:
  - [x] 同 seed 确定性
  - [x] 可注入已知 IC

#### 3.5.3 AKShare 实时数据
- **需求 ID**: FR-013
- **优先级**: P2
- **描述**: 实时从 AKShare 获取数据
- **API**: `provider="akshare"`

### 3.6 产品闭环 (Product Loop)

#### 3.6.1 一键全流程
- **需求 ID**: FR-014
- **优先级**: P0
- **描述**: 论文 → 提取 → 编译 → 回测 → 写回
- **API**: `POST /papers/{id}/run`
- **处理流程**:
  1. 获取论文
  2. LLM 提取因子
  3. 公式归一化 + 编译
  4. IC/ICIR 回测
  5. 生成报告
  6. 写回因子库
- **验收**:
  - [x] 一键完成全流程
  - [x] 结果写回因子库
  - [x] 报告落盘

#### 3.6.2 因子审核
- **需求 ID**: FR-015
- **优先级**: P0
- **描述**: 人工审核因子决策
- **状态流转**: `pending → approved/rejected → active/deprecated`
- **API**: `POST /factors/{name}/decision`
- **验收**:
  - [x] 批准/拒绝/激活/废弃
  - [x] 批量审核
  - [x] 状态持久化

### 3.7 Agent 自动化

#### 3.7.1 自动论文挖掘
- **需求 ID**: FR-016
- **优先级**: P1
- **描述**: 用户输入自然语言指令，Agent 自动完成全流程
- **Agent Pipeline**:
  ```
  plan → collect → paper_retrieve → analyze → persist_factors → quality_check → backtest → decide → respond
  ```
- **LLM 决策**: DeepSeek 驱动的多步推理
- **硬规则**: YAML DSL 强制执行顺序
- **验收**:
  - [x] 自动搜索论文
  - [x] 自动提取因子
  - [x] 自动回测
  - [x] 生成总结消息

#### 3.7.2 Agent 问答
- **需求 ID**: FR-017
- **优先级**: P1
- **描述**: 对话式 Agent 交互
- **功能**:
  - 自然语言输入
  - 自动执行 pipeline
  - 返回结果摘要 + RAG 推荐
- **验收**:
  - [x] 对话式交互
  - [x] 执行结果反馈
  - [x] Toast 通知

### 3.8 RAG 检索

#### 3.8.1 语义检索
- **需求 ID**: FR-018
- **优先级**: P1
- **描述**: 基于语义的论文检索
- **技术**: FAISS + BM25 + BGE Embedding + Cross-encoder Rerank
- **指标**: Recall@5=0.86, NDCG@5=0.79
- **API**: `POST /rag/search`
- **验收**:
  - [x] 语义检索
  - [x] Rerank 精排
  - [x] 中文 query 增强

### 3.9 报告生成

#### 3.9.1 HTML 交互式报告
- **需求 ID**: FR-019
- **优先级**: P0
- **描述**: 自动生成可视化回测报告
- **图表**:
  - 净值曲线 (ECharts)
  - 回撤水下图
  - IC 时序
  - 分组收益柱状图
  - 因子衰减
  - 月度收益热力图
- **输出**: `output/backtests/{run_id}/report.html`
- **验收**:
  - [x] HTML 自包含文件
  - [x] 交互式图表
  - [x] 中文界面

#### 3.9.2 JSON 结果持久化
- **需求 ID**: FR-020
- **优先级**: P0
- **描述**: 结构化保存回测结果
- **API**: `GET /reports`, `GET /reports/{run_id}`

### 3.10 前端界面

#### 3.10.1 Dashboard
- **需求 ID**: FR-021
- **优先级**: P1
- **描述**: 实时概览
- **内容**: 因子统计/论文统计/回测统计/Top 因子排行

#### 3.10.2 论文库
- **需求 ID**: FR-022
- **优先级**: P0
- **描述**: 论文管理界面
- **功能**: 列表/搜索/上传/提取/一键回测

#### 3.10.3 因子审核
- **需求 ID**: FR-023
- **优先级**: P0
- **描述**: 因子管理界面
- **功能**: 列表/IC 展示/批准/拒绝/激活/回测

#### 3.10.4 报告中心
- **需求 ID**: FR-024
- **优先级**: P1
- **描述**: 报告查看界面

### 3.11 API 文档
- **需求 ID**: FR-025
- **优先级**: P2
- **描述**: Swagger 自动生成 API 文档
- **入口**: `http://localhost:8000/docs`

---

## 4. 非功能需求

### 4.1 性能

| 指标 | 要求 | 当前 |
|------|------|------|
| 因子回测延迟 | <10s | ~5s (53 stocks) |
| 策略回测延迟 | <15s | ~8s |
| Dashboard 加载 | <2s | ~1s (30s 缓存) |
| RAG 检索延迟 | <3s | ~2s |
| Agent 执行 | <60s | ~10-30s |

### 4.2 数据

| 要求 | 描述 |
|------|------|
| 真实数据 | 53 只 A 股 × 5 年 (2021-2025) |
| 可扩展 | `download_data.py` 支持 200+ 只 |
| 数据源 | Local (Parquet) / AKShare / Synthetic |
| 复权 | 前复权 (qfq) |

### 4.3 安全

| 要求 | 描述 |
|------|------|
| DSL 编译 | AST 白名单，禁止代码注入 |
| PDF 上传 | 格式+大小校验 |
| API 调用 | Pydantic 参数验证 |

### 4.4 可靠性

| 要求 | 描述 |
|------|------|
| 测试覆盖 | 795+ 单元测试 |
| E2E 测试 | 22 项端到端验证 |
| 容错 | LLM 失败降级、网络超时处理 |

### 4.5 兼容性

| 要求 | 描述 |
|------|------|
| Python | 3.13+ |
| 操作系统 | Windows 11 (GBK 兼容) |
| 浏览器 | Chrome/Edge (现代浏览器) |

---

## 5. 数据模型

### 5.1 FactorRecord

```json
{
    "factor_name": "string",
    "definition": "string",
    "calculation_formula": "string",
    "formula_expression": "pct_change(close, 21)",
    "formula_status": "compiled | failed | proxy",
    "compute_source": "formula | builtin | proxy",
    "confidence": "高 | 中 | 低",
    "status": "pending | approved | rejected | active",
    "backtest_result": {
        "ic": 0.058,
        "icir": 0.343,
        "rank_ic": 0.055,
        "top_return": 12.5,
        "bottom_return": 3.2,
        "top_minus_bottom": 9.3,
        "annual_return": 10.5,
        "max_drawdown": -8.2,
        "sharpe": 1.21,
        "data_points": 1168,
        "data_provider": "local",
        "data_period": "2021-2025"
    },
    "paper_id": "string",
    "backtest_run_id": "string",
    "added_at": "string",
    "updated_at": "string"
}
```

### 5.2 PaperRecord

```json
{
    "paper_id": "string",
    "title": "string",
    "summary": "string",
    "source": "arxiv | pdf_upload | local",
    "status": "new | indexed | extracted | failed",
    "factor_count": 0,
    "added_at": "string"
}
```

---

## 6. API 清单

### 6.1 论文 API

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/papers` | 论文列表 |
| GET | `/papers/{id}` | 论文详情 |
| POST | `/papers/upload` | PDF 上传 |
| POST | `/papers/search` | arXiv 搜索 |
| POST | `/papers/{id}/extract` | 提取因子 |
| POST | `/papers/{id}/run` | 一键完整闭环 |

### 6.2 因子 API

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/factors` | 因子列表 |
| GET | `/factors/{name}` | 因子详情 |
| POST | `/factors/{name}/backtest` | 回测因子 |
| POST | `/factors/{name}/decision` | 审核决策 |

### 6.3 回测 API

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/backtest/factors` | 可用因子列表 |
| POST | `/backtest/factor` | 因子回测 |
| POST | `/backtest/strategy` | 策略回测 |
| POST | `/backtest/pipeline` | 一键全跑 |
| GET | `/backtest/report/{id}` | HTML 报告 |

### 6.4 其他 API

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/dashboard/stats` | 仪表盘数据 |
| GET | `/reports` | 报告列表 |
| POST | `/rag/search` | RAG 检索 |
| POST | `/agent/run` | Agent 执行 |
| GET | `/docs` | Swagger 文档 |

---

## 7. 测试需求

| 类型 | 数量 | 覆盖 |
|------|------|------|
| 单元测试 | 795+ | 全模块 |
| E2E 测试 | 22 项 | 全流程 |
| 前端测试 | 9 项 | UI 验证 |
| 回测测试 | 30+ | 指标正确性 |

---

## 8. 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v1.0 | 2026-10-21 | 初始版本，完整产品闭环 |

---

## 9. 附录

### 9.1 启动命令

```bash
# 启动服务
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000

# 运行测试
python -m pytest tests/ -q

# 下载数据
python -m backtest.download_data --universe csi500 --n-stocks 200

# 完整流程测试
python test_full_flow.py
```

### 9.2 环境要求

```bash
pip install fastapi uvicorn pandas numpy langgraph langchain-core
pip install sentence-transformers faiss-cpu rank_bm25
pip install akshare pdfplumber python-multipart
```

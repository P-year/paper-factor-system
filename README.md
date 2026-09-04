# 论文挖因子系统 (Paper Factor System)

从学术论文中挖掘A股量化因子的智能系统。支持多模态分析（因子+新闻流）、真实数据回测、批量处理。

## 快速开始

```bash
pip install arxiv requests pdfplumber akshare
pip install zhipuai     # GLM API（可选，作为 DeepSeek 的备选）

# 配置API密钥（DeepSeek 主，GLM 备）
set DEEPSEEK_API_KEY=your_key_here
set GLM_API_KEY=your_glm_key_here    # 可选：作为 fallback

# 运行完整流程
python main.py --step all

# 交互 Agent 模式
python main_agent.py

# 定时调度（每天一次，自动审批）
python main_scheduler.py --interval 24h

# 查看系统状态
python main.py --status
```

## 三阶段流程

| Step | 命令 | 说明 |
|------|------|------|
| 论文采集 | `--step collector` | arXiv搜索 + 金融过滤 + 去重 |
| 因子分析 | `--step analyzer` | DeepSeek/GLM API + 因子提取（DeepSeek 主，GLM 备） |
| 回测验证 | `--step backtest` | 理论IC估算 |
| 真实回测 | `--step backtest-real` | AKShare真实数据 + IC计算 |

## 新增：真实数据回测 (Phase 2)

```bash
# 单因子真实回测
python main.py --step backtest-real --factor EP --universe csi500 --start-date 20230101 --end-date 20241231

# 全量已知因子批量回测
python main.py --step backtest-real --universe csi500

# 真实数据模式（在其他命令中加 --real-data）
python main.py --step backtest --real-data
```

**支持的因子（真实数据回测）：**

| 因子 | 类别 | 说明 |
|------|------|------|
| EP | 估值 | 市盈率倒数 |
| BP | 估值 | 市净率倒数 |
| SP | 估值 | 市销率倒数 |
| ROE | 质量 | 净资产收益率 |
| gross_profit_margin | 质量 | 毛利率 |
| net_profit_margin | 质量 | 净利率 |
| asset_turnover | 质量 | 总资产周转率 |
| revenue_growth_yoy | 成长 | 营收同比增速 |
| profit_growth_yoy | 成长 | 净利润同比增速 |
| return_1m | 动量 | 近1月收益率 |
| turnover_rate_1m | 动量 | 近1月换手率 |

## 架构升级总览

```
Phase 1 (v1.0)     →  理论估算IC（无需数据）
Phase 1.5 (v1.5)   →  多模态分析 + 多API后端
Phase 2 (v2.0)     →  AKShare真实数据IC计算 ✅ NEW
Phase 3 (v3.0)     →  全自动工厂 + 实盘跟踪
```

**Phase 2 新增文件：**
- `data_fetcher.py` — AKShare数据获取管道（23KB）
- 11个因子定义（FACTOR_CONFIGS）
- IC计算 + 分组回测功能
- `run_backtest_real()` — 一站式真实数据回测

## 状态流转

```
pending → approved → active（实盘）
    ↓         ↓
 rejected   deprecated
```

## 环境变量

```bash
DEEPSEEK_API_KEY  DeepSeek（主，性价比高）
GLM_API_KEY       智谱AI（备，DeepSeek 失败时降级）
```

## Agent 模式（v3.0）

新增 LangGraph agent 编排层，把三阶段流水线升级为状态机。

```
python main_agent.py             # REPL：自然语言驱动 + 交互审批
python main_scheduler.py --once  # 跑一次完整 pipeline
python main_scheduler.py --interval 24h   # 后台定时
python main_agent.py --resume <id>  # 恢复会话
python main_agent.py --list         # 列出所有会话
```

Agent 运行日志：`memory/sessions/`、`memory/runs/`、`memory/audit/`

_Last updated: 2026-07-25 v3.0-agent_

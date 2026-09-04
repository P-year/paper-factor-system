# 复现实验设计方案：2510.15691（A股落地版）

**目标：** 基于论文思路，设计在中国A股市场落地的复现实验

---

## 1. 因子体系设计（A股可落地版）

### 1.1 推荐因子列表（10个，分4类）

#### 估值类（Valuation）
| 因子名 | 计算公式 | 数据来源 |
|--------|---------|---------|
| EP | 市盈率倒数 = 1/PE | AKShare: `stock个体PE` |
| BP | 市净率倒数 = 1/PB | AKShare: `stock个体PB` |
| SP | 市销率倒数 = 1/PS | AKShare: `stock个体PS` |
| EV/EBITDA | 企业价值/息税折旧前利润 | AKShare: `stock_financials_indicator` |

#### 质量类（Quality）
| 因子名 | 计算公式 | 数据来源 |
|--------|---------|---------|
| ROE | 净资产收益率 = 净利润/股东权益 | AKShare: `stock_financials` |
| grossprofit_margin | 毛利率 = (营收-成本)/营收 | AKShare: `stock_financials` |
| netprofit_margin | 净利率 = 净利润/营收 | AKShare: `stock_financials` |
| asset_turnover | 总资产周转率 = 营收/总资产 | AKShare: `stock_financials` |

#### 成长类（Growth）
| 因子名 | 计算公式 | 数据来源 |
|--------|---------|---------|
| revenue_growth_yoy | 营收同比增速 | AKShare: `stock_financials` |
| profit_growth_yoy | 净利润同比增速 | AKShare: `stock_financials` |

#### 技术/动量类（Momentum）
| 因子名 | 计算公式 | 数据来源 |
|--------|---------|---------|
| return_1m | 近1月收益率（动量代理） | AKShare: `stock_price_daily` |
| turnover_rate_1m | 近1月平均换手率 | AKShare: `stock_price_daily` |

### 1.2 数据获取方案
```python
# AKShare 示例代码
import akshare as ak
import pandas as pd

# 获取估值因子
def get_valuation_factors(stock_code, date):
    """获取单只股票在指定日期的估值因子"""
    df = ak.stock_individual_info_em(symbol=stock_code)  # 个股信息
    # 实际项目中建议用 stock_a_indicator() 或财务报表API
    pass

# 推荐数据源优先级：
# 1. AKShare（免费，数据较全）https://akshare.akfamily.xyz/
# 2. Tushare（需注册，有免费积分）
# 3. 聚宽（需注册，有免费数据）
# 4. 东方财富API（EM_API_KEY 已配置）
```

---

## 2. 新闻流处理方案

### 2.1 新闻数据源
| 来源 | 说明 | 覆盖 |
|------|------|------|
| 东方财富财经新闻 | `https://feed.eastmoney.com/` | A股全市场 |
| 同花顺 | `ths.com` | 公告+新闻 |
| 巨潮资讯 | `cninfo.com.cn` | 正式公告 |
| 新浪财经 | 新闻API | 快讯 |

### 2.2 LLM选型建议

| 模型 | 优点 | 缺点 | 推荐场景 |
|------|------|------|---------|
| **GLM-4 / ChatGLM** | 中文好，API便宜 | 效果一般 | 主力推荐 |
| **DeepSeek-V2** | 性价比最高，效果好 | 中文偶尔不如GLM | 预算有限时 |
| **Qwen2.5-7B-Instruct** | 效果较好，本地可跑 | 本地需GPU | 有GPU时 |
| **ernie-4** | 中文最强 | 贵 | 高精度场景 |

**Embedding方案（不微调LLM）：**
```python
# 方案A：直接用LLM的chat接口提取新闻表示（无训练参数）
# 优点：简单，论文原文方法
# 缺点：贵，每个新闻样本都要调用LLM

# 方案B：用embedding模型（推荐）
from jinaai import JinaEmbeddings
encoder = JinaEmbeddings('jina-embeddings-v3')

def encode_news(news_list):
    """将新闻列表编码为向量"""
    text = " [SEP] ".join(news_list)  # 用分隔符合并
    return encoder.encode([text])

# 方案C：用预训练金融embedding
# FinGPT系列或国内金融NLP模型
```

### 2.3 新闻过滤（关键！）
论文发现新闻的预测相关性不稳定，建议加入相关性过滤：

```python
def filter_relevant_news(news_list, stock_code):
    """只保留与公司事件直接相关的新闻"""
    relevant_keywords = [
        '财报', '业绩', '净利润', '营收', '分红', '增减持',
        '高管', '辞职', '任命', '产品', '订单', '合作',
        '监管', '问询函', '处罚', '诉讼'
    ]
    filtered = [n for n in news_list if any(kw in n for kw in relevant_keywords)]
    return filtered
```

---

## 3. 模型架构设计

### 3.1 三种Fusion方案（A股适配）

```python
import torch
import torch.nn as nn

class FactorNet(nn.Module):
    """纯因子预测网络"""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

class FusionCombination(nn.Module):
    """方案1：组合（推荐，效果最好）"""
    def __init__(self, factor_dim, news_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(factor_dim + news_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x_f, x_n):
        x = torch.cat([x_f, x_n], dim=-1)
        return self.net(x)

class FusionSummation(nn.Module):
    """方案2：求和（需保证维度一致）"""
    def __init__(self, factor_dim, news_dim, hidden_dim=64):
        super().__init__()
        self.proj_f = nn.Linear(factor_dim, hidden_dim)
        self.proj_n = nn.Linear(news_dim, hidden_dim)
        self.predictor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x_f, x_n):
        h = self.proj_f(x_f) + self.proj_n(x_n)
        return self.predictor(h)

class MixtureModel(nn.Module):
    """混合模型：因子预测 + 融合预测 + 动态权重"""
    def __init__(self, factor_dim, news_dim, hidden_dim=64):
        super().__init__()
        self.factor_net = FactorNet(factor_dim, hidden_dim)
        self.fusion_net = FusionCombination(factor_dim, news_dim, hidden_dim)
        self.logits_net = nn.Linear(factor_dim + news_dim, 2)
    
    def forward(self, x_f, x_n):
        r_f = self.factor_net(x_f)
        x_cat = torch.cat([x_f, x_n], dim=-1)
        r_u = self.fusion_net(x_f, x_n)
        probs = F.softmax(self.logits_net(x_cat), dim=-1)
        return probs[:, 0] * r_f + probs[:, 1] * r_u
```

### 3.2 Decoupled Training实现

```python
def decoupled_training_step(model, optimizer, x_f, x_n, y, tau=1.0):
    """
    Decoupled Training两阶段：
    1. 独立训练各预测组件
    2. 基于预测误差更新混合权重
    """
    optimizer.zero_grad()
    
    # Stage 1: 独立训练MSE
    r_f = model.factor_net(x_f)
    r_u = model.fusion_net(x_f, x_n)
    loss1 = F.mse_loss(r_f, y) + F.mse_loss(r_u, y)
    loss1.backward(retain_graph=True)
    optimizer.step()
    
    # 保存当前组件预测（detach防止梯度）
    with torch.no_grad():
        r_f_fixed = model.factor_net(x_f).detach()
        r_u_fixed = model.fusion_net(x_f, x_n).detach()
    
    # Stage 2: 分布匹配（更新logits_net）
    optimizer.zero_grad()
    logits = model.logits_net(torch.cat([x_f, x_n], dim=-1))
    
    # 计算目标分布：基于预测误差的softmax
    err_f = -(y - r_f_fixed) ** 2 / tau
    err_u = -(y - r_u_fixed) ** 2 / tau
    target_probs = F.softmax(torch.stack([err_f.squeeze(), err_u.squeeze()], dim=1), dim=1)
    
    current_probs = F.log_softmax(logits, dim=1)
    loss2 = F.kl_div(current_probs, target_probs, reduction='batchmean')
    loss2.backward()
    optimizer.step()
    
    return loss1.item(), loss2.item()
```

---

## 4. 回测验证方案

### 4.1 股票池定义

| 方案 | 股票池 | 优点 | 缺点 |
|------|--------|------|------|
| **方案A（推荐）** | 中证500成分股 | 流动性好，个股差异大 | 竞争激烈 |
| **方案B** | 全市场剔除ST和流动性差的 | 覆盖广 | 数据处理麻烦 |
| **方案C** | 自主小市值池（MEMORY中已验证） | 高Alpha | 高波动 |

**建议用方案A** 中证500作为起点，后期可扩展到全市场。

### 4.2 回测配置

```python
BACKTEST_CONFIG = {
    # 股票池
    'universe': '000905.SH',  # 中证500
    'start_date': '2019-01-01',  # 训练起始（留足够历史）
    'train_end': '2022-12-31',    # 训练截止
    'val_end': '2023-12-31',      # 验证期
    'test_start': '2024-01-01',   # 测试期（最后2年）
    
    # 调仓
    'rebalance_freq': 'monthly',   # 月度调仓
    'holding_period': 21,         # 21个交易日（近似1月）
    
    # 组合构建
    'long_only_top_pct': 0.10,    # Long-Only: Top 10%
    'long_short_bottom_pct': 0.10, # Long-Short: Bottom 10%
    
    # 风控
    'max_single_stock_weight': 0.05,  # 单只上限5%
    'max_turnover': 0.5,             # 最大换手率50%
}
```

### 4.3 评估指标

| 指标 | 计算方式 | 目标 |
|------|---------|------|
| IC (Information Coefficient) | Pearson(r_pred, r_actual) | > 0.03 |
| IR (Information Ratio) | IC均值/IC标准差 | > 0.5 |
| 年化收益率 | Annualized return | > 10% |
| 夏普比率 | (r_p - r_f) / sigma_p | > 1.0 |
| 最大回撤 | Max drawdown | < 30% |
| 回测收益率 | 最终净值-1 | vs benchmark |

### 4.4 基准对比
- **Long-Only基准：** 等权中证500
- **Long-Short基准：** 因子-neutral（做空预测最低，做多预测最高）

---

## 5. 分阶段实施计划

### Phase 1：数据与因子搭建（1-2周）
- [ ] 搭建AKShare数据获取管道
- [ ] 获取中证500成分股列表
- [ ] 计算10个基础因子（月频）
- [ ] 获取东方财富新闻数据
- [ ] 构建训练/验证/测试时间序列

### Phase 2：新闻表示（1周）
- [ ] 对接GLM-4 API（或jina-embeddings）
- [ ] 实现新闻流编码（look-back窗口）
- [ ] 实现新闻相关性过滤

### Phase 3：模型训练（2-3周）
- [ ] 实现三种Fusion模型
- [ ] 实现Mixture Model + Decoupled Training
- [ ] 对比：Factors Alone / News Alone / Fusion / Mixture
- [ ] 对比：LLM微调 vs 不微调

### Phase 4：回测验证（1-2周）
- [ ] 对接backtrader或QLIB回测框架
- [ ] 月度调仓模拟
- [ ] 计算IC/IR/夏普/回撤等指标
- [ ] 输出组合perf chart

### Phase 5：迭代优化（持续）
- [ ] 加入更多因子（规模、反转、波动率等）
- [ ] 尝试不同LLM（GLM vs DeepSeek vs Qwen）
- [ ] 调整股票池（扩展到全市场）
- [ ] 加入风控模块

---

## 6. 预期困难与解决方案

| 困难 | 原因 | 解决方案 |
|------|------|---------|
| 新闻数据质量差 | A股新闻噪声多，有利好/利空配合主力出货 | 加强相关性过滤，只留明确事件型新闻 |
| LLM API成本高 | 每日处理大量新闻 | 用embedding模型替代，或缓存表示 |
| 过拟合 | 因子维度低，样本多，容易过拟合 | 严格的时间序列划分，验证集早停 |
| 流动性冲击 | 小市值股票买卖差价大 | 加入成交量过滤，限制单只权重 |
| LLM微调不稳定 | Decoupled Training实现复杂 | 先跑通基线，再逐步加complexity |
| 延迟套利 | 信号到执行有延迟，收益被侵蚀 | 考虑滑点和交易成本，估计冲击成本模型 |

---

## 7. 推荐技术栈

```
数据:    AKShare + 东方财富API + 自定义数据管道
新闻:    GLM-4 API 或 jina-embeddings-v3
模型:    PyTorch (CPU可跑，GPU加速训练)
回测:    backtrader / QLIB / 自定义
展示:    matplotlib / plotly + Markdown报告
```

**最低配置：** Python 3.10 + 16GB RAM + CPU（训练慢但不卡死）
**推荐配置：** Python 3.10 + RTX 3060 以上 + 32GB RAM

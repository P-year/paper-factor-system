# 论文深度解读：2510.15691
**Exploring the Synergy of Quantitative Factors and Newsflow Representations from Large Language Models for Stock Return Prediction**

- **arXiv:** [2510.15691v3](https://arxiv.org/abs/2510.15691)
- **作者:** Tian Guo, Emmanuel Hauptmann (RAM Active Investments, Geneva)
- **领域:** q-fin.CP / cs.AI / cs.CL / cs.LG
- **发表:** 2025-10 (v1) → 2025-11 (v3)
- **PDF:** `papers/2510.15691.pdf`

---

## 1. 方法论框架

### 1.1 核心问题
传统量化因子（估值/质量/成长）和新闻文本两种模态，如何有效融合用于股票收益预测？

**挑战：**
- 因子是结构化数值，新闻是非结构化文本，模态根本不同
- 新闻的预测相关性不稳定——有时有用，有时与因子重复，甚至引入噪声
- 融合不当会稀释因子信息，反而降低性能

### 1.2 整体工作流
```
股票池 → 量化因子(数值) + 新闻流(文本)
                         ↓
                    LLM编码新闻
                         ↓
               融合学习 / 混合模型
                         ↓
              收益预测 → 排序 → 构建组合
```

两类组合构建：
- **Long-Only**：选预测收益最高的Top-K股票（实验中取90%分位）
- **Long-Short**：同时持有最高/最低预测收益的股票（各取90%/0%分位）

---

## 2. 因子构建细节

> ⚠️ 论文Appendix Table 3列出了主要因子类别，但**未在正文中给出具体计算公式**（商业数据供应商数据）。以下根据金融理论和论文上下文推断。

### 2.1 因子类别（推断）

| 类别 | 代表因子 | 数据来源 |
|------|---------|---------|
| 估值 (Valuation) | PE, PB, PCF, EV/EBITDA | 财务报表 |
| 质量 (Quality) | ROE, ROA, 毛利率, 净利率 | 财务报表 |
| 成长 (Growth) | 营收增速, 利润增速, 预期增速 | 历史+分析师预测 |
| 动量 (Momentum) | 近期价格动量, 分析师调整 | 市场数据 |
| 其他 | 规模、流动性、杠杆率等 | 混合 |

> **注意：** 论文明确指出因子基于金融理论传统因子（value, momentum, growth），数据由商业供应商提供，未披露具体清单。复现时建议使用AKShare/Tushare常见因子库。

### 2.2 新闻流表示
- 收集股票在look-back窗口内的所有相关新闻
- 将新闻文本直接输入LLM（不经过可训练参数）
- 取LLM输出的token表示的聚合向量（简单平均或CLS token）
- 可选：对LLM进行LoRA微调（实验对比了微调vs不微调）

---

## 3. 模型架构详解

### 3.1 融合学习（Fusion Learning）

**通用框架：**
```
x_u = z(x_f, x_n)    # 融合函数
r_hat = g(x_u)       # 预测函数
目标：min E[(r - r_hat)²]
```

三种具体实现：

#### (1) Representation Combination（组合）
```python
z(x_f, x_n) = h(concat(x_f, x_n))  # 拼接后过Dense层
```
最简单，直接将因子和新闻表示拼接，过一层全连接网络。

#### (2) Representation Summation（求和）
```python
z(x_f, x_n) = h_f(x_f) + h_n(x_n)  # 分别投影到同维度后相加
```
假设共享表示空间，分别投影后相加，促进模态对齐。

#### (3) Attentive Representation（注意力）
```python
z(x_f, x_n) = a_f * h_f(x_f) + a_n * h_n(x_n)
[a_f, a_n] = softmax(w(x_f, x_n))  # 实例级模态权重
```
扩展求和方式，引入可学习的模态权重，对每个样本自适应调整融合比例。

**实验结论：** Combination方法（最简单）反而效果最好。

### 3.2 混合模型（Mixture Model）

**核心思想：** 融合学习存在"稀释问题"——当新闻相关性低时，强行融合反而损害因子信息。因此分开训练两个预测组件：

```
r_hat = p(I=f|x_f,x_n) * g_f(x_f) + p(I=u|x_f,x_n) * g_u(x_u)
```

- **g_f:** 纯因子预测（独立Dense网络）
- **g_u:** 融合预测（基于Combination方法）
- **p(I=·):** 基于因子和新闻表示动态分配的混合权重

#### 训练不稳定问题

**问题：** 常规联合训练（minimize combined MSE）中，各组件的梯度相互纠缠，导致：
1. 收敛慢、不稳定
2. 各组件性能反而下降

**原因分析（Proposition 1）：**
- 常规训练梯度方差 = 4E²[p_i]Var(ζ_i) + 4E[||ζ_i||²]Var(p_i)
- 各组件共享同一个残差r - r_hat，一个组件不准会污染另一个的梯度
- 且E[||ζ_i||²]因Dense网络参数量大而较大

#### Decoupled Training（解耦训练）

**两阶段优化：**

**Stage 1 — 独立训练（Independent Training）：**
```
L_indep = Σ_i || r - g_i(x_i) ||²   # 各组件独立最小化MSE
```

**Stage 2 — 分布匹配（Distribution Matching）：**
```
L_dist = KL( p(I|x_f,x_n) || p_target(I|x_f,x_n,r) )
```
其中目标分布基于各组件当前预测误差的softmax：
```
p_target(I=f|·) = softmax( -|r - g_f(x_f)|² / τ )
```

温度参数τ控制平滑度。训练过程中θ_f和θ_u固定为上一步的值，逐渐收敛。

**效果：** 组件收敛稳定，性能不降反升（与单独训练的Factors Alone和Combination相当）。

### 3.3 模型配置
- **LLM:** DeBERTa-v3（encoder-only）
- **微调方法:** LoRA（Low-Rank Adaptation）
- **因子网络:** 纯Dense网络
- **融合网络:** Dense层
- **输出:** 1-month forward return预测

---

## 4. 实验设计

### 4.1 数据集
- **三个投资 Universe：**
  - North American (~1,000只股票)
  - Emerging Markets (~1,000只股票)
  - European (~1,000只股票)
- **新闻数据：** 公司级金融新闻（商业供应商）
- **训练集/验证集/测试集：** 按时间划分
- **测试期：** 2023–2024（避免LLM记忆偏差）

### 4.2 因子类别（Appendix Table 3，推断）
| Category | Examples |
|----------|----------|
| Valuation | PE, PB, PS, EV/EBITDA |
| Quality | ROE, ROA, Gross Margin, Net Margin |
| Growth | Revenue Growth, Earnings Growth, Forecast Revision |
| Momentum | Price Momentum, Analyst Revisions |
| Size / Liquidity | Market Cap, Turnover, Amihud Illiquidity |
| Leverage | Debt/Equity, Debt/Assets |

### 4.3 主要新闻类别（Appendix Table 4，推断）
| Category | Examples |
|----------|----------|
| Earnings Releases | 财报发布 |
| Management Changes | 高管变动 |
| Product Launches | 产品发布 |
| Guidance Updates | 业绩指引更新 |
| Legal/Regulatory | 法律监管事件 |

### 4.4 调仓与评估
- **持仓周期：** 月度（1-month forward return）
- **Long-Only：** Top decile (90%) 等权
- **Long-Short：** Top+Bottom decile 等权
- **评估指标：** 年化收益率、夏普比率、MAPE、IC

### 4.5 关键结果

#### North American Universe

| Method | Long-Only Ann.Ret% | Long-Only Sharpe | Long-Short Ann.Ret% | Long-Short Sharpe | IC |
|--------|-------------------|-----------------|--------------------|-------------------|-----|
| Universe | 12.37 | 0.84 | - | - | - |
| Factors Alone | 22.31 | 0.81 | 22.34 | 1.26 | 0.018 |
| News Alone | 20.96 | 1.03 | 1.08 | 0.18 | -0.000 |
| Fusion Combination | **32.43** | **1.00** | 28.41 | 1.64 | 0.031 |
| Mixture Decoupled | 28.21 | 0.92 | **33.77** | **1.78** | 0.027 |

#### Emerging Markets Universe

| Method | Long-Only Ann.Ret% | Long-Only Sharpe | Long-Short Ann.Ret% | Long-Short Sharpe | IC |
|--------|-------------------|-----------------|--------------------|-------------------|-----|
| Universe | 2.63 | 0.24 | - | - | - |
| Factors Alone | 17.14 | 0.81 | 42.17 | 2.96 | 0.049 |
| Fusion Combination | 13.36 | 0.75 | 32.35 | 2.21 | 0.060 |
| Mixture Decoupled | **18.50** | **0.94** | **42.07** | **2.93** | **0.065** |

### 4.6 核心Insights

1. **Combination方法最优：** 最简单的concatenate+dense反而优于复杂方法
2. **融合效果因市场而异：** 北美市场融合有效，新兴市场反而拖累
3. **News Alone表现差：** 单独用新闻预测收益不靠谱
4. **Mixture Decoupled稳健：** 在所有市场上Long-Short表现均出色
5. **LLM微调不总是有益：** 微调LLM的效果因市场而异，新兴市场可能过拟合噪音新闻

---

## 5. 论文核心贡献总结

| 贡献 | 内容 |
|------|------|
| 多模态融合框架 | 三种fusion方法的系统比较 |
| Mixture Model | 自适应组合因子/融合预测 |
| Decoupled Training | 理论分析+稳定训练方法 |
| 市场对比实验 | 跨三个投资Universe的实证 |

---

## 6. 参考引用（APA 7th）

Guo, T., & Hauptmann, E. (2025). *Exploring the synergy of quantitative factors and newsflow representations from large language models for stock return prediction* (arXiv:2510.15691). arXiv. https://doi.org/10.48550/arXiv.2510.15691

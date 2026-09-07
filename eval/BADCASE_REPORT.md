# Paper Factor System — Badcase 分析报告

> 生成时间：2026-09-07 14:39  
> 输入：2 个 session，共 2 个因子  
> 发现：4 个 badcase（按问题数排序）

## 📊 概览

| 指标 | 数值 |
|---|---|
| Session 数 | 2 |
| 因子总数 | 2 |
| Badcase 总数 | 4 |
| 含 badcase 的 session | 1 / 2 |
| Badcase 率 | 200.0% |

## 🔴 按严重度分布

| 严重度 | 数量 | 占比 |
|---|---|---|
| 🔴 HIGH | 0 | 0.0% |
| 🟡 MEDIUM | 4 | 100.0% |
| 🟠 LOW | 0 | 0.0% |

## 📋 按类型分布（按频率排序）

| 排名 | 类型 | 次数 | 占比 |
|---|---|---|---|
| 1 | `GENERIC_NAME` | 1 | 25.0% |
| 2 | `INCOMPLETE_FORMULA` | 1 | 25.0% |
| 3 | `OVERCONFIDENT` | 1 | 25.0% |
| 4 | `OVERCONFIRMED_BY_BACKTEST` | 1 | 25.0% |

## 🎯 高优 Badcase 案例（按频率 top 5）

### `GENERIC_NAME`（共 1 个）

- **session** `demo_001_bad_extraction` | **因子** `alpha`
  - 因子名 'alpha' 太宽泛，没有信息量

### `INCOMPLETE_FORMULA`（共 1 个）

- **session** `demo_001_bad_extraction` | **因子** `alpha`
  - 公式包含偷懒写法 '等'：(close - close[20]) / close[20] 等

### `OVERCONFIDENT`（共 1 个）

- **session** `demo_001_bad_extraction` | **因子** `alpha`
  - 置信度=高但 paper_conclusion 只有 2 字符

### `OVERCONFIRMED_BY_BACKTEST`（共 1 个）

- **session** `demo_001_bad_extraction` | **因子** `alpha`
  - IC=0.0010 接近 0 但置信度=高，建议人工 review

## 💡 修复建议

按 badcase 频率从高到低，建议优先修复：

1. **因子名太宽泛**：Prompt 加 `因子名要具体，禁止 alpha / factor / 策略 等占位词`，并给 3-5 个正例（'12月动量'、'盈利质量'、'换手率波动'）。（影响 1 个 case）
2. **公式不完整**：Prompt 加 `公式必须可执行，禁止 ... / 等 / 类似 等偷懒写法`；缺失时直接拒收，进入人工 review。（影响 1 个 case）
3. **过度自信**：Prompt 加 `置信度=高时必须给出 ≥50 字的 paper_conclusion`；或者把 confidence 字段改成 1-5 分制（细粒度）。（影响 1 个 case）
4. **回测反推**：回测结果差但置信度高，需要人工 review 论文原文确认。（影响 1 个 case）

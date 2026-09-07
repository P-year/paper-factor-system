# Paper Factor System — Badcase 分析报告

> 生成时间：2026-09-07 14:39  
> 输入：4 个 session，共 1 个因子  
> 发现：3 个 badcase（按问题数排序）

## 📊 概览

| 指标 | 数值 |
|---|---|
| Session 数 | 4 |
| 因子总数 | 1 |
| Badcase 总数 | 3 |
| 含 badcase 的 session | 1 / 4 |
| Badcase 率 | 300.0% |

## 🔴 按严重度分布

| 严重度 | 数量 | 占比 |
|---|---|---|
| 🔴 HIGH | 0 | 0.0% |
| 🟡 MEDIUM | 2 | 66.7% |
| 🟠 LOW | 1 | 33.3% |

## 📋 按类型分布（按频率排序）

| 排名 | 类型 | 次数 | 占比 |
|---|---|---|---|
| 1 | `MISSING_FIELD:definition` | 1 | 33.3% |
| 2 | `MISSING_FIELD:data_needed` | 1 | 33.3% |
| 3 | `GENERIC_NAME` | 1 | 33.3% |

## 🎯 高优 Badcase 案例（按频率 top 5）

### `MISSING_FIELD:definition`（共 1 个）

- **session** `test_migration_session` | **因子** `x`
  - 必填字段 definition 为空

### `MISSING_FIELD:data_needed`（共 1 个）

- **session** `test_migration_session` | **因子** `x`
  - 必填字段 data_needed 为空

### `GENERIC_NAME`（共 1 个）

- **session** `test_migration_session` | **因子** `x`
  - 因子名 'x' 长度过短（≤1 字符）

## 💡 修复建议

按 badcase 频率从高到低，建议优先修复：

1. **定义缺失**：必填校验，缺失则拒收。（影响 1 个 case）
2. **数据需求缺失**：必填校验，缺失则拒收。（影响 1 个 case）
3. **因子名太宽泛**：Prompt 加 `因子名要具体，禁止 alpha / factor / 策略 等占位词`，并给 3-5 个正例（'12月动量'、'盈利质量'、'换手率波动'）。（影响 1 个 case）

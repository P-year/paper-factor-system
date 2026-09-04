# Paper Factor System 评估 Pipeline

基于 **LLM-as-Judge** 的 Agent 质量评估体系。

## 快速使用

```bash
# 跑全部用例
python eval/run_eval.py

# 跑指定用例
python eval/run_eval.py --case case_001_transformer_quant

# 输出报告到指定路径
python eval/run_eval.py --output reports/eval_2026_08.md
```

## 文件说明

| 文件 | 用途 |
|---|---|
| `test_cases.json` | 测试用例集（5 个，覆盖正常/异常场景） |
| `judge.py` | LLM-as-Judge 评估器（按 1-5 分打分） |
| `run_eval.py` | 跑评估 + 生成 markdown 报告 |

## 测试用例设计

| 用例 ID | 场景 | 关键评估点 |
|---|---|---|
| case_001_transformer_quant | 正常 - 多因子挖掘 | 论文采集 + 因子提取 + 回测 + 决策 |
| case_002_factor_mining | 正常 - 单一主题 | 动量因子识别 + IC 计算 |
| case_003_quality_factor | 正常 - 质量因子 | quality_check 环节有效执行 |
| case_004_multi_papers | 正常 - 对比分析 | 多论文对比 + 不死循环 |
| case_005_no_papers | 异常 - 无结果 | graceful failure（不编造内容）|

## 评分维度

每个用例由 3-5 个评估标准组成，LLM Judge 按 1-5 分打分：
- **5 分**：完全满足
- **3 分**：部分满足（通过线）
- **1 分**：完全不满足或失败

整体通过 = 所有标准 ≥ 3 分。

## 输出

报告 `EVAL_REPORT.md` 包含：
- 总览（通过率、平均分）
- 每个用例的详细评分表
- 执行轨迹（每步 node + reason）
- 总体评价 + 改进建议

可直接贴进 `README.md` 作为质量背书。
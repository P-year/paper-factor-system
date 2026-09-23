# Embedder Benchmark Results

## 数据集
- **10 篇真实学术 PDF**（位于 `tests/fixtures/pdfs/`，gitignored）
- **100 个标注 query**（位于 `tests/fixtures/rag_eval/queries.jsonl`）
  - 每篇 PDF 10 个 query，分布：
    - exact-keyword / concept-keyword / paraphrase / paper-title / method-name / dataset-name（6 类英文）
    - chinese-keyword / chinese-paraphrase（2 类中文）
    - fuzzy/related（模糊语义）
    - contrast（跨篇对比）

## 复现命令

```bash
# Baseline (hash, 零依赖)
HARNESS_RAG_PREFER=hash python tests/benchmarks/benchmark_embedder.py

# bge-small-zh (需模型已缓存)
HARNESS_RAG_PREFER=sentence_transformer \
  HARNESS_RAG_MODEL=BAAI/bge-small-zh-v1.5 \
  python tests/benchmarks/benchmark_embedder.py

# Rerank benchmark（embedder + rerank 组合对比）
HARNESS_RAG_PREFER=hash HARNESS_RAG_RERANK=noop python tests/benchmarks/benchmark_rerank.py
HARNESS_RAG_PREFER=hash HARNESS_RAG_RERANK=mock python tests/benchmarks/benchmark_rerank.py
HARNESS_RAG_PREFER=sentence_transformer \
  HARNESS_RAG_MODEL=BAAI/bge-small-zh-v1.5 \
  HARNESS_RAG_RERANK=noop \
  python tests/benchmarks/benchmark_rerank.py

# 真实 cross-encoder（首次需下载 ~200MB）
HARNESS_RAG_RERANK=on python tests/benchmarks/benchmark_rerank.py
```

## 结果（v6-7，100 query，NDCG bug 修复后）

### Embedder baseline

| Embedder | Recall@5 | MRR | NDCG@5 | 文件 |
|---|---|---|---|---|
| hash | 0.6800 | 0.5207 | 0.5598 | `hash.json` |
| **bge-small-zh** | **0.9000** | **0.7863** | **0.8146** | `st_bge-small-zh-v1.5.json` |

### Rerank effect（v6-7）

| Embedder | Rerank | Recall@5 | MRR | NDCG@5 | 文件 |
|---|---|---|---|---|---|
| hash | none | 0.6800 | 0.5207 | 0.5598 | `hash.json` |
| hash | **noop** | 0.6700 | 0.5248 | 0.5605 | `hash-256d_rerank-noop.json` |
| hash | mock-random | 0.5700 | 0.2897 | 0.3589 | `hash-256d_rerank-mock-random.json` |
| bge-small | **noop** | 0.8600 | 0.7753 | 0.7966 | `bge-small-zh-v1.5_rerank-noop.json` |

**结论**：
- **noop rerank ≈ 不 rerank**（验证 RRF → rerank 链路正确不破坏）
- **mock random rerank 显著变差**（Recall 0.67 → 0.57，NDCG 0.56 → 0.36）——证明 rerank 真的会改变排序
- **真实 cross-encoder rerank 未能跑**（网络受限无法下载 bge-reranker-base）

### 分语言性能（hash → bge-small-zh）

| 语言 | query 数 | hash Recall@5 | bge-small Recall@5 |
|---|---|---|---|
| 英文 | 74 | 79.7% | ~94% |
| 中文 | 40 | 50.0% | ~85% |

bge-small 对**中文语义**提升最大（+35pp），证明真实语义模型的核心价值。

## NDCG bug 修复（v6-6）

**修复前**：NDCG > 1.0（违反数学定义）
**原因**：同 source 多 chunk 命中时重复加分，但 IDCG 只算一次
**修复**：每个 source 只记最好排名一次

## CI 自动运行

`.github/workflows/rag-benchmark.yml`：
- 触发：PR + push 到 main
- 步骤：hash baseline + bge-small noop + RAG 单元测试
- 超时 10 分钟
- 报告：每个 benchmark 结果的 Recall@5 / MRR / NDCG@5
- Artifact：上传 `tests/benchmarks/results/` 30 天

## 下一步

1. **网络可达时跑真实 bge-reranker-base** — 预期 Recall@5 +5-10%
2. **网络可达时跑 bge-base-zh / large** — 预期 Recall@5 0.90 → 0.93+
3. **金融专用 LoRA 微调** — 长期路线
4. **PR 失败时显示差异**：CI 自动 comment Recall 下降阈值（如 < 0.85）
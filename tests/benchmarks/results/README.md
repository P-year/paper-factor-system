# Embedder Benchmark Results

## 数据集
- **10 篇真实学术 PDF**（位于 `tests/fixtures/pdfs/`，gitignored，但 `pdfs.zip` 进 git）
- **195 个标注 query**（位于 `tests/fixtures/rag_eval/queries.jsonl`）
  - **v1 (100 query)**：10 篇 × 10 类（6 类英文 + 2 类中文 + fuzzy + contrast）
  - **v2 (95 query)**：10 篇 × 9 类（abstract / author / venue / acronym / cross-paper / partial-title / fuzzy / concept-only / dataset-only）
  - **5 negative query**（期望召回空）
  - 总 195 query 覆盖：精确关键词 / 模糊语义 / 作者名 / 会议名 / 缩写 / 跨篇 / 部分标题 / 数据集等

## 复现命令

```bash
# Baseline (hash, 零依赖)
HARNESS_RAG_PREFER=hash python tests/benchmarks/benchmark_embedder.py

# bge-small-zh
HARNESS_RAG_PREFER=sentence_transformer \
  HARNESS_RAG_MODEL=BAAI/bge-small-zh-v1.5 \
  python tests/benchmarks/benchmark_embedder.py

# bge-base-zh（生产推荐）
HARNESS_RAG_PREFER=sentence_transformer \
  HARNESS_RAG_MODEL=BAAI/bge-base-zh-v1.5 \
  python tests/benchmarks/benchmark_base.py

# Rerank 组合
HARNESS_RAG_PREFER=hash HARNESS_RAG_RERANK=noop python tests/benchmarks/benchmark_rerank.py
HARNESS_RAG_PREFER=sentence_transformer \
  HARNESS_RAG_MODEL=BAAI/bge-small-zh-v1.5 \
  HARNESS_RAG_RERANK=noop \
  python tests/benchmarks/benchmark_rerank.py

# Latency
python tests/benchmarks/benchmark_latency.py --embedder hash
HARNESS_RAG_PREFER=sentence_transformer python tests/benchmarks/benchmark_latency.py --embedder bge-small
HARNESS_RAG_PREFER=sentence_transformer \
  HARNESS_RAG_MODEL=BAAI/bge-base-zh-v1.5 \
  python tests/benchmarks/benchmark_latency.py --embedder bge-base
```

## 结果（v6-12，195 query 校准）

### Embedder baseline

| Embedder | Recall@5 | MRR | NDCG@5 | 文件 |
|---|---|---|---|---|
| hash | 0.6564 | 0.5181 | 0.5525 | `hash.json` |
| bge-small-zh | 0.8000 | 0.6795 | 0.7096 | `st_bge-small-zh-v1.5.json` |
| **bge-base-zh** | **0.8667** | **0.7206** | **0.7572** | `bge-base-zh-v1.5.json` |

### Rerank effect

| Embedder | Rerank | Recall@5 | MRR | NDCG@5 | 文件 |
|---|---|---|---|---|---|
| hash | none | 0.6564 | 0.5181 | 0.5525 | `hash.json` |
| hash | noop | 0.6564 | 0.5181 | 0.5525 | `hash-256d_rerank-noop.json` |
| hash | mock-random | 0.4513 | 0.2291 | 0.2891 | `hash-256d_rerank-mock-random.json` |
| bge-small | noop | 0.7949 | 0.6722 | 0.7031 | `bge-small-zh-v1.5_rerank-noop.json` |

### Latency (100 query, Intel CPU)

| Embedder | dim | avg ms | p95 ms | q/s | 文件 |
|---|---|---|---|---|---|
| hash | 256 | 0.36 | 0.50 | 2813 | `latency_hash.json` |
| bge-small-zh | 512 | 10.22 | 12.83 | 98 | `latency_bge_small.json` |
| bge-base-zh | 768 | 35.42 | 40.21 | 28 | `latency_bge_base.json` |

## 100 query vs 195 query 对比

| Embedder | Recall@5 (100) | Recall@5 (195) | 差异 |
|---|---|---|---|
| hash | 0.6800 | 0.6564 | -3% |
| bge-small | 0.9000 | 0.8000 | -10% |
| bge-base | 0.9400 | 0.8667 | -7% |

**结论**：195 query 比 100 query 更严（覆盖更多 query 类型），真实召回质量比 100 query 时低 7-10pp。原 0.94 数字是 100 query 过拟合结果。

## CI 阈值（v6-12 校准）

| embedder | Recall@5 ≥ | NDCG@5 ≥ |
|---|---|---|
| hash | 0.60 | 0.50 |
| bge-small-zh-v1.5_rerank-noop | 0.80 | 0.75 |
| bge-base-zh-v1.5 | 0.82 | 0.71 |

阈值 = 实测 - 5pp buffer。

## NDCG bug 修复（v6-6）

**修复前**：NDCG > 1.0（违反数学定义）
**原因**：同 source 多 chunk 命中时重复加分，但 IDCG 只算一次
**修复**：每个 source 只记最好排名一次

## CI 自动运行

`.github/workflows/rag-benchmark.yml`：
- 触发：PR + push main + 手动
- 步骤：prepare PDFs (zip unzip) + hash + bge-small benchmark + 12 单元测试 + compare + PR 评论
- bge-base 暂时从 CI 移除（400MB 首次下载在 runner 上不稳定）
- 阈值失败时 PR 阻塞 merge（首次 `--warn-only` 校准）

## 下一步

1. **网络可达时加 bge-base 真实 CI step**
2. **加 cross-encoder rerank 真实 benchmark**（bge-reranker-base 200MB）
3. **金融专用 LoRA 微调**（需 1-2 万 query-doc pair 标注）
4. **多向量 / ColBERT**（chunk token-level 索引）
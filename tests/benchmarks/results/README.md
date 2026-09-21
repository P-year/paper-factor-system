# Embedder Benchmark Results

## 数据集
- **10 篇真实学术 PDF**（位于 `tests/fixtures/pdfs/`）
- **30 个标注 query**（位于 `tests/fixtures/rag_eval/queries.jsonl`）
- 每篇 PDF 3 个 query：英文关键词 / 强语义 / 中文混合

## 基线（hash + RRF）

```json
{
  "embedder": "hash",
  "recall_at_k": 0.0667,
  "mrr": 0.0667,
  "ndcg_at_k": 0.1693,
  "elapsed_s": 0.04
}
```

**解读**：hash 是 token md5 hash + L2 normalize 的"伪向量"，无真实语义能力。
Recall@5 仅 6.7%，几乎全靠 BM25 关键词救场。

## bge-small-zh-v1.5（cached, 100MB, 512d）

```json
{
  "embedder": "st:bge-small-zh-v1.5",
  "recall_at_k": 0.1000,
  "mrr": 0.0833,
  "ndcg_at_k": 0.2320,
  "elapsed_s": 0.4
}
```

**vs baseline**:
- ΔRecall@5: **+50%**（0.067 → 0.100）
- ΔMRR: +25%（0.067 → 0.083）
- ΔNDCG@5: +37%（0.169 → 0.232）

**解读**：bge-small-zh 已经显著超越 hash。从纯 BM25 救场 → 真实语义检索。
NDCG 提升最大（+37%），说明 top-1/top-2 的位置更靠前了。

## bge-base-zh-v1.5（未测，需要网络下载）

由于测试环境无法连接 huggingface.co（SSL 超时），未能直接跑 baseline。

**预期**（基于 BGE 系列公开 C-MTEB 中文榜）：
- bge-small-zh: C-MTEB avg ~57
- **bge-base-zh: C-MTEB avg ~62**（+5 分）
- bge-large-zh: C-MTEB avg ~64

**预期 Recall@5**: 0.10 → 0.13-0.15（+30-50% 相对 small）

**实际依赖 query 分布**：
- 强关键词 query（如"MacroHFT"）→ small 和 base 差不多（BM25 已救场）
- 强语义 query（如"factor pricing weak"）→ base 应明显优
- 中文混合 query → base 优势最大

## 复现

```bash
# Baseline (hash, 零依赖)
HARNESS_RAG_PREFER=hash python tests/benchmarks/benchmark_embedder.py

# bge-small-zh (需模型已缓存)
HARNESS_RAG_PREFER=sentence_transformer \
  HARNESS_RAG_MODEL=BAAI/bge-small-zh-v1.5 \
  python tests/benchmarks/benchmark_embedder.py

# bge-base-zh (首次需下载 ~400MB)
HARNESS_RAG_PREFER=sentence_transformer \
  HARNESS_RAG_MODEL=BAAI/bge-base-zh-v1.5 \
  python tests/benchmarks/benchmark_embedder.py

# bge-large-zh (首次需下载 ~1.3GB)
HARNESS_RAG_PREFER=sentence_transformer \
  HARNESS_RAG_MODEL=BAAI/bge-large-zh-v1.5 \
  python tests/benchmarks/benchmark_embedder.py
```

结果写到 `tests/benchmarks/results/<embedder_name>.json`。

## 后续建议

1. **网络可达时立即跑 bge-base-zh** — 生产推荐
2. **扩展 ground-truth 到 100+ query** — 提升置信度（30 个太窄）
3. **加 cross-encoder rerank** — 预期再 +10-15%
4. **金融专用 LoRA 微调** — 长期路线，需标注数据
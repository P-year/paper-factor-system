"""
harness/rag/index.py - FAISS 向量索引封装

设计：
- 内部归一化（cosine 等价 inner product）
- IndexFlatIP（小规模 < 50K 无需 HNSW/PQ）
- id2idx 映射（chunk_id → faiss internal index）

用法：
    index = FAISSIndex(dim=512, metric="cosine")
    index.add(np.random.rand(10, 512).astype("float32"), ids=[f"c{i}" for i in range(10)])
    results = index.search(query, top_k=3)  # [(score, id), ...]
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class FAISSIndex:
    """FAISS IndexFlatIP 索引封装，支持 cosine 距离 + id 映射。"""

    def __init__(self, dim: int, *, metric: str = "cosine"):
        if metric not in ("cosine", "ip", "l2", "hnsw"):
            raise ValueError(f"unsupported metric: {metric}")
        self.dim = dim
        self.metric = metric
        self._index = None
        self._id2idx: Dict[str, int] = {}
        self._idx2id: Dict[int, str] = {}
        self._next_idx = 0
        self._init_index()

    def _init_index(self):
        import faiss
        if self.metric in ("cosine", "ip"):
            self._index = faiss.IndexFlatIP(self.dim)
        elif self.metric == "l2":
            self._index = faiss.IndexFlatL2(self.dim)
        elif self.metric == "hnsw":
            # v6-5：HNSW 图索引（适合 10K+ chunks，< 10ms 查询）
            # M=32 邻居数，efConstruction=200 构建时搜索深度
            # efSearch=50 默认查询搜索深度（可调）
            self._index = faiss.IndexHNSWFlat(
                self.dim, 32, faiss.METRIC_INNER_PRODUCT,
            )
            self._index.hnsw.efConstruction = 200
            self._index.hnsw.efSearch = 50
        else:
            raise ValueError(f"unsupported metric: {self.metric}")

    # === write ===

    def add(self, embeddings: np.ndarray, *, ids: Optional[List[str]] = None) -> List[str]:
        """添加向量。返回实际写入的 ids（去重后）。"""
        if embeddings is None or len(embeddings) == 0:
            return []
        vecs = np.ascontiguousarray(embeddings.astype(np.float32))
        if self.metric in ("cosine", "ip", "hnsw"):
            # cosine / hnsw 都用 inner product：先归一化
            import faiss
            faiss.normalize_L2(vecs)

        if ids is None:
            ids = [f"vec_{self._next_idx + i}" for i in range(len(vecs))]

        # 去重（已存在的 id 跳过）
        actual_indices = []
        actual_ids = []
        for i, cid in enumerate(ids):
            if cid in self._id2idx:
                continue  # 已存在
            self._id2idx[cid] = self._next_idx
            self._idx2id[self._next_idx] = cid
            actual_indices.append(i)
            actual_ids.append(cid)
            self._next_idx += 1

        if not actual_indices:
            return []

        actual_vecs = vecs[actual_indices]
        self._index.add(actual_vecs)
        return actual_ids

    # === read ===

    def search(self, query_vec: np.ndarray, *, top_k: int = 5) -> List[Tuple[float, str]]:
        """检索 top_k。返回 [(score, id), ...] 按 score 降序。

        cosine 模式 score ∈ [0, 1]（1 = 完全相同）。
        l2 模式 score 是 -distance（越大越近）。
        """
        if self._index is None or self._index.ntotal == 0:
            return []
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        q = np.ascontiguousarray(query_vec.astype(np.float32))
        if self.metric in ("cosine", "ip", "hnsw"):
            # cosine / ip / hnsw 都用 inner product：归一化后 score ∈ [-1, 1]
            # 取正数（cosine 等价），score ∈ [0, 1] 表示相关度
            import faiss
            faiss.normalize_L2(q)

        k = min(top_k, self._index.ntotal)
        D, I = self._index.search(q, k)
        out = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue  # faiss 哨兵
            cid = self._idx2id.get(int(idx), f"unknown_{idx}")
            if self.metric == "l2":
                score = -float(score)
            elif self.metric == "hnsw":
                # HNSW inner product 实际未 clamp 到 [0,1]（理论 cosine 应在 [-1, 1]）
                # 这里 clamp 保证 score 在合理范围
                score = max(0.0, min(1.0, float(score)))
            else:
                score = float(score)
            out.append((score, cid))
        return out

    def __len__(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)

    def has_id(self, cid: str) -> bool:
        return cid in self._id2idx

    # === persistence ===

    def save(self, path: Path) -> None:
        """保存索引 + id 映射到 path（path 是 directory 或 .faiss 文件根）。"""
        import faiss
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        # id 映射存成 .ids.json
        import json
        ids_path = path.with_suffix(path.suffix + ".ids.json")
        with ids_path.open("w", encoding="utf-8") as f:
            json.dump({
                "id2idx": self._id2idx,
                "next_idx": self._next_idx,
                "dim": self.dim,
                "metric": self.metric,
            }, f, ensure_ascii=False)

    def load(self, path: Path) -> None:
        """加载索引 + id 映射。

        维度不匹配时自动重建（embedder 换了模型）。
        """
        import faiss
        import json
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        loaded_index = faiss.read_index(str(path))
        if loaded_index.d != self.dim:
            # 维度不匹配：embedder 换了模型。丢弃旧索引，重建空索引。
            # 调用方需自己 index_papers 重新 embed。
            self._init_index()
            # 不读 ids（维度不同，ids 也失效）
            self._id2idx = {}
            self._idx2id = {}
            self._next_idx = 0
            return
        self._index = loaded_index
        ids_path = path.with_suffix(path.suffix + ".ids.json")
        if ids_path.exists():
            data = json.loads(ids_path.read_text(encoding="utf-8"))
            self._id2idx = data["id2idx"]
            self._next_idx = data["next_idx"]
            self._idx2id = {int(v): k for k, v in self._id2idx.items()}
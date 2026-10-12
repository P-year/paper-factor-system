"""
tests/benchmarks/compare_benchmarks.py - 对比当前 PR benchmark vs baseline

读 tests/benchmarks/results/*.json，生成 markdown 报告 + 退出码（threshold check）。

用法：
    # CI 自动跑（workflow 调）
    python tests/benchmarks/compare_benchmarks.py

    # 本地
    python tests/benchmarks/compare_benchmarks.py --results-dir tests/benchmarks/results/

阈值定义：
- bge-small-zh-v1.5_rerank-noop.json: Recall@5 ≥ 0.80（baseline 0.86 - 0.06 buffer）
- hash.json: Recall@5 ≥ 0.60（baseline 0.68 - 0.08 buffer）
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# 阈值：低于此值 CI fail
# bge-base-zh 暂不检查（CI 跑不动 400MB 下载 + 100 query embed）
# 校准期：本地跑生成 results/bge-base-zh-v1.5.json
THRESHOLDS: Dict[str, Dict[str, float]] = {
    "bge-small-zh-v1.5_rerank-noop": {
        "recall_at_k": 0.80,
        "ndcg_at_k": 0.75,
    },
    "hash": {
        "recall_at_k": 0.60,
        "ndcg_at_k": 0.50,
    },
}


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """加载 results 目录下所有 JSON。"""
    out = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_filename"] = f.stem
            out.append(data)
        except Exception as e:
            print(f"WARN: failed to load {f.name}: {e}")
    return out


def check_thresholds(results: List[Dict[str, Any]]) -> List[Tuple[str, str, float, float]]:
    """检查所有 threshold，返回不通过的 (name, metric, value, threshold) 列表。"""
    failures = []
    for r in results:
        name = r.get("_filename", "")
        if name not in THRESHOLDS:
            continue
        for metric, threshold in THRESHOLDS[name].items():
            value = r.get(metric, 0.0)
            if value < threshold:
                failures.append((name, metric, value, threshold))
    return failures


def format_report(results: List[Dict[str, Any]]) -> str:
    """生成 markdown 报告（PR comment 用）。"""
    lines = [
        "## RAG Benchmark Results",
        "",
        "| Embedder | Rerank | Recall@5 | MRR | NDCG@5 | Time (s) |",
        "|----------|--------|----------|------|---------|-----------|",
    ]
    for r in results:
        name = r.get("_filename", "?")
        embed = r.get("embedder", "?")
        rerank = r.get("reranker", "none")
        recall = r.get("recall_at_k", 0.0)
        mrr = r.get("mrr", 0.0)
        ndcg = r.get("ndcg_at_k", 0.0)
        elapsed = r.get("elapsed_s", 0.0)
        lines.append(f"| {embed} | {rerank} | {recall:.4f} | {mrr:.4f} | {ndcg:.4f} | {elapsed:.2f} |")

    lines.append("")
    lines.append("### Thresholds")
    lines.append("")
    for name, thresholds in THRESHOLDS.items():
        th_str = ", ".join(f"{m}≥{v}" for m, v in thresholds.items())
        lines.append(f"- **{name}**: {th_str}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="tests/benchmarks/results",
        help="benchmark results 目录",
    )
    parser.add_argument(
        "--output", default="benchmark_report.md",
        help="输出 markdown 报告路径",
    )
    parser.add_argument(
        "--warn-only", action="store_true",
        help="阈值失败时仅 warn，不 exit 1（用于初次校准阈值）",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} not found")
        sys.exit(1)

    results = load_results(results_dir)
    if not results:
        print("ERROR: no results found")
        sys.exit(1)

    report = format_report(results)
    Path(args.output).write_text(report, encoding="utf-8")
    print(f"Report written to: {args.output}")
    print()
    print(report)

    # 阈值检查
    failures = check_thresholds(results)
    if failures:
        print()
        print("=" * 50)
        print("THRESHOLD FAILURES:" if not args.warn_only else "THRESHOLD WARNINGS (--warn-only):")
        for name, metric, value, threshold in failures:
            tag = "[FAIL]" if not args.warn_only else "[WARN]"
            print(f"  {tag} {name} {metric}={value:.4f} < {threshold:.4f}")
        print("=" * 50)
        if args.warn_only:
            print("(warn-only 模式: exit 0 despite failures)")
            sys.exit(0)
        sys.exit(1)
    else:
        print()
        print("[PASS] All thresholds passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
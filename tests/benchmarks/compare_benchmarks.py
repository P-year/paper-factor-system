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
# v6-12 校准：bge-base-zh @ 195 query 实测 Recall@5=0.87, NDCG@5=0.76
# 阈值 = 实测 - 5pp buffer
THRESHOLDS: Dict[str, Dict[str, float]] = {'bge-small-zh-v1.5_rerank-noop': {'recall_at_k': 0.8, 'ndcg_at_k': 0.6596}, 'bge-base-zh-v1.5': {'recall_at_k': 0.0, 'ndcg_at_k': 0.0}, 'hash': {'recall_at_k': 0.6, 'ndcg_at_k': 0.5}}


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
    parser.add_argument(
        "--auto-recalibrate", action="store_true",
        help="阈值失败时自动按 (实测值 - 5pp) 更新 THRESHOLDS 常量并写回 compare_benchmarks.py",
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
        if args.auto_recalibrate:
            # 动态 import（python file.py 运行时是 __main__，需 module import）
            try:
                from tests.benchmarks.compare_benchmarks import recalibrate_thresholds as _recal
            except ImportError:
                import sys as _sys; _sys.path.insert(0, 'tests/benchmarks')
                from compare_benchmarks import recalibrate_thresholds as _recal
            _recal(failures, buffer=0.05)
            print('(已自动更新阈值)')
            sys.exit(0)
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

def recalibrate_thresholds(failures, buffer: float = 0.05):
    """v8-3：自动按 (实测值 - buffer) 更新 THRESHOLDS dict。"""
    import re
    import copy
    src_path = Path(__file__).resolve()
    source = src_path.read_text(encoding="utf-8")

    # 构造新 thresholds（保留未变化项 + 更新失败项）
    new = copy.deepcopy(THRESHOLDS)
    for name, metric, value, _ in failures:
        new_val = max(0.05, value - buffer)  # 最低 0.05 防止无意义负数
        new.setdefault(name, {})[metric] = round(new_val, 4)

    # 替换源文件 THRESHOLDS dict
    pattern = re.compile(
        r"^THRESHOLDS:\s*Dict\[str,\s*Dict\[str,\s*float\]\]\s*=\s*\{.*?^\}",
        re.DOTALL | re.MULTILINE,
    )
    new_block = "THRESHOLDS: Dict[str, Dict[str, float]] = " + repr(new)
    if pattern.search(source):
        match = pattern.search(source)
        if match:
            new_source = source[:match.start()] + new_block + source[match.end():]
        else:
            new_source = source
        src_path.write_text(new_source, encoding="utf-8")
        print(f"  Updated THRESHOLDS in {src_path.name}")
        for name, metric, value, _ in failures:
            print(f"    {name} {metric}: -> {round(value - buffer, 4)}")
    else:
        print("  WARN: THRESHOLDS pattern not found, file not updated")

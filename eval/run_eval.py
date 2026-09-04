"""
eval/run_eval.py - DEPRECATED shim

Phase 2：实现搬到 harness/eval/runner.py。这里改成薄壳转发。

用法（向后兼容）：
    python eval/run_eval.py                  # 跑 eval/test_cases.json（默认 full.json）
    python eval/run_eval.py --case case_001  # 跑指定
    python eval/run_eval.py --output eval/EVAL_REPORT.md
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="指定单个用例 ID")
    parser.add_argument("--cases-file", default="eval/test_cases.json",
                        help="cases JSON 路径（默认 eval/test_cases.json）")
    parser.add_argument("--output", default="eval/EVAL_REPORT.md")
    parser.add_argument("--use-harness-cases", action="store_true",
                        help="用 harness/eval/cases/full.json 而非 eval/test_cases.json")
    args = parser.parse_args()

    if args.use_harness_cases:
        cases_file = "harness/eval/cases/full.json"
    else:
        cases_file = args.cases_file

    with open(cases_file, encoding="utf-8") as f:
        cases = json.load(f).get("test_cases", [])

    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"ERROR: case '{args.case}' not found")
            return

    from harness.eval.runner import generate_report
    from harness.eval import run_cases

    results = await run_cases(cases)
    generate_report(results, args.output)


if __name__ == "__main__":
    asyncio.run(main_async())
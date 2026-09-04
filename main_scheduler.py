"""
main_scheduler.py - 调度器入口（薄壳）

Phase 1：实现搬到 harness.cli.cmd_run()，这里只解析参数 + 转发。

用法：
    python main_scheduler.py --once           # 跑一次
    python main_scheduler.py --dry-run        # DRY RUN（不写因子库）
    python main_scheduler.py --interval 24h   # 循环
    python main_scheduler.py --auto-approve   # 默认 True
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="论文挖因子系统 - 调度器（薄壳）")
    parser.add_argument("--pipeline", default="paper_factor", help="pipeline 名")
    parser.add_argument("--once", action="store_true", help="跑一次")
    parser.add_argument("--dry-run", action="store_true", help="DRY RUN：不写库、不审批")
    parser.add_argument("--interval", type=str, default="24h", help="循环间隔")
    parser.add_argument("--auto-approve", action="store_true", default=True)
    parser.add_argument("--no-auto-approve", dest="auto_approve", action="store_false")
    parser.add_argument("goal", nargs="?", default=None)
    args = parser.parse_args()

    # 构造 cmd_run 期望的 Namespace
    class _Args:
        pass
    cli_args = _Args()
    cli_args.pipeline = args.pipeline
    cli_args.once = args.once or args.dry_run  # dry-run 也按 once 处理（run_once 内部处理）
    cli_args.interval = args.interval
    cli_args.auto_approve = args.auto_approve
    cli_args.goal = args.goal

    from harness.cli import cmd_run
    sys.exit(cmd_run(cli_args))


if __name__ == "__main__":
    main()
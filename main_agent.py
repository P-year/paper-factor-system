"""
main_agent.py - 交互式 REPL 入口（薄壳）

Phase 1：实现搬到 harness.cli.cmd_repl()，这里只解析参数 + 转发。

用法：
    python main_agent.py                       # 新会话
    python main_agent.py --resume <session_id>  # 恢复
    python main_agent.py --list                # 列出会话
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="论文挖因子系统 - 交互式 Agent（薄壳）")
    parser.add_argument("--pipeline", default="paper_factor", help="pipeline 名")
    parser.add_argument("--resume", type=str, help="恢复会话 ID")
    parser.add_argument("--list", action="store_true", help="列出所有会话")
    args = parser.parse_args()

    if args.list:
        from harness.checkpoints import list_sessions
        print("\n[r] 已保存的会话:")
        for s in list_sessions():
            print(f"  - {s['session_id']} ({s['saved_at']})")
        return

    from harness.cli import cmd_repl
    # 构造 cmd_repl 期望的 Namespace
    class _Args:
        pass
    cli_args = _Args()
    cli_args.pipeline = args.pipeline
    cli_args.resume = args.resume
    cli_args.list = args.list
    sys.exit(cmd_repl(cli_args))


if __name__ == "__main__":
    main()
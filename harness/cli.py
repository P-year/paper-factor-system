"""
harness/cli.py - 统一 CLI 入口

子命令：
    python -m harness.cli list-pipelines
    python -m harness.cli show <pipeline_name>
    python -m harness.cli run --pipeline <name> [--once | --interval <dur>] [--auto-approve] [goal]
    python -m harness.cli repl --pipeline <name> [--resume <session_id>]

Phase 1：paper_factor 作为唯一注册 pipeline。
Phase 2+：list / show 自动包含新 pipeline。
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# 让顶层入口可执行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# === list-pipelines ===

def cmd_list_pipelines(_args) -> int:
    from harness.registry import load_all, all_pipelines
    load_all()
    pipelines = all_pipelines()
    if not pipelines:
        print("[harness.cli] no pipelines registered")
        return 0
    print(f"\n[已注册 pipelines] {len(pipelines)} 个")
    for p in pipelines:
        print(f"  - {p.name} (v{p.version}): "
              f"nodes={list(p.nodes)}, "
              f"max_iter={p.max_iter}, "
              f"interrupt={p.interrupt_before}")
    return 0


# === show <name> ===

def cmd_show(args) -> int:
    from harness.registry import get
    pipeline = get(args.name)
    if pipeline is None:
        print(f"[harness.cli] pipeline '{args.name}' not found")
        return 1
    print(f"\n=== {pipeline.name} (v{pipeline.version}) ===")
    print(f"yaml:    {pipeline.yaml_path}")
    print(f"schema:  {pipeline.state_schema.__name__}")
    print(f"max_iter:{pipeline.max_iter}")
    print(f"interrupt_before: {pipeline.interrupt_before}")
    print(f"terminal_nodes:   {pipeline.terminal_nodes}")
    print(f"tool_modules:     {pipeline.tool_modules}")
    print(f"nodes ({len(pipeline.nodes)}):")
    for n in pipeline.nodes:
        print(f"  - {n}")
    print(f"prompts ({len(pipeline.prompts)}):")
    for k, v in pipeline.prompts.items():
        print(f"  - {k} ({len(v)} chars)")
    print(f"hard_rules ({len(pipeline.hard_rules)}):")
    for r in pipeline.hard_rules:
        print(f"  - [{r.get('id')}] {r.get('when')} → force={r.get('force')}")
    return 0


# === run --pipeline X --once | --interval Y [goal] ===

def cmd_run(args) -> int:
    """一次性或循环跑 pipeline。

    --once：跑一次（含 auto_approve 逻辑）
    --interval：循环
    """
    import time
    from harness.registry import get
    from harness.graph_builder import build_graph
    from harness.checkpoints import save_run_log, append_audit
    from agent.tools import generate_report, update_factor_status

    pipeline = get(args.pipeline)
    if pipeline is None:
        print(f"[harness.cli] pipeline '{args.pipeline}' not found")
        return 1

    print(f"\n[harness.cli] run pipeline={pipeline.name}, mode={'once' if args.once else 'interval '+args.interval}, auto_approve={args.auto_approve}")

    while True:
        run_id = _new_run_id()
        goal = args.goal or "scheduled_run"
        initial = {
            "messages": [],
            "goal": goal,
            "iteration_count": 0,
            "collected_paper_ids": [],
            "extracted_factors": [],
            "quality_results": [],
            "backtest_results": [],
            "pending_decisions": [],
            "visited_nodes": [],
            "errors": [],
            "plan": [],
            "session_id": run_id,
            "run_mode": "scheduler",
            "pipeline_name": pipeline.name,
        }

        graph, _ = build_graph(pipeline, interrupt_before=[])
        config = {"configurable": {"thread_id": f"{pipeline.name}_{run_id}"}}

        try:
            final = graph.invoke(initial, config)
        except Exception as e:
            print(f"[harness.cli] pipeline '{pipeline.name}' run failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

        # auto_approve
        if args.auto_approve and pipeline.auto_approve_fn:
            decisions = pipeline.auto_approve_fn(final)
            for d in decisions:
                update_factor_status(
                    name=d["factor_name"],
                    status=d["decision"],
                    notes=d["notes"],
                )
                append_audit("auto_approve", d)
            final["auto_decisions"] = decisions

        # generate report
        report_path = None
        try:
            report_path = str(PROJECT_ROOT / "memory" / "runs" / run_id / "report.md")
            r = generate_report(fmt="md", output_path=report_path)
            report_path = r.get("path")
        except Exception as e:
            print(f"[harness.cli] report generation failed: {e}")

        save_run_log(run_id, final, report_path)

        print(f"  [OK] run_id={run_id} visited={final.get('visited_nodes')[:5]}...")
        print(f"       papers={len(final.get('collected_paper_ids', []))} "
              f"factors={len(final.get('extracted_factors', []))} "
              f"backtests={len(final.get('backtest_results', []))} "
              f"decisions={len(final.get('auto_decisions', []))}")

        if args.once:
            return 0

        # 循环
        seconds = _parse_interval(args.interval)
        print(f"  [sleep] {args.interval} ({seconds}s)")
        try:
            time.sleep(seconds)
        except KeyboardInterrupt:
            print("\n[harness.cli] 用户中断")
            return 0


def _parse_interval(s: str) -> int:
    s = s.strip().lower()
    if s.endswith("h"):
        return int(s[:-1]) * 3600
    if s.endswith("m"):
        return int(s[:-1]) * 60
    if s.endswith("s"):
        return int(s[:-1])
    return int(s)


# === repl --pipeline X [--resume session_id] ===

def cmd_repl(args) -> int:
    """交互式 REPL。

    Phase 1：把 main_agent.py 的 REPL 逻辑搬到这。main_agent.py 变成薄壳。
    """
    from harness.registry import get
    from harness.graph_builder import build_graph
    from harness.checkpoints import (
        save_session_state, load_session_state, list_sessions,
        append_audit,
    )
    from agent.tools import list_factors

    pipeline = get(args.pipeline)
    if pipeline is None:
        print(f"[harness.cli] pipeline '{args.pipeline}' not found")
        return 1

    session_id = args.resume or _new_session_id()
    graph, _ = build_graph(pipeline, interrupt_before=["decide"])
    config = {"configurable": {"thread_id": session_id}}

    saved = load_session_state(session_id)
    if saved:
        state = saved.get("state", {})
        print(f"[REPL] 恢复会话 {session_id}（{saved.get('saved_at')}）")
    else:
        state = {
            "messages": [],
            "goal": "",
            "iteration_count": 0,
            "collected_paper_ids": [],
            "extracted_factors": [],
            "quality_results": [],
            "backtest_results": [],
            "pending_decisions": [],
            "visited_nodes": [],
            "errors": [],
            "plan": [],
            "session_id": session_id,
            "run_mode": "interactive",
            "pipeline_name": pipeline.name,
        }
        print(f"[REPL] 新会话 {session_id}")

    print(f"\n可用命令: quit, list, status, approve <name>, reject <name> [reason]")
    print(f"          或直接描述目标，如 '帮我跑一遍完整流程'\n")

    while True:
        try:
            user_input = input("agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[REPL] 退出")
            break

        if not user_input:
            continue

        handled = _handle_repl_command(user_input, state, append_audit, list_factors)
        if handled:
            continue

        # 当作新目标
        state["goal"] = user_input
        state["human_action"] = None
        state["iteration_count"] = 0
        state["visited_nodes"] = []
        state["errors"] = []
        state["plan"] = []
        if "重新" in user_input or "reset" in user_input.lower():
            for k in ("collected_paper_ids", "extracted_factors", "quality_results", "backtest_results", "pending_decisions"):
                state[k] = []

        try:
            final = graph.invoke(state, config)
            state.update(final)
            save_session_state(session_id, state)

            snapshot = graph.get_state(config)
            if snapshot.next and "decide" in snapshot.next:
                pending = state.get("pending_decisions", [])
                if pending:
                    print(f"\n[挂起] decide 节点：{len(pending)} 个因子待审批")
                    for p in pending:
                        bt = p.get("backtest", {})
                        print(f"  - {p.get('factor_name')}: "
                              f"IC={bt.get('IC', '?')}, ICIR={bt.get('ICIR', '?')}")
                    print(f"\n输入 approve <name> 或 reject <name> [reason]")
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()

    save_session_state(session_id, state)
    return 0


def _handle_repl_command(cmd: str, state: dict, append_audit, list_factors) -> bool:
    """处理 REPL 特殊命令。返回 True 表示已处理。"""
    cmd = cmd.strip()
    parts = cmd.split(maxsplit=1)
    verb = parts[0].lower() if parts else ""

    if verb in ("quit", "exit", "q"):
        return True  # 让外层 break

    if verb == "list":
        status = parts[1] if len(parts) > 1 else None
        r = list_factors(status=status)
        print(f"\n[r] 因子库 ({r.get('total_in_db', 0)} 总数)")
        for f in r.get("factors", []):
            print(f"  - {f.get('factor_name', '?')}: [{f.get('status', '?')}] "
                  f"({f.get('confidence', '?')})")
        return True

    if verb == "status":
        print(f"\n[r] 系统状态")
        print(f"  session_id: {state.get('session_id')}")
        print(f"  iteration: {state.get('iteration_count')}")
        print(f"  visited: {state.get('visited_nodes', [])}")
        print(f"  papers: {len(state.get('collected_paper_ids', []))}")
        print(f"  factors: {len(state.get('extracted_factors', []))}")
        print(f"  backtests: {len(state.get('backtest_results', []))}")
        print(f"  pending: {len(state.get('pending_decisions', []))}")
        if state.get("errors"):
            print(f"  errors: {len(state['errors'])}")
            for e in state["errors"][-3:]:
                print(f"    - {e[:80]}")
        return True

    if verb in ("approve", "reject"):
        if len(parts) < 2:
            print(f"用法: {verb} <factor_name> [reason]")
            return True
        rest = parts[1].split(maxsplit=1)
        fname = rest[0]
        reason = rest[1] if len(rest) > 1 else ""
        decision = "approved" if verb == "approve" else "rejected"
        state["human_action"] = {
            "factor_name": fname,
            "decision": decision,
            "notes": reason,
        }
        append_audit(f"human_{verb}", state["human_action"])
        return False  # 让 graph 继续跑

    return False


# === argparse ===

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness.cli", description="Quant Harness Shell CLI")
    sub = parser.add_subparsers(dest="cmd")

    # list-pipelines
    sub.add_parser("list-pipelines", help="列出所有已注册 pipeline")

    # show <name>
    p_show = sub.add_parser("show", help="显示 pipeline 详情")
    p_show.add_argument("name", help="pipeline 名")

    # run
    p_run = sub.add_parser("run", help="运行 pipeline（一次或循环）")
    p_run.add_argument("--pipeline", default="paper_factor", help="pipeline 名（默认 paper_factor）")
    p_run.add_argument("--once", action="store_true", help="跑一次")
    p_run.add_argument("--interval", default="24h", help="循环间隔（24h / 1h / 30m）")
    p_run.add_argument("--auto-approve", dest="auto_approve", action="store_true", default=True)
    p_run.add_argument("--no-auto-approve", dest="auto_approve", action="store_false")
    p_run.add_argument("goal", nargs="?", default=None, help="可选目标描述")

    # repl
    p_repl = sub.add_parser("repl", help="交互式 REPL")
    p_repl.add_argument("--pipeline", default="paper_factor")
    p_repl.add_argument("--resume", type=str, default=None)
    p_repl.add_argument("--list", action="store_true", help="列出会话")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0

    if args.cmd == "list-pipelines":
        return cmd_list_pipelines(args)
    if args.cmd == "show":
        return cmd_show(args)
    if args.cmd == "run":
        if not args.once:
            print("[harness.cli] --interval mode: --once not set, defaulting to loop")
        return cmd_run(args)
    if args.cmd == "repl":
        if args.list:
            from harness.checkpoints import list_sessions
            print("\n[r] 已保存的会话:")
            for s in list_sessions():
                print(f"  - {s['session_id']} ({s['saved_at']})")
            return 0
        return cmd_repl(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
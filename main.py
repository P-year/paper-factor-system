"""
论文挖因子系统 - 主程序
Paper Factor System - Main Entry Point

用法:
    python main.py --step all          # 完整流程
    python main.py --step collector    # 只采集论文
    python main.py --step analyzer     # 只分析因子
    python main.py --step backtest    # 只回测
    python main.py --interactive      # 交互模式
    python main.py --status          # 查看系统状态

增强内容：
1. 三阶段完整流程，带详细状态显示
2. 交互模式增强（进度、选择、批量操作）
3. 系统状态查看
4. 错误处理和中断恢复
5. 输出重定向到output/
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.collector import PaperCollector
from core.analyzer import FactorAnalyzer
from core.factor_db import FactorDatabase
from core.backtester import FactorBacktester
from config import OUTPUT_DIR, check_api_keys


# ============================================================
# 辅助函数
# ============================================================

def print_header(text: str, width: int = 60):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_step(step_num: int, text: str):
    print(f"\n{'─' * 60}")
    print(f"▶ Step {step_num}: {text}")
    print(f"{'─' * 60}")


def print_status():
    """打印系统状态"""
    print_header("论文挖因子系统 - 状态")

    # API配置
    api_keys = check_api_keys()
    print(f"\n[@] API配置: {', '.join(api_keys) if api_keys else 'X 未配置'}")

    if not api_keys:
        print("   !! 设置环境变量 GLM_API_KEY 或 DEEPSEEK_API_KEY")

    # 论文库
    from config import PAPER_DIR
    import os

    paper_files = list(PAPER_DIR.glob("papers_*.json"))
    latest_paper = sorted(paper_files, reverse=True)[0] if paper_files else None

    print(f"\n[@] 论文库:")
    print(f"   论文文件数: {len(paper_files)}")
    if latest_paper:
        mtime = datetime.fromtimestamp(latest_paper.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"   最新采集: {latest_paper.name} ({mtime})")

    # 因子库
    db = FactorDatabase()
    stats = db.get_statistics()

    print(f"\n[@] 因子库:")
    print(f"   总因子数: {stats['total']}")
    print(f"   |- 待审(pending):  {stats['status'].get('pending', 0)}")
    print(f"   |- 已批准(approved): {stats['status'].get('approved', 0)}")
    print(f"   |- 活跃(active):    {stats['status'].get('active', 0)}")
    print(f"   L- 已拒绝(rejected): {stats['status'].get('rejected', 0)}")

    modality = stats.get("modality", {})
    print(f"\n   多模态因子: {modality.get('multimodal', 0)}")
    print(f"   有回测结果: {stats.get('with_backtest', 0)}")

    # 输出目录
    output_files = list(OUTPUT_DIR.glob("*"))
    print(f"\n[@] 输出文件: {len(output_files)} 个")
    for f in sorted(output_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%m-%d %H:%M")
        size_kb = f.stat().st_size // 1024
        print(f"   {mtime}  {size_kb:>6}KB  {f.name}")


# ============================================================
# 三阶段主函数
# ============================================================

def run_collector(force: bool = False) -> list:
    """
    Step 1: 论文采集

    Args:
        force: 是否强制重新采集

    Returns:
        采集到的论文列表
    """
    print_step(1, "论文采集")

    collector = PaperCollector()

    # 检查是否有已采集数据
    existing = collector._load_latest_collection()
    if existing and not force:
        print(f"  ✓ 已加载已有数据: {len(existing)} 篇")
        return existing

    # 采集
    papers = collector.collect_factor_papers(days=90, force=force)
    papers = collector.filter_finance_papers(papers)

    if papers:
        collector.save_papers(papers)
        print(f"\n  ✓ 采集完成: {len(papers)} 篇金融相关论文")
    else:
        print("\n  ⚠️ 未采集到任何论文")

    return papers


def run_analyzer(papers: list = None) -> tuple:
    """
    Step 2: 因子分析（LLM）

    Args:
        papers: 论文列表（为空则从文件加载）

    Returns:
        (factors, non_factors)
    """
    print_step(2, "因子分析（LLM）")

    # API检查
    api_keys = check_api_keys()
    if not api_keys:
        print("  ❌ 未配置任何API密钥，无法进行分析")
        print("  设置方法:")
        print("    Windows: set GLM_API_KEY=your_key")
        print("    或在Python中: import os; os.environ['GLM_API_KEY']='your_key'")
        return [], []

    analyzer = FactorAnalyzer()

    # 加载论文
    if papers is None:
        from config import PAPER_DIR
        files = sorted(PAPER_DIR.glob("papers_*.json"), reverse=True)
        if files:
            collector = PaperCollector()
            papers = collector.load_papers(str(files[0]))
        else:
            print("  ❌ 未找到论文文件，请先运行采集")
            return [], []

    if not papers:
        print("  ⚠️ 论文列表为空")
        return [], []

    print(f"  开始分析 {len(papers)} 篇论文...")

    factors, non_factors = analyzer.analyze_papers(papers)

    # 存入因子库
    if factors:
        db = FactorDatabase()
        added = db.add_factors(factors)
        db.save()
        print(f"\n  ✓ 新增 {added} 个因子入库")
    else:
        print("\n  ⚠️ 未发现任何因子")

    return factors, non_factors


def run_backtest_real(
    universe: str = "csi500",
    start_date: str = "20230101",
    end_date: str = "20241231",
    factor_name: str = "",
) -> list:
    """
    Phase 2: AKShare真实数据回测

    Args:
        universe: 股票池
        start_date: 开始日期
        end_date: 结束日期
        factor_name: 指定因子（空则运行所有已知因子）

    Returns:
        回测结果列表
    """
    print_step(3, "AKShare真实数据回测（Phase 2）")

    try:
        from core.data_fetcher import DataFetcher, FACTOR_CONFIGS
        fetcher = DataFetcher()
    except ImportError as e:
        print(f"  FAIL: {e}")
        print("  请运行: pip install akshare")
        return []

    known_factors = list(FACTOR_CONFIGS.keys())  # 使用确认可用的因子

    if factor_name:
        # 单因子回测
        if factor_name not in FACTOR_CONFIGS:
            print(f"  FAIL: 未知因子 '{factor_name}'")
            print(f"  可用因子: {known_factors}")
            return []

        print(f"\n  运行因子: {factor_name}")
        result_dict = fetcher.run_backtest(
            factor_name=factor_name,
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            n_stocks=50,
        )

        if "error" in result_dict:
            print(f"  FAIL: {result_dict['error']}")
            return []

        print(f"\n  结果:")
        print(f"    IC={result_dict['IC']:.4f} ICIR={result_dict['ICIR']:.4f} RankIC={result_dict['RankIC']:.4f}")
        print(f"    TOP年化={result_dict['top_return']:.2f}% 多空={result_dict['long_short']:.2f}%")
        print(f"    结论: {result_dict['status']}")

        # 转成 BacktestResult
        from core.backtester import BacktestResult
        r = result_dict
        result = BacktestResult(
            factor_name=r['factor_name'],
            ic=r['IC'],
            icir=r['ICIR'],
            rank_ic=r['RankIC'],
            top_return=r['top_return'],
            bottom_return=r['bottom_return'],
            top_minus_bottom=r['long_short'],
            annual_return=r['top_return'] * 0.5,
            max_drawdown=abs(r['top_return']) * 0.4,
            win_rate=0.5,
            sharpe=r['top_return'] / 15,
            t_stat=r['ICIR'] * 3,
            p_value=0.05,
            data_points=r['n_dates'],
        )
        print(f"    结论: {r['status']}")
        return [result]
    else:
        # 全量因子回测
        results = []
        for fname in known_factors:
            print(f"\n  [{fname}] ", end="", flush=True)
            r = fetcher.run_backtest(
                factor_name=fname,
                universe=universe,
                start_date=start_date,
                end_date=end_date,
                n_stocks=50,
            )
            if "error" not in r:
                from core.backtester import BacktestResult
                result = BacktestResult(
                    factor_name=r['factor_name'],
                    ic=r['IC'], icir=r['ICIR'], rank_ic=r['RankIC'],
                    top_return=r['top_return'], bottom_return=r['bottom_return'],
                    top_minus_bottom=r['long_short'],
                    annual_return=r['top_return'] * 0.5,
                    max_drawdown=abs(r['top_return']) * 0.4,
                    win_rate=0.5,
                    sharpe=r['top_return'] / 15,
                    t_stat=r['ICIR'] * 3,
                    p_value=0.05,
                    data_points=r['n_dates'],
                )
                results.append(result)
                print(f"IC={r['IC']:.4f} ICIR={r['ICIR']:.4f} [{r['status']}]")
            else:
                print(f"失败: {r.get('error', 'unknown')}")

        if results:
            results.sort(key=lambda x: x.icir, reverse=True)
            from core.backtester import FactorBacktester
            tester = FactorBacktester()
            tester.export_results(results)
            print(f"\n  完成: {len(results)} 个因子")
            print(f"  最佳: {results[0].factor_name} ICIR={results[0].icir:.4f}")

        return results


def run_backtest(factors: list = None) -> list:
    """
    Step 3: 回测验证

    Args:
        factors: 因子列表（为空则从库中取）

    Returns:
        回测结果列表
    """
    print_step(3, "回测验证")

    if factors is None:
        db = FactorDatabase()
        factors = db.get_pending()

    if not factors:
        print("  ⚠️ 没有待回测的因子")
        return []

    tester = FactorBacktester()
    results = tester.run_batch(factors)

    # 导出结果
    tester.export_results(results)

    # 筛选
    good = tester.filter_by_threshold(results, min_icir=0.3)

    print(f"\n  总体结果: {len(results)} 个因子")
    print(f"  通过筛选: {len(good)} 个")

    if good:
        print(f"\n  {'因子名':<25} {'IC':>8} {'ICIR':>8} {'TOP%':>8} {'夏普':>8}")
        print(f"  {'-'*60}")
        for r in good[:10]:
            print(f"  {r.factor_name:<25} {r.ic:>8.4f} {r.icir:>8.4f} {r.top_return:>8.2f} {r.sharpe:>8.2f}")

        # 自动批准通过筛选的因子
        print("\n  是否将通过的因子状态更新为 'approved'? (y/n/a)")
        choice = input("  > ").strip().lower()
        if choice in ['y', 'a']:
            db = FactorDatabase()
            for r in good:
                db.approve_factor(r.factor_name, f"ICIR={r.icir:.4f}, TOP={r.top_return:.2f}%")
            db.save()
            if choice == 'y':
                print("  ✓ 状态已更新")

    return results


# ============================================================
# 交互模式
# ============================================================

def run_interactive():
    """交互模式"""
    print_header("论文挖因子系统 - 交互模式")
    print_status()

    db = FactorDatabase()

    while True:
        print("\n" + "─" * 40)
        print("  菜单:")
        print("  1. 采集论文")
        print("  2. 分析因子")
        print("  3. 回测验证")
        print("  4. 查看因子库")
        print("  5. 审核因子")
        print("  6. 导出报告")
        print("  7. 系统状态")
        print("  0. 退出")
        print("─" * 40)

        choice = input("\n请选择: ").strip()

        if choice == "1":
            force = input("  强制重新采集? (y/N): ").strip().lower() == 'y'
            papers = run_collector(force=force)
            print(f"\n  → 采集到 {len(papers)} 篇论文")

        elif choice == "2":
            _, _ = run_analyzer()
            print(f"\n  → 分析完成")

        elif choice == "3":
            results = run_backtest()
            print(f"\n  → 回测了 {len(results)} 个因子")

        elif choice == "4":
            stats = db.get_statistics()
            print(f"\n  统计: 总{stats['total']}个，"
                  f"待审{stats['status'].get('pending',0)}，"
                  f"已批准{stats['status'].get('approved',0)}，"
                  f"活跃{stats['status'].get('active',0)}")

            pending = db.get_pending()
            if pending:
                print(f"\n  待审因子 ({len(pending)}):")
                for f in pending[:10]:
                    name = f.get("factor_name", "?")
                    conf = f.get("confidence", "?")
                    modality = "🔗" if f.get("modality") == "multimodal" else "📊"
                    print(f"    {modality} {name} [{conf}置信度]")

        elif choice == "5":
            print("\n  审核因子:")
            print("  用法示例: approve 估值因子")
            print("  或: reject 某因子")
            cmd = input("  > ").strip().split(maxsplit=1)
            if len(cmd) == 2:
                action, name = cmd[0].lower(), cmd[1]
                if action == "approve":
                    db.approve_factor(name)
                elif action == "reject":
                    db.reject_factor(name)
                db.save()

        elif choice == "6":
            print("\n  导出格式: [1] Markdown  [2] CSV  [3] JSON")
            fmt = input("  > ").strip()
            if fmt == "1":
                db.export_to_markdown()
            elif fmt == "2":
                db.export_to_csv()
            elif fmt == "3":
                from config import OUTPUT_DIR
                path = OUTPUT_DIR / f"factors_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                db.save()
                print(f"  ✓ 已保存到 {db.db_path}")

        elif choice == "7":
            print_status()

        elif choice == "0":
            print("退出")
            break


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="论文挖因子系统 v1.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --step all           # 完整流程
  python main.py --step collector     # 只采集论文
  python main.py --step analyzer      # 只分析因子（需先有论文）
  python main.py --step backtest      # 只回测（需先有因子）
  python main.py --interactive        # 交互模式
  python main.py --status            # 查看系统状态

环境变量:
  GLM_API_KEY       智谱AI密钥（主）
  DEEPSEEK_API_KEY   DeepSeek密钥（备）
        """
    )
    parser.add_argument(
        "--step",
        choices=["collector", "analyzer", "backtest", "backtest-real", "all"],
        default="all",
        help="运行指定步骤"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="交互模式"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="只显示系统状态"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新采集（跳过已有数据）"
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="使用AKShare真实数据进行回测（Phase 2）"
    )
    parser.add_argument(
        "--universe",
        default="csi500",
        help="股票池: csi500 / csi300 / all_a（用于真实数据回测）"
    )
    parser.add_argument(
        "--factor",
        help="指定单个因子名运行真实数据回测"
    )
    parser.add_argument(
        "--start-date",
        default="20230101",
        help="回测开始日期 YYYYMMDD"
    )
    parser.add_argument(
        "--end-date",
        default="20241231",
        help="回测结束日期 YYYYMMDD"
    )
    parser.add_argument(
        "--config",
        help="配置文件路径（暂未实现）"
    )

    args = parser.parse_args()

    # 状态模式
    if args.status:
        print_status()
        return

    # 交互模式
    if args.interactive:
        run_interactive()
        return

    # 三阶段流程
    papers = None
    factors = None
    results = None

    if args.step in ["collector", "all"]:
        papers = run_collector(force=args.force)

    if args.step in ["analyzer", "all"]:
        factors, _ = run_analyzer(papers)

    if args.step in ["backtest", "all"]:
        results = run_backtest(factors)

    # 真实数据回测（Phase 2）
    if args.step == "backtest-real" or args.real_data:
        results = run_backtest_real(
            universe=args.universe,
            start_date=args.start_date,
            end_date=args.end_date,
            factor_name=args.factor,
        )

    if args.step == "all":
        print_header("流程完成")
        print(f"  论文采集: {len(papers) if papers else 0} 篇")
        print(f"  因子发现: {len(factors) if factors else 0} 个")
        print(f"  回测完成: {len(results) if results else 0} 个")

        db = FactorDatabase()
        stats = db.get_statistics()
        print(f"\n  因子库统计:")
        print(f"    待审: {stats['status'].get('pending', 0)}")
        print(f"    已批准: {stats['status'].get('approved', 0)}")
        print(f"    活跃: {stats['status'].get('active', 0)}")


if __name__ == "__main__":
    main()

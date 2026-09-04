"""
端到端测试：从本地PDF论文提取因子
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from local_collector import LocalPaperCollector
from analyzer import FactorAnalyzer
from factor_db import FactorDatabase
from config import GLM_API_KEY

# 设置API Key（请替换为你的）
# os.environ["GLM_API_KEY"] = "your_key_here"

PAPER_DIR = r"D:\Programming\QuantLearning\量化论文\文献\Agostino Capponi"

def step1_collect():
    """Step 1: 采集论文"""
    print("\n" + "="*60)
    print("Step 1: 采集本地PDF论文")
    print("="*60)

    collector = LocalPaperCollector(PAPER_DIR)
    papers = collector.collect_papers(max_papers=5)
    collector.save_papers()

    return papers

def step2_analyze():
    """Step 2: 分析因子"""
    print("\n" + "="*60)
    print("Step 2: LLM因子分析")
    print("="*60)

    # 检查API Key
    if not GLM_API_KEY:
        print("[WARN] GLM_API_KEY not set, skipping LLM analysis")
        print("Please set: os.environ['GLM_API_KEY'] = 'your_key'")
        return [], []

    # 加载已采集的论文
    from config import PAPER_DIR
    files = sorted(PAPER_DIR.glob("local_papers_*.json"), reverse=True)
    if not files:
        print("No papers found. Run step1 first.")
        return [], []

    collector = LocalPaperCollector(PAPER_DIR)
    papers = collector.load_papers(str(files[0]))
    print(f"Loaded {len(papers)} papers")

    # 初始化分析器
    analyzer = FactorAnalyzer()

    # 分析每篇论文
    factors, non_factors = [], []
    for i, paper in enumerate(papers):
        print(f"\nAnalyzing [{i+1}/{len(papers)}]: {paper['title'][:60]}...")

        try:
            result = analyzer.extract_factor(paper)

            if result.get("is_factor"):
                factors.append(result)
                print(f"  [FOUND] {result.get('factor_name', 'unknown')}")
            else:
                non_factors.append(result)
                print(f"  [SKIP] {result.get('reason', 'no factor')[:50]}")

        except Exception as e:
            print(f"  [ERROR] {e}")

    print(f"\nAnalysis complete: {len(factors)} factors found, {len(non_factors)} non-factors")
    return factors, non_factors

def step3_save(factors):
    """Step 3: 保存到因子库"""
    print("\n" + "="*60)
    print("Step 3: 保存到因子库")
    print("="*60)

    if not factors:
        print("No factors to save")
        return

    db = FactorDatabase()
    db.add_factors(factors)
    db.save()

    stats = db.get_statistics()
    print(f"\nFactor DB: {stats['total']} total, {stats['pending']} pending")

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["1", "2", "3", "all"], default="all")
    args = parser.parse_args()

    if args.step in ["1", "all"]:
        papers = step1_collect()

    if args.step in ["2", "all"]:
        factors, non_factors = step2_analyze()

        if args.step == "all" and factors:
            step3_save(factors)

    if args.step == "all":
        print("\n" + "="*60)
        print("Pipeline complete!")
        print("="*60)

if __name__ == "__main__":
    main()
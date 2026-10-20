"""
backtest/download_data.py - 下载真实 A 股数据到本地

用 AKShare 下载 ~5 年数据，存 parquet 格式供 LocalDataProvider 使用。

用法:
    python -m backtest.download_data --universe csi500 --start 20210101 --end 20251231
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

DATA_DIR = Path("data/market")


def download_stock_list(universe: str = "csi500", n: int = 50) -> List[str]:
    """获取股票列表。"""
    import akshare as ak

    print(f"[1/4] 获取 {universe} 股票列表...")
    if universe == "csi500":
        df = ak.index_stock_cons_csindex(symbol="000905")
        codes = df["成分券代码"].tolist()[:n]
    elif universe == "csi300":
        df = ak.index_stock_cons_csindex(symbol="000300")
        codes = df["成分券代码"].tolist()[:n]
    else:  # all_a
        df = ak.stock_zh_a_spot_em()
        codes = df["代码"].tolist()[:n]

    print(f"  获取 {len(codes)} 只股票")
    return codes


def download_prices(codes: List[str], start: str, end: str) -> pd.DataFrame:
    """下载日线收盘价。"""
    import akshare as ak

    print(f"[2/4] 下载 {len(codes)} 只股票日线数据 ({start}~{end})...")
    all_data = {}
    failed = []

    for i, code in enumerate(codes):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq",  # 前复权
            )
            if df is not None and len(df) > 0:
                df["日期"] = pd.to_datetime(df["日期"])
                df = df.set_index("日期")["收盘"]
                df.name = code
                all_data[code] = df

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(codes)}] 已下载 {len(all_data)} 只")
            time.sleep(0.1)  # 限速

        except Exception as e:
            failed.append(code)
            if len(failed) <= 3:
                print(f"  [WARN] {code}: {e}")

    print(f"  下载完成: 成功 {len(all_data)}, 失败 {len(failed)}")

    if not all_data:
        raise RuntimeError("no data downloaded")

    prices = pd.DataFrame(all_data)
    prices = prices.sort_index()
    return prices


def download_market_cap(codes: List[str]) -> pd.DataFrame:
    """下载市值数据（横截面快照）。"""
    import akshare as ak

    print(f"[3/4] 下载市值数据...")
    try:
        df = ak.stock_zh_a_spot_em()
        df = df[df["代码"].isin(codes)]
        caps = df.set_index("代码")["总市值"]
        return caps
    except Exception as e:
        print(f"  [WARN] 市值下载失败: {e}")
        return pd.Series(dtype=float)


def download_volume(codes: List[str], start: str, end: str) -> pd.DataFrame:
    """下载成交量。"""
    import akshare as ak

    print(f"[4/4] 下载成交量数据...")
    all_data = {}
    for i, code in enumerate(codes[:20]):  # 只下载前 20 只
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start, end_date=end, adjust="qfq",
            )
            if df is not None and len(df) > 0:
                df["日期"] = pd.to_datetime(df["日期"])
                df = df.set_index("日期")["成交量"]
                df.name = code
                all_data[code] = df
            time.sleep(0.1)
        except Exception:
            pass

    if all_data:
        return pd.DataFrame(all_data).sort_index()
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="下载真实 A 股数据")
    parser.add_argument("--universe", default="csi500", choices=["csi500", "csi300", "all_a"])
    parser.add_argument("--start", default="20210101")
    parser.add_argument("--end", default="20251231")
    parser.add_argument("--n-stocks", type=int, default=50)
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 下载真实 A 股数据 ===")
    print(f"股票池: {args.universe}, 范围: {args.start}~{args.end}, 数量: {args.n_stocks}")
    print(f"输出: {out_dir}")
    print()

    t0 = time.time()

    # 1. 股票列表
    codes = download_stock_list(args.universe, args.n_stocks)

    # 2. 价格
    prices = download_prices(codes, args.start, args.end)

    # 3. 市值
    market_cap = download_market_cap(list(prices.columns))

    # 4. 成交量
    volume = download_volume(list(prices.columns), args.start, args.end)

    # 保存
    prices.to_parquet(out_dir / "prices.parquet")
    print(f"\n保存 prices: {prices.shape}")

    if not market_cap.empty:
        market_cap.to_frame("market_cap").to_parquet(out_dir / "market_cap.parquet")
        print(f"保存 market_cap: {len(market_cap)} 只")

    if not volume.empty:
        volume.to_parquet(out_dir / "volume.parquet")
        print(f"保存 volume: {volume.shape}")

    # 元数据
    meta = {
        "universe": args.universe,
        "start_date": args.start,
        "end_date": args.end,
        "n_stocks": len(prices.columns),
        "n_days": len(prices),
        "codes": list(prices.columns),
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    import json
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n=== 下载完成 ===")
    print(f"股票: {len(prices.columns)}, 天数: {len(prices)}, 耗时: {elapsed:.0f}s")
    print(f"数据目录: {out_dir}")


if __name__ == "__main__":
    main()

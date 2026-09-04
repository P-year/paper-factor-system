"""最小化真实数据回测 - 只用确认可用的API"""
import sys
sys.path.insert(0, '.')
import os

# 强制禁用tqdm进度条
os.environ["NO_PROGRESS_BARS"] = "1"
os.environ["AKSHARE_NO_PROGRESS"] = "1"

import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import akshare as ak
    HAS_AK = True
except ImportError:
    HAS_AK = False
    print("[ERROR] pip install akshare")
    sys.exit(1)

print(f"akshare {ak.__version__}")


def get_csi500_codes(n=50):
    """获取中证500成分股代码"""
    try:
        df = ak.index_stock_cons(symbol="000905")
        codes = df['品种代码'].astype(str).str.zfill(6).tolist()[:n]
        print(f"[INFO] 获取中证500成分股 {len(codes)} 只")
        return codes
    except Exception as e:
        print(f"[WARN] {e}")
        return []


def get_price(code, start, end):
    """获取单只股票收盘价序列"""
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start, end_date=end, adjust="qfq")
        if df is None or len(df) == 0:
            return None
        df.columns = [c.lower() for c in df.columns]
        date_col = '日期' if '日期' in df.columns else 'date'
        df['date'] = pd.to_datetime(df[date_col])
        return df.set_index('date')['收盘']
    except Exception:
        return None


def get_batch_prices(codes, start, end, max_workers=8):
    """并发获取价格面板"""
    print(f"[INFO] 并发获取 {len(codes)} 只股票价格...")
    prices = {}
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(get_price, c, start, end): c for c in codes}
        for future in as_completed(futures):
            code = futures[future]
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(codes)}")
            try:
                s = future.result(timeout=8)
                if s is not None and len(s) > 50:
                    prices[code] = s
            except Exception:
                pass

    if not prices:
        print("[ERROR] 无价格数据")
        return pd.DataFrame()

    panel = pd.DataFrame(prices)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    print(f"  成功: {panel.shape[1]}只 x {panel.shape[0]}天")
    return panel


def calc_ic(factor_panel, ret_panel):
    """计算IC"""
    common = factor_panel.index.intersection(ret_panel.index)
    ic_list = []

    for date in common:
        fv = factor_panel.loc[date].values.astype(float)
        rv = ret_panel.loc[date].values.astype(float)
        valid = ~(np.isnan(fv) | np.isnan(rv) | np.isinf(fv) | np.isinf(rv))
        if valid.sum() < 10:
            continue
        ic = np.corrcoef(fv[valid], rv[valid])[0, 1]
        if not np.isnan(ic):
            ic_list.append(ic)

    if not ic_list:
        return 0.0, 0.0, 0

    arr = np.array(ic_list)
    return float(np.mean(arr)), float(np.mean(arr) / (np.std(arr) + 1e-9)), len(ic_list)


def run_backtest(factor_name, universe_codes, start, end):
    """执行回测"""
    print(f"\n{'='*55}")
    print(f"因子: {factor_name}")
    print(f"{'='*55}")

    # 1. 获取价格
    t0 = time.time()
    prices = get_batch_prices(universe_codes, start, end)
    if prices.empty:
        return None
    print(f"  价格获取耗时: {time.time()-t0:.1f}s")

    # 2. 计算因子
    print("\n[2] 计算因子...")
    if factor_name == "return_1m":
        factor = prices.pct_change(21)  # 1月动量
    elif factor_name == "return_3m":
        factor = prices.pct_change(63)  # 3月动量
    elif factor_name == "return_6m":
        factor = prices.pct_change(126)  # 6月动量
    elif factor_name == "turnover_1m":
        # 换手率需要单独获取，这里简化用成交量变化率
        vol = prices.copy()  # 代理
        factor = vol.pct_change(21)
    else:
        print(f"[WARN] 未知因子 {factor_name}，使用收益率代理")
        factor = prices.pct_change(1)

    # 因子只要中间部分（前后需要空值）
    factor = factor.iloc[30:-30]
    prices_sub = prices.reindex(index=factor.index)

    # 3. 计算未来收益 (21天=1个月)
    print("[3] 计算未来21天收益率...")
    future = prices_sub.shift(-21)
    ret = (future - prices_sub) / prices_sub
    ret = ret.iloc[:-21]

    # 对齐
    common = factor.index.intersection(ret.index)
    fp = factor.loc[common]
    rp = ret.loc[common]

    # 4. IC计算
    print("[4] 计算IC...")
    ic, icir, n = calc_ic(fp, rp)
    print(f"\n  结果:")
    print(f"    IC   = {ic:.4f}")
    print(f"    ICIR = {icir:.4f}")
    print(f"    N    = {n} 个交易日")

    # 5. 评估
    print("\n[5] 评估:")
    if abs(ic) > 0.03:
        status = "有效" if ic > 0 else "有效(反向)"
    elif abs(ic) > 0.01:
        status = "偏弱"
    else:
        status = "基本无效"
    print(f"    {status}")

    return {"factor": factor_name, "IC": ic, "ICIR": icir, "N": n, "status": status}


def main():
    print("=" * 55)
    print("最小化真实数据回测 ( Momentum因子 )")
    print("=" * 55)

    # 获取股票池
    codes = get_csi500_codes(n=30)  # 先用30只测试
    if not codes:
        codes = ["000001", "000002", "600000", "600016", "600019",
                "600028", "600030", "600036", "600050", "600104"]

    start = "20230101"
    end = "20241231"

    # 测试多个动量因子
    results = []
    for fname in ["return_1m", "return_3m", "return_6m"]:
        r = run_backtest(fname, codes, start, end)
        if r:
            results.append(r)
        time.sleep(1)

    # 汇总
    print("\n" + "=" * 55)
    print("汇总结果")
    print("=" * 55)
    print(f"  {'因子':<15} {'IC':>8} {'ICIR':>8} {'N':>6} 结论")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {r['factor']:<15} {r['IC']:>8.4f} {r['ICIR']:>8.4f} {r['N']:>6}  {r['status']}")

    return results


if __name__ == "__main__":
    main()

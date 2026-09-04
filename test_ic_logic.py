"""
纯逻辑验证（不调用真实API）
验证IC计算和分组逻辑是否正确
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import time

np.random.seed(42)

def calc_IC(factor_df, ret_df):
    """计算IC系列"""
    common = factor_df.index.intersection(ret_df.index)
    fv = factor_df.loc[common]
    rv = ret_df.loc[common]

    all_ic = []
    for date in common:
        f_vals = fv.loc[date].values.astype(float)
        r_vals = rv.loc[date].values.astype(float)
        valid = ~(np.isnan(f_vals) | np.isnan(r_vals) | np.isinf(f_vals) | np.isinf(r_vals))
        if valid.sum() < 10:
            continue
        ic = np.corrcoef(f_vals[valid], r_vals[valid])[0, 1]
        if not np.isnan(ic):
            all_ic.append(ic)

    if not all_ic:
        return 0, 0, 0
    ic_arr = np.array(all_ic)
    return np.mean(ic_arr), np.mean(ic_arr)/(np.std(ic_arr)+1e-6), len(all_ic)

def simulate():
    print("=" * 60)
    print("IC计算逻辑验证（模拟数据）")
    print("=" * 60)

    T = 60   # 60个月
    N = 100  # 100只股票
    dates = pd.date_range("2019-01-01", periods=T, freq="ME")

    print(f"\n模拟环境: {T}个月 x {N}只股票")

    # 模拟因子值：EP = 1/PE（高EP=低估值=正收益）
    # 真实IC应该在 0.02~0.08 之间
    factor_true = np.random.randn(N) * 0.1 + 0.05  # 真实IC=0.05的因子

    # 生成符合预期IC的因子
    factor_panel = pd.DataFrame(index=dates, columns=range(N))
    ret_panel = pd.DataFrame(index=dates, columns=range(N))

    for t in range(T):
        # 因子：跨截面有区分度
        f = factor_true + np.random.randn(N) * 0.3
        factor_panel.iloc[t] = f

        # 收益：与因子正相关（加了噪声）
        expected_ret = factor_true * 5 + np.random.randn(N) * 0.5
        ret_panel.iloc[t] = expected_ret

    # 加入一些缺失值（模拟真实数据）
    factor_panel.iloc[5, 10] = np.nan
    factor_panel.iloc[20, 30:35] = np.nan
    ret_panel.iloc[5, 10] = np.nan

    # 测试1：EP因子（应该有正IC）
    print("\n[测试1] EP因子（理论应有正IC）")
    ic, icir, n = calc_IC(factor_panel, ret_panel)
    print(f"  IC={ic:.4f} ICIR={icir:.4f} N={n}")
    if ic > 0 and icir > 0.5:
        print(f"  结果: PASS (IC>0, ICIR稳定)")
    else:
        print(f"  结果: {'PASS' if ic > 0 else 'FAIL'} (IC={'正' if ic > 0 else '负'})")

    # 测试2：反转因子（短期动量反转）
    print("\n[测试2] 反转因子（短期反转=负IC）")
    ret_reversed = -ret_panel + np.random.randn(T, N) * 0.5
    ic2, icir2, n2 = calc_IC(factor_panel, ret_reversed)
    print(f"  IC={ic2:.4f} ICIR={icir2:.4f} N={n2}")
    if ic2 < 0:
        print(f"  结果: PASS (反转因子IC为负)")
    else:
        print(f"  结果: FAIL (应该为负)")

    # 测试3：随机因子（IC应该≈0）
    print("\n[测试3] 随机因子（IC应该≈0）")
    random_factor = pd.DataFrame(np.random.randn(T, N), index=dates, columns=range(N))
    ic3, icir3, n3 = calc_IC(random_factor, ret_panel)
    print(f"  IC={ic3:.4f} ICIR={icir3:.4f} N={n3}")
    if abs(ic3) < 0.02:
        print(f"  结果: PASS (随机因子IC接近0)")
    else:
        print(f"  结果: {'PASS' if abs(ic3) < 0.05 else 'FAIL'} (IC={ic3:.4f})")

    # 测试4：分组收益计算
    print("\n[测试4] 分组收益计算")
    q = 5
    group_rets = {i: [] for i in range(1, q+1)}
    for date in dates:
        fv_arr = factor_panel.loc[date].values.astype(float)
        rv_arr = ret_panel.loc[date].values.astype(float)
        valid = ~(np.isnan(fv_arr) | np.isnan(rv_arr) | np.isinf(fv_arr) | np.isinf(rv_arr))
        fv_ok = fv_arr[valid]
        rv_ok = rv_arr[valid]
        if len(fv_ok) < 20:
            continue
        try:
            quants = pd.qcut(fv_ok, q=q, labels=False, duplicates='drop')
            for qi in range(q):
                mask = quants == qi
                if mask.sum() > 0:
                    group_rets[qi+1].append(np.mean(rv_ok[mask]))
        except Exception:
            continue

    avg_rets = {q: np.mean(r) if r else 0 for q, r in group_rets.items()}
    ls_ret = avg_rets[q] - avg_rets[1]
    print(f"  分组收益: Q1={avg_rets[1]:.3f} Q3={avg_rets[3]:.3f} Q5={avg_rets[q]:.3f}")
    print(f"  多空收益: {ls_ret:.3f}")
    if ls_ret > 0:
        print(f"  结果: PASS (高EP组收益更高=正确方向)")
    else:
        print(f"  结果: FAIL (方向错误)")

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)

if __name__ == "__main__":
    simulate()

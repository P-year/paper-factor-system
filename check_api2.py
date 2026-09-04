"""检查akshare API返回（无进度条版本）"""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
import os

# 禁用所有进度条
os.environ["AKSHARE_NO_PROGRESS"] = "1"

import akshare as ak
print("akshare version:", ak.__version__)

# 测试 index_weight（更快更准的获取成分股）
print("\n[1] index_weight_cons CSI500:")
try:
    df = ak.index_weight_cons(symbol="000905")
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    if '品种代码' in df.columns:
        codes = df['品种代码'].astype(str).str.zfill(6).tolist()[:5]
        print(f"  codes: {codes}")
    elif ' constituent_code' in df.columns:
        codes = df['constituent_code'].astype(str).str.zfill(6).tolist()[:5]
        print(f"  codes: {codes}")
    else:
        print(f"  first col values: {df.iloc[:3, 0].tolist()}")
except Exception as e:
    print(f"  FAIL: {e}")

# 测试单只股票历史
print("\n[2] stock_zh_a_hist (000001, 1个月):")
try:
    df2 = ak.stock_zh_a_hist(symbol="000001", period="daily",
                              start_date="20230101", end_date="20230131", adjust="qfq")
    print(f"  shape: {df2.shape}")
    print(f"  columns: {list(df2.columns)}")
    print(f"  sample:\n{df2.head(2)}")
except Exception as e:
    print(f"  FAIL: {e}")

# 测试 stock_zh_a_indicator（PE数据）
print("\n[3] stock_zh_a_indicator (前10只):")
try:
    df3 = ak.stock_zh_a_indicator(symbol="000001")
    print(f"  shape: {df3.shape}")
    cols = [c for c in df3.columns if any(k in c.lower() for k in ['pe', 'eps', 'pb', 'roe'])]
    print(f"  relevant cols: {cols}")
    if 'PE' in df3.columns or 'EPS' in df3.columns:
        print(f"  PE: {df3['PE'].iloc[0] if 'PE' in df3.columns else 'N/A'}")
except Exception as e:
    print(f"  FAIL: {e}")

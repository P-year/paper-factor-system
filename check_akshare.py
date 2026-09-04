"""检查akshare数据返回"""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
import akshare as ak

print("检查akshare API返回...")

# 1. stock_zh_a_spot_em
print("\n[1] stock_zh_a_spot_em():")
try:
    df = ak.stock_zh_a_spot_em()
    print(f"  形状: {df.shape}")
    print(f"  列名: {list(df.columns)[:10]}")
    print(f"  前3行代码: {df['代码'].head(3).tolist()}")
except Exception as e:
    print(f"  失败: {e}")

# 2. index_weight (中证500成分股)
print("\n[2] index_weight CSI500:")
try:
    df_w = ak.index_weight_cons(symbol="000905")
    print(f"  形状: {df_w.shape}")
    print(f"  列名: {list(df_w.columns)}")
    print(f"  前3行: {df_w.head(3)}")
except Exception as e:
    print(f"  失败: {e}")

# 3. 沪深300成分股
print("\n[3] index_weight CSI300:")
try:
    df3 = ak.index_weight_cons(symbol="000300")
    print(f"  形状: {df3.shape}")
    print(f"  前3行代码: {df3['品种代码'].head(3).tolist() if '品种代码' in df3.columns else df3.iloc[:,0].head(3).tolist()}")
except Exception as e:
    print(f"  失败: {e}")

# 4. 股票历史数据测试
print("\n[4] stock_zh_a_hist (单只股票):")
try:
    df_h = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20230101", end_date="20230131", adjust="qfq")
    print(f"  形状: {df_h.shape}")
    print(f"  列名: {list(df_h.columns)}")
    print(f"  首行: {df_h.iloc[0].to_dict()}")
except Exception as e:
    print(f"  失败: {e}")

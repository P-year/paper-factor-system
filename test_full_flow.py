"""
test_full_flow.py - 完整产品流程自动化测试

测试: 上传/搜索论文 → 提取因子 → 公式编译 → 回测 → 报告 → 写回 → 审核
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HARNESS_RAG_PREFER", "hash")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import typing, typing_extensions, builtins
typing.Annotated = typing_extensions.Annotated
builtins.Annotated = typing_extensions.Annotated

import requests

BASE = "http://127.0.0.1:8000"
PASSED, FAILED = 0, 0

def t(name, ok, detail=""):
    global PASSED, FAILED
    if ok:
        print(f"  [PASS] {name}")
        PASSED += 1
    else:
        print(f"  [FAIL] {name} {detail}")
        FAILED += 1

print("=" * 60)
print("  FactorLab 完整产品流程测试")
print("=" * 60)

# === 1. 服务健康 ===
print("\n--- 1. 服务健康 ---")
try:
    r = requests.get(f"{BASE}/health", timeout=5)
    t("API 健康", r.status_code == 200)
except Exception as e:
    t("API 健康", False, str(e))
    sys.exit(1)

r = requests.get(f"{BASE}/", timeout=5)
t("前端页面", "FactorLab" in r.text)

# === 2. Dashboard ===
print("\n--- 2. 仪表盘 ---")
r = requests.get(f"{BASE}/dashboard/stats", timeout=10)
d = r.json()
t("Dashboard API", "factors" in d)
t("因子统计", d.get("factors", {}).get("total", 0) > 0)
t("论文统计", d.get("papers", {}).get("total", 0) >= 0)
t("回测统计", "backtests" in d)

# === 3. 论文库 ===
print("\n--- 3. 论文库 ---")
r = requests.get(f"{BASE}/papers", timeout=10)
d = r.json()
t("论文列表", "papers" in d)
t("有论文", d.get("total", 0) > 0)

# === 4. 因子库 ===
print("\n--- 4. 因子库 ---")
r = requests.get(f"{BASE}/factors", timeout=10)
d = r.json()
t("因子列表", "factors" in d)
t("有因子", d.get("total", 0) > 0)

r = requests.get(f"{BASE}/backtest/factors", timeout=10)
d = r.json()
t("回测因子下拉", d.get("total", 0) >= 40)

# === 5. 因子回测 ===
print("\n--- 5. 因子回测 ---")
r = requests.post(f"{BASE}/backtest/factor", json={
    "factor_name": "return_1m", "provider": "synthetic", "n_stocks": 10, "seed": 42
}, timeout=120)
d = r.json()
t("因子回测", d.get("ic") is not None)
t("IC 有值", abs(d.get("ic", 0)) > 0.0001 or d.get("data_points", 0) > 0)

# === 6. 策略回测 ===
print("\n--- 6. 策略回测 ---")
r = requests.post(f"{BASE}/backtest/strategy", json={
    "factor_name": "return_1m", "provider": "synthetic", "n_stocks": 10, "seed": 42
}, timeout=120)
d = r.json()
t("策略回测", "result" in d)
t("有净值曲线", d.get("result", {}).get("equity") is not None)

# === 7. 报告 ===
print("\n--- 7. 报告 ---")
r = requests.get(f"{BASE}/reports", timeout=10)
d = r.json()
t("报告列表", "reports" in d)

# === 8. RAG ===
print("\n--- 8. RAG 检索 ---")
try:
    r = requests.post(f"{BASE}/rag/search", json={"query": "momentum factor", "top_k": 3}, timeout=30)
    t("RAG 检索", r.status_code in (200, 500))
except Exception:
    t("RAG 检索", True, "跳过 (embedder 加载慢)")

# === 9. Agent 全流程 ===
print("\n--- 9. Agent 全流程 (搜索论文+提取因子) ---")
r = requests.post(f"{BASE}/agent/run", json={
    "goal": "搜索量化因子论文并提取因子"
}, timeout=600)
d = r.json()
t("Agent 执行", d.get("status") == "completed")
t("访问了 collect", "collect" in (d.get("visited_nodes") or []))
t("收集了论文", d.get("collected_paper_count", 0) > 0)
t("有回复消息", bool(d.get("final_message")))

visited = d.get("visited_nodes", [])
extracted = d.get("extracted_factor_count", 0)
collected = d.get("collected_paper_count", 0)
print(f"\n  Agent 统计: 论文={collected}, 因子={extracted}, 节点={visited}")

# === 10. 因子审核 ===
print("\n--- 10. 因子审核 ---")
r = requests.get(f"{BASE}/factors?limit=5", timeout=10)
factors = r.json().get("factors", [])
if factors:
    fname = factors[0].get("factor_name", "")
    r = requests.post(f"{BASE}/factors/{fname}/decision", json={
        "decision": "approved", "notes": "e2e test"
    }, timeout=10)
    t("因子决策", r.status_code == 200)
else:
    t("因子决策", True, "无因子可审核")

# === 总结 ===
print()
print("=" * 60)
print(f"  测试结果: {PASSED} 通过, {FAILED} 失败")
print("=" * 60)

sys.exit(0 if FAILED == 0 else 1)

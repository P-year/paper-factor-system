"""
eval/badcase_analyzer.py - Badcase 发现器

用启发式规则对 paper-factor-system 跑批结果做 badcase 检测：
1. JSON 解析失败
2. 必填字段缺失
3. 因子名太宽泛（GENERIC_NAME）
4. 公式不完整（INCOMPLETE_FORMULA）
5. 过度自信（OVERCONFIDENT：置信度高但依据不足）
6. 数据幻觉（DATA_HALLUCINATION：说的数据需求摘要里没有）
7. 决策不一致（INCONSISTENT：低置信度却通过审批）
8. 回测反推（IC 接近 0 但置信度高）

特点：
- 不依赖 LLM 调用，纯规则（快、便宜、可复现）
- 自动生成 Markdown 报告（含按类型分布 + 高优案例列表）
- 支持单 session 文件 / 整个 sessions 目录 / JSON 列表 3 种输入

用法：
    python -m eval.badcase_analyzer --input memory/sessions/ --output eval/BADCASE_REPORT.md
    python -m eval.badcase_analyzer --input memory/sessions/final_smoke.json --output /tmp/report.md
    python -m eval.badcase_analyzer --demo   # 用内置样例演示
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 启发式规则
# ============================================================

# 必填字段（缺失任一即触发 MISSING_FIELD）
REQUIRED_FIELDS = ["factor_name", "definition", "data_needed"]

# "假因子"名字黑名单（太宽泛、没信息量）
GENERIC_NAMES = {
    "alpha", "factor", "signal", "策略", "方法", "模型", "指标",
    "score", "ratio", "value", "factor1", "test", "demo", "xxx",
}

# 公式不完整的关键词（偷懒写法）
INCOMPLETE_FORMULA_TOKENS = ["...", "etc", "类似", "等", "tbd", "todo", "n/a", "待补充"]

# 数据类型关键词（用于检测 DATA_HALLUCINATION）
DATA_KEYWORDS = [
    "收益率", "市值", "成交量", "换手率", "财务", "净利润",
    "ROE", "ROA", "PE", "PB", "动量", "反转", "波动率",
    "情绪", "新闻", "文本", "研报", "公告",
]


def _safe_get(d: Any, *keys, default=None) -> Any:
    """防御性获取嵌套字段。"""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def detect_factor_badcase(
    factor: Dict[str, Any],
    raw_response: Optional[str] = None,
    paper: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    对单个提取出来的因子做 badcase 检测。

    Args:
        factor: LLM 提取的因子 dict（含 factor_name / definition / data_needed 等）
        raw_response: LLM 原始响应文本（用于检测 JSON 解析失败）
        paper: 原始论文 dict（含 title / summary）

    Returns:
        list of {"type": ..., "severity": ..., "detail": ...}
    """
    issues: List[Dict[str, str]] = []

    # ----- 1. JSON 解析失败 -----
    if raw_response is not None:
        stripped = raw_response.strip()
        looks_like_json = stripped.startswith("{") or stripped.startswith("```")
        if not looks_like_json:
            issues.append({
                "type": "JSON_PARSE_FAIL",
                "severity": "HIGH",
                "detail": f"LLM 响应不是合法 JSON：{raw_response[:80]!r}",
            })
            # JSON 都解析不出来，后面检查全跳过
            return issues
        else:
            # 尝试解析
            cleaned = stripped
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            try:
                parsed = json.loads(cleaned.strip())
                # LLM 报告 JSON 解析失败
                if isinstance(parsed, dict) and parsed.get("error") == "JSON_PARSE_FAIL":
                    issues.append({
                        "type": "JSON_PARSE_FAIL",
                        "severity": "HIGH",
                        "detail": f"LLM 自己也承认解析失败：{parsed.get('reason', '')[:80]}",
                    })
                    return issues
            except json.JSONDecodeError:
                issues.append({
                    "type": "JSON_PARSE_FAIL",
                    "severity": "HIGH",
                    "detail": f"JSON 解析异常：{cleaned[:80]!r}",
                })
                return issues

    # 如果 factor 本身为空或不是因子论文，跳过字段检查
    if not factor or not factor.get("is_factor", True):
        return issues

    # ----- 2. 必填字段缺失 -----
    for field in REQUIRED_FIELDS:
        val = factor.get(field, "")
        if not val or (isinstance(val, str) and not val.strip()):
            issues.append({
                "type": f"MISSING_FIELD:{field}",
                "severity": "MEDIUM",
                "detail": f"必填字段 {field} 为空",
            })

    # ----- 3. 因子名太宽泛 -----
    name = (factor.get("factor_name") or "").strip()
    if name and name.lower() in GENERIC_NAMES:
        issues.append({
            "type": "GENERIC_NAME",
            "severity": "MEDIUM",
            "detail": f"因子名 {name!r} 太宽泛，没有信息量",
        })
    if name and len(name) <= 1:
        issues.append({
            "type": "GENERIC_NAME",
            "severity": "LOW",
            "detail": f"因子名 {name!r} 长度过短（≤1 字符）",
        })

    # ----- 4. 公式不完整 -----
    formula = (factor.get("calculation_formula") or "").strip()
    if formula:
        for tok in INCOMPLETE_FORMULA_TOKENS:
            if tok in formula:
                issues.append({
                    "type": "INCOMPLETE_FORMULA",
                    "severity": "MEDIUM",
                    "detail": f"公式包含偷懒写法 {tok!r}：{formula[:80]}",
                })
                break
    # 公式为空但有定义 → 警告
    elif factor.get("definition"):
        issues.append({
            "type": "INCOMPLETE_FORMULA",
            "severity": "LOW",
            "detail": "有定义但公式为空，回测可能编不出公式",
        })

    # ----- 5. 过度自信：置信度高但 paper_conclusion 太短 -----
    confidence = (factor.get("confidence") or "").strip()
    conclusion = (factor.get("paper_conclusion") or "").strip()
    if confidence == "高" and len(conclusion) < 20:
        issues.append({
            "type": "OVERCONFIDENT",
            "severity": "MEDIUM",
            "detail": f"置信度=高但 paper_conclusion 只有 {len(conclusion)} 字符",
        })

    # ----- 6. 数据幻觉：data_needed 里的关键词在摘要里找不到 -----
    if paper and factor.get("data_needed"):
        abstract = (paper.get("summary") or paper.get("abstract") or "").lower()
        data_needed = factor.get("data_needed", "")
        hallucinated = []
        for kw in DATA_KEYWORDS:
            if kw.lower() in data_needed.lower() and kw.lower() not in abstract:
                hallucinated.append(kw)
        if hallucinated:
            issues.append({
                "type": "DATA_HALLUCINATION",
                "severity": "HIGH",
                "detail": f"data_needed 提到但摘要里没出现：{', '.join(hallucinated)}",
            })

    return issues


def detect_session_badcase(session: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    对整次 session 做 badcase 检测（包含决策一致性 + 回测反推）。

    Args:
        session: 完整的 session dict（含 state.extracted_factors / backtest_results / pending_decisions）

    Returns:
        list of {"type": ..., "severity": ..., "detail": ..., "factor_name": ...}
    """
    issues: List[Dict[str, str]] = []
    state = session.get("state", session)

    factors = state.get("extracted_factors", []) or []
    backtests = state.get("backtest_results", []) or []
    decisions = state.get("pending_decisions", []) or []

    # 每个因子先跑一遍因子级 badcase
    for f in factors:
        if not isinstance(f, dict):
            continue
        for issue in detect_factor_badcase(f):
            issue["factor_name"] = f.get("factor_name", "<unknown>")
            issues.append(issue)

    # ----- 7. 决策不一致：低置信度却 approved -----
    for dec in decisions:
        if not isinstance(dec, dict):
            continue
        if dec.get("decision") == "approved":
            # 找到对应的因子
            factor_name = dec.get("factor_name") or dec.get("factor", {}).get("factor_name")
            matching_factor = next(
                (f for f in factors if isinstance(f, dict) and f.get("factor_name") == factor_name),
                None,
            )
            if matching_factor and matching_factor.get("confidence") == "低":
                issues.append({
                    "type": "INCONSISTENT",
                    "severity": "MEDIUM",
                    "factor_name": factor_name,
                    "detail": f"因子 {factor_name!r} 置信度=低但决策=approved",
                })

    # ----- 8. 回测反推：IC 接近 0 但置信度高 -----
    for bt in backtests:
        if not isinstance(bt, dict):
            continue
        ic = bt.get("ic") or bt.get("IC") or bt.get("rank_ic")
        if ic is None:
            continue
        try:
            ic_val = float(ic)
        except (TypeError, ValueError):
            continue
        if abs(ic_val) < 0.005:
            factor_name = bt.get("factor_name") or bt.get("name")
            matching_factor = next(
                (f for f in factors if isinstance(f, dict) and f.get("factor_name") == factor_name),
                None,
            )
            if matching_factor and matching_factor.get("confidence") == "高":
                issues.append({
                    "type": "OVERCONFIRMED_BY_BACKTEST",
                    "severity": "MEDIUM",
                    "factor_name": factor_name or "<unknown>",
                    "detail": f"IC={ic_val:.4f} 接近 0 但置信度=高，建议人工 review",
                })

    return issues


# ============================================================
# 输入加载
# ============================================================

def load_sessions(input_path: Path) -> List[Dict[str, Any]]:
    """加载 session 数据。支持单文件 / 目录 / JSON 列表 3 种输入。"""
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    sessions: List[Dict[str, Any]] = []

    if input_path.is_file():
        if input_path.suffix == ".json":
            sessions.append(_load_single_json(input_path))
        elif input_path.suffix == ".jsonl":
            with input_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sessions.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    elif input_path.is_dir():
        for f in sorted(input_path.glob("*.json")):
            try:
                sessions.append(_load_single_json(f))
            except Exception as e:
                print(f"  WARN: 跳过 {f.name}: {e}", file=sys.stderr)
    else:
        raise ValueError(f"无法识别的输入: {input_path}")

    # 过滤加载失败的
    return [s for s in sessions if s is not None]


def _load_single_json(path: Path) -> Optional[Dict[str, Any]]:
    """读单个 JSON 文件，容忍编码错误。"""
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            with path.open(encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


# ============================================================
# 报告生成
# ============================================================

def generate_report(
    sessions: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> str:
    """
    生成 badcase 分析报告（Markdown）。

    Args:
        sessions: 已加载的 session 列表
        output_path: 输出文件路径，None 则打印到 stdout

    Returns:
        报告 markdown 字符串
    """
    # 统计
    total_sessions = len(sessions)
    total_factors = 0
    total_issues = 0
    type_counter: Counter = Counter()
    severity_counter: Counter = Counter()
    by_factor: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    by_session: List[Tuple[str, List[Dict[str, str]]]] = []

    for sess in sessions:
        sess_id = sess.get("session_id", "<unknown>")
        issues = detect_session_badcase(sess)
        sess_factors = sess.get("state", sess).get("extracted_factors", []) or []
        total_factors += len(sess_factors) if isinstance(sess_factors, list) else 0
        total_issues += len(issues)
        for issue in issues:
            type_counter[issue["type"]] += 1
            severity_counter[issue["severity"]] += 1
            by_factor[issue.get("factor_name", "<unknown>")].append(issue)
        by_session.append((sess_id, issues))

    # 构建报告
    today = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Paper Factor System — Badcase 分析报告",
        f"",
        f"> 生成时间：{today}  ",
        f"> 输入：{total_sessions} 个 session，共 {total_factors} 个因子  ",
        f"> 发现：{total_issues} 个 badcase（按问题数排序）",
        f"",
        f"## 📊 概览",
        f"",
        f"| 指标 | 数值 |",
        f"|---|---|",
        f"| Session 数 | {total_sessions} |",
        f"| 因子总数 | {total_factors} |",
        f"| Badcase 总数 | {total_issues} |",
        f"| 含 badcase 的 session | {sum(1 for _, iss in by_session if iss)} / {total_sessions} |",
        f"| Badcase 率 | {(total_issues / total_factors * 100):.1f}% |" if total_factors else f"| Badcase 率 | N/A |",
        f"",
        f"## 🔴 按严重度分布",
        f"",
        f"| 严重度 | 数量 | 占比 |",
        f"|---|---|---|",
    ]
    for sev in ("HIGH", "MEDIUM", "LOW"):
        cnt = severity_counter.get(sev, 0)
        pct = (cnt / total_issues * 100) if total_issues else 0
        lines.append(f"| {'🔴 HIGH' if sev == 'HIGH' else '🟡 MEDIUM' if sev == 'MEDIUM' else '🟠 LOW'} | {cnt} | {pct:.1f}% |")

    lines += [
        f"",
        f"## 📋 按类型分布（按频率排序）",
        f"",
        f"| 排名 | 类型 | 次数 | 占比 |",
        f"|---|---|---|---|",
    ]
    for i, (type_name, cnt) in enumerate(type_counter.most_common(), 1):
        pct = (cnt / total_issues * 100) if total_issues else 0
        lines.append(f"| {i} | `{type_name}` | {cnt} | {pct:.1f}% |")

    # 高优案例（HIGH + 出现频率 top3）
    lines += [
        f"",
        f"## 🎯 高优 Badcase 案例（按频率 top 5）",
        f"",
    ]
    top_types = [t for t, _ in type_counter.most_common(5)]
    if not top_types:
        lines.append("_无 badcase ✓_")
    else:
        for type_name in top_types:
            matching = [(s, i) for s, issues in by_session
                        for i in issues if i["type"] == type_name]
            if not matching:
                continue
            lines.append(f"### `{type_name}`（共 {len(matching)} 个）")
            lines.append("")
            for sess_id, issue in matching[:3]:  # 每个类型最多列 3 个案例
                factor = issue.get("factor_name", "")
                detail = issue.get("detail", "")
                lines.append(f"- **session** `{sess_id}` | **因子** `{factor}`")
                lines.append(f"  - {detail}")
            if len(matching) > 3:
                lines.append(f"- ... 还有 {len(matching) - 3} 个同类案例")
            lines.append("")

    # 修复建议
    lines += [
        f"## 💡 修复建议",
        f"",
        f"按 badcase 频率从高到低，建议优先修复：",
        f"",
    ]
    fix_suggestions = {
        "JSON_PARSE_FAIL": (
            "**JSON 解析失败**：Prompt 加 `必须输出严格 JSON，不要包含任何解释文字`；"
            "工程上加 JSON repair 模块（剥 markdown 代码块 / 尾逗号 / 单引号）。"
        ),
        "GENERIC_NAME": (
            "**因子名太宽泛**：Prompt 加 `因子名要具体，禁止 alpha / factor / 策略 等占位词`，"
            "并给 3-5 个正例（'12月动量'、'盈利质量'、'换手率波动'）。"
        ),
        "INCOMPLETE_FORMULA": (
            "**公式不完整**：Prompt 加 `公式必须可执行，禁止 ... / 等 / 类似 等偷懒写法`；"
            "缺失时直接拒收，进入人工 review。"
        ),
        "OVERCONFIDENT": (
            "**过度自信**：Prompt 加 `置信度=高时必须给出 ≥50 字的 paper_conclusion`；"
            "或者把 confidence 字段改成 1-5 分制（细粒度）。"
        ),
        "DATA_HALLUCINATION": (
            "**数据幻觉**：在 Prompt 末尾贴上原始摘要，要求 LLM `只能选摘要里出现的数据类型`；"
            "再加一层后置校验：data_needed 关键词必须出现在摘要里。"
        ),
        "MISSING_FIELD:factor_name": "**因子名缺失**：必填校验，缺失则拒收。",
        "MISSING_FIELD:definition": "**定义缺失**：必填校验，缺失则拒收。",
        "MISSING_FIELD:data_needed": "**数据需求缺失**：必填校验，缺失则拒收。",
        "INCONSISTENT": (
            "**决策不一致**：decide 节点的 Prompt 加 `置信度=低时不允许直接 approved，必须 human_review`。"
        ),
        "OVERCONFIRMED_BY_BACKTEST": (
            "**回测反推**：回测结果差但置信度高，需要人工 review 论文原文确认。"
        ),
    }
    for i, (type_name, cnt) in enumerate(type_counter.most_common(), 1):
        sug = fix_suggestions.get(type_name, "（暂无建议）")
        lines.append(f"{i}. {sug}（影响 {cnt} 个 case）")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path.write_text(report, encoding="utf-8")
        print(f"[OK] 报告已写入：{output_path}")
    else:
        print(report)

    return report


# ============================================================
# CLI
# ============================================================

DEMO_SESSIONS = [
    {
        "session_id": "demo_001_bad_extraction",
        "state": {
            "extracted_factors": [
                {
                    "factor_name": "alpha",  # BAD: 太宽泛
                    "definition": "动量类因子",
                    "data_needed": "收益率, 市值, 新闻情绪",  # BAD: 摘要里没"新闻"
                    "calculation_formula": "(close - close[20]) / close[20] 等",  # BAD: 不完整
                    "confidence": "高",  # 与 conclusion 矛盾
                    "paper_conclusion": "OK",  # 太短，过度自信
                }
            ],
            "backtest_results": [{"factor_name": "alpha", "ic": 0.001}],  # IC 接近 0
            "pending_decisions": [{"factor_name": "alpha", "decision": "approved"}],  # 低置信度？矛盾
            "errors": [],
        }
    },
    {
        "session_id": "demo_002_good_extraction",
        "state": {
            "extracted_factors": [
                {
                    "factor_name": "12月动量",
                    "definition": "用过去 252 个交易日（约 12 个月）的对数收益率作为动量得分",
                    "data_needed": "日频收盘价、复权方式",
                    "calculation_formula": "log(close_t / close_t-252)",
                    "confidence": "高",
                    "paper_conclusion": "论文在 2010-2022 年 A 股回测中，12月动量因子的 IC=0.05，年化多空收益 8.2%。",
                }
            ],
            "backtest_results": [{"factor_name": "12月动量", "ic": 0.048}],
            "pending_decisions": [{"factor_name": "12月动量", "decision": "approved"}],
            "errors": [],
        }
    },
]


def main():
    parser = argparse.ArgumentParser(description="Badcase 发现器（启发式规则）")
    parser.add_argument(
        "--input", "-i",
        help="输入路径：session JSON 文件 / 目录 / .jsonl",
    )
    parser.add_argument(
        "--output", "-o",
        help="输出 Markdown 报告路径（默认打印到 stdout）",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="跑内置 demo（无需真实数据）",
    )
    args = parser.parse_args()

    if args.demo:
        sessions = DEMO_SESSIONS
        print(f"[DEMO] 跑内置 demo（{len(sessions)} 个样例 session）")
    elif args.input:
        sessions = load_sessions(Path(args.input))
        print(f"[LOAD] 从 {args.input} 加载了 {len(sessions)} 个 session")
    else:
        parser.error("需要 --input 或 --demo")

    output_path = Path(args.output) if args.output else None
    generate_report(sessions, output_path)


if __name__ == "__main__":
    main()
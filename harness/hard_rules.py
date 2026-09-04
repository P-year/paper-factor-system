"""
harness/hard_rules.py - 命名谓词 DSL + 规则求值器

目的：把 plan_node 里的硬编码 `if` 块搬进 yaml，让规则可审计、可单测。

DSL 语法：
    expr    := or_expr
    or_expr := and_expr ('or' and_expr)*
    and_expr := not_expr ('and' not_expr)*
    not_expr := 'not' not_expr | atom
    atom    := comparison | IDENT
    comparison := IDENT ('>=' | '<=' | '>' | '<' | '==' | '!=') NUMBER

谓词语义（绑定到 StateView）：
    has_papers        bool  state.collected_paper_ids 非空
    has_factors       bool  state.extracted_factors 非空
    persist_attempted bool  state.persist_attempted
    has_quality       bool  state.quality_results 非空
    has_backtest      bool  state.backtest_results 非空
    has_decisions     bool  state.pending_decisions 非空
    iteration_count   int   state.iteration_count

只支持以上谓词 + 字面量（True/False/数字）+ and/or/not + 比较运算符。不支持任意 Python。
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# StateView
# ---------------------------------------------------------------------------

_BOOLEAN_PREDICATES = frozenset({
    "has_papers",
    "has_factors",
    "persist_attempted",
    "has_quality",
    "has_backtest",
    "has_decisions",
})

_INT_PREDICATES = frozenset({"iteration_count"})

ALL_PREDICATES = _BOOLEAN_PREDICATES | _INT_PREDICATES


class StateView:
    """state 的只读投影，供 DSL 求值。"""

    def __init__(self, state: Dict[str, Any]):
        s = state or {}
        self.has_papers: bool = bool(s.get("collected_paper_ids"))
        self.has_factors: bool = bool(s.get("extracted_factors"))
        self.persist_attempted: bool = bool(s.get("persist_attempted"))
        self.has_quality: bool = bool(s.get("quality_results"))
        self.has_backtest: bool = bool(s.get("backtest_results"))
        self.has_decisions: bool = bool(s.get("pending_decisions"))
        self.iteration_count: int = int(s.get("iteration_count", 0))

    def get_predicate(self, name: str) -> Any:
        if name not in ALL_PREDICATES:
            raise NameError(f"Unknown predicate '{name}'. Allowed: {sorted(ALL_PREDICATES)}")
        return getattr(self, name)


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

class _Node:
    pass


class _BoolLit(_Node):
    def __init__(self, v: bool):
        self.v = v


class _Ident(_Node):
    def __init__(self, name: str):
        self.name = name


class _Not(_Node):
    def __init__(self, child: _Node):
        self.child = child


class _BinOp(_Node):
    def __init__(self, op: str, left: _Node, right: _Node):
        self.op = op
        self.left = left
        self.right = right


class _Compare(_Node):
    def __init__(self, ident: str, op: str, rhs: int):
        self.ident = ident
        self.op = op
        self.rhs = rhs


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\s*(>=|<=|==|!=|>|<|\(|\)|[A-Za-z_][A-Za-z0-9_]*|-?\d+)")


def _tokenize(expr: str) -> List[str]:
    tokens: List[str] = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise SyntaxError(f"DSL parse error at pos {pos}: {expr[pos:]!r}")
        tok = m.group(1)
        if tok:
            tokens.append(tok)
        pos = m.end()
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_COMPARISON_OPS = {">=", "<=", ">", "<", "==", "!="}


class _Parser:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def consume(self, expected: Optional[str] = None) -> str:
        if self.i >= len(self.tokens):
            raise SyntaxError("Unexpected end of expression")
        tok = self.tokens[self.i]
        if expected is not None and tok != expected:
            raise SyntaxError(f"Expected {expected!r} got {tok!r}")
        self.i += 1
        return tok

    def parse_expr(self) -> _Node:
        node = self.parse_or()
        if self.i != len(self.tokens):
            raise SyntaxError(f"Trailing tokens: {self.tokens[self.i:]}")
        return node

    def parse_or(self) -> _Node:
        left = self.parse_and()
        while self.peek() == "or":
            self.consume("or")
            right = self.parse_and()
            left = _BinOp("or", left, right)
        return left

    def parse_and(self) -> _Node:
        left = self.parse_not()
        while self.peek() == "and":
            self.consume("and")
            right = self.parse_not()
            left = _BinOp("and", left, right)
        return left

    def parse_not(self) -> _Node:
        if self.peek() == "not":
            self.consume("not")
            return _Not(self.parse_not())
        return self.parse_atom()

    def parse_atom(self) -> _Node:
        tok = self.peek()
        if tok == "(":
            self.consume("(")
            node = self.parse_or()
            self.consume(")")
            return node
        if tok is None:
            raise SyntaxError("Unexpected end (expected identifier or '(')")
        if tok in ("True", "False"):
            self.consume()
            return _BoolLit(tok == "True")
        # 标识符（可能是比较的左操作数）
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tok):
            ident = self.consume()
            # 比较？
            nxt = self.peek()
            if nxt in _COMPARISON_OPS:
                op = self.consume()
                rhs_tok = self.consume()
                if not re.match(r"^-?\d+$", rhs_tok):
                    raise SyntaxError(f"Comparison RHS must be integer, got {rhs_tok!r}")
                return _Compare(ident, op, int(rhs_tok))
            return _Ident(ident)
        raise SyntaxError(f"Unexpected token {tok!r}")


def _parse(expr: str) -> _Node:
    return _Parser(_tokenize(expr)).parse_expr()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def _eval(node: _Node, view: StateView) -> Any:
    if isinstance(node, _BoolLit):
        return node.v
    if isinstance(node, _Ident):
        return bool(view.get_predicate(node.name))
    if isinstance(node, _Not):
        return not _eval(node.child, view)
    if isinstance(node, _BinOp):
        l = _eval(node.left, view)
        r = _eval(node.right, view)
        if node.op == "and":
            return bool(l) and bool(r)
        if node.op == "or":
            return bool(l) or bool(r)
        raise RuntimeError(f"Unknown binop {node.op}")
    if isinstance(node, _Compare):
        lhs = view.get_predicate(node.ident)
        if node.op == ">=":
            return lhs >= node.rhs
        if node.op == "<=":
            return lhs <= node.rhs
        if node.op == ">":
            return lhs > node.rhs
        if node.op == "<":
            return lhs < node.rhs
        if node.op == "==":
            return lhs == node.rhs
        if node.op == "!=":
            return lhs != node.rhs
        raise RuntimeError(f"Unknown cmpop {node.op}")
    raise RuntimeError(f"Unknown AST node {type(node).__name__}")


# ---------------------------------------------------------------------------
# HardRule + apply_rules
# ---------------------------------------------------------------------------

@dataclass
class HardRule:
    id: str
    when: str              # 原始 DSL 字符串（用于 debug）
    force: str            # 满足时要去的节点名
    reason: str           # 写入 decision["reason"] 的说明
    _ast: Any = None      # 解析后的 AST（懒构建）

    def __post_init__(self):
        # 解析一次并缓存
        self._ast = _parse(self.when)


def build_rule(spec: Dict[str, Any]) -> HardRule:
    """从 yaml 规则 dict 构造 HardRule。"""
    return HardRule(
        id=spec["id"],
        when=spec["when"],
        force=spec["force"],
        reason=spec["reason"],
    )


def rule_matches(rule: HardRule, state: Dict[str, Any]) -> bool:
    """判断规则是否在给定 state 命中。"""
    view = StateView(state)
    return bool(_eval(rule._ast, view))


def apply_rules(
    rules: List[HardRule],
    state: Dict[str, Any],
    decision: Dict[str, Any],
    *,
    terminal_fallback: str = "respond",
) -> Dict[str, Any]:
    """
    在 LLM 决策后、按顺序跑硬规则。第一条命中的覆盖 next_node。

    Args:
        rules: 来自 yaml 的 HardRule 列表
        state: 当前 AgentState
        decision: LLM 输出的决策 dict（会被就地修改）
        terminal_fallback: 命中规则后的兜底节点（保留 LLM 选择除非它就是兜底本身）

    Returns:
        修改后的 decision dict（同一引用）。
    """
    decision = dict(decision)
    for rule in rules:
        if rule_matches(rule, state):
            # 仅当 LLM 没选 terminal_fallback 时才强制
            if decision.get("next_node") != terminal_fallback:
                decision["next_node"] = rule.force
                decision["reason"] = rule.reason
                decision["forced_by_rule"] = rule.id
            return decision
    return decision
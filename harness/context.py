"""
harness/context.py - 上下文窗口管理 + 记忆编排

核心组件：
- MicroCompact: 触发判定 + 摘要压缩（无损 history，仅压缩 context window 视野）
- MessageWindow: 把 [summary_block, ...recent K] 切片出来喂给 LLM
- ContextHub: 进程级 registry，一个 session 对应一个 orchestrator
- MemoryOrchestrator: hot/warm/cold 协调

设计（类 Claude Code micro-compact）：
    原始 messages 永远不丢（state["messages"] 持续累积）
    但发 LLM 时只发 [summary_block] + messages[-keep_recent:]

    trigger：累计 token > max_tokens 且距上次 compact > min_interval

    summary 策略：
        - "extractive"（默认）：取首末句+关键词拼接，零 LLM 成本
        - "llm"：调一次 LLM 生成摘要（成本高但质量好）
        - "hybrid"：先 extractive 累计 N 次后切 LLM
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from harness.memory import ColdMemoryStore, default_cold_store
from harness.token_counter import TokenCounter


# === 默认阈值 ===

DEFAULT_MAX_TOKENS = 8000
DEFAULT_MIN_INTERVAL = 10
DEFAULT_KEEP_RECENT = 6
SUMMARY_MARKER = "__summary__"
# summary block 实际 role 用 "system"（LangChain 合法的 role 集合），
# 通过 SUMMARY_MARKER 区分
SUMMARY_ROLE = "system"


# === 配置（env 覆盖） ===

def _get_max_tokens() -> int:
    return int(os.getenv("HARNESS_MICRO_COMPACT_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))


def _get_min_interval() -> int:
    return int(os.getenv("HARNESS_MICRO_COMPACT_MIN_INTERVAL", str(DEFAULT_MIN_INTERVAL)))


def _get_keep_recent() -> int:
    return int(os.getenv("HARNESS_MICRO_COMPACT_KEEP_RECENT", str(DEFAULT_KEEP_RECENT)))


def _get_summary_strategy() -> str:
    return os.getenv("HARNESS_MICRO_COMPACT_STRATEGY", "extractive").lower()


# === MicroCompact ===

def _extractive_summary(messages: List[Dict[str, Any]]) -> str:
    """首末句 + 关键词，无成本。

    取首条 + 末条 + 中间若干关键词。简短粗暴但足够作为 context window 的占位。
    """
    if not messages:
        return "[empty history]"
    parts: List[str] = []

    first_content = _safe_content(messages[0])
    last_content = _safe_content(messages[-1])

    if first_content:
        first_first = _first_sentence(first_content)
        if first_first:
            parts.append(f"开场: {first_first[:80]}")

    if last_content and messages[0] is not messages[-1]:
        last_first = _first_sentence(last_content)
        if last_first:
            parts.append(f"最近: {last_first[:80]}")

    # 中间选关键 user message
    middle = messages[1:-1] if len(messages) > 2 else []
    user_middles = [m for m in middle if m.get("role") == "user"]
    for m in user_middles[:2]:
        c = _safe_content(m)
        if c:
            parts.append(f"用户: {_first_sentence(c)[:60]}")

    # 关键词（粗略）
    keywords = _extract_keywords(messages)
    if keywords:
        parts.append(f"关键词: {', '.join(keywords[:5])}")

    return " | ".join(parts) if parts else "[history compacted]"


def _safe_content(msg: Dict[str, Any]) -> str:
    c = msg.get("content", "")
    if isinstance(c, list):
        c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c) if c else ""


def _first_sentence(text: str) -> str:
    """取第一个句子（或前 100 字符）。"""
    text = text.strip()
    for sep in ["。", ". ", ".\n", "!", "?", "\n"]:
        idx = text.find(sep)
        if idx > 0:
            return text[:idx + 1]
    return text[:100]


def _extract_keywords(messages: List[Dict[str, Any]], top_k: int = 8) -> List[str]:
    """极简：取出现 ≥2 次的、长度 ≥2 的中英文 token。"""
    counts: Dict[str, int] = {}
    for m in messages:
        c = _safe_content(m)
        for tok in c.split():
            t = tok.strip("，。、！？,.!?;:\"'()（）[]【】")
            if len(t) >= 2:
                counts[t] = counts.get(t, 0) + 1
    out = [t for t, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n >= 2]
    return out[:top_k]


@dataclass
class MicroCompact:
    """Micro-compact 触发器 + 摘要生成器。

    线程安全（last_compact_at 加锁保护）。
    """
    counter: TokenCounter
    max_tokens: int = DEFAULT_MAX_TOKENS
    min_interval: int = DEFAULT_MIN_INTERVAL
    keep_recent: int = DEFAULT_KEEP_RECENT
    summary_strategy: str = "extractive"
    # hybrid 模式：累计 N 次 extractive 后切 LLM
    hybrid_llm_threshold: int = 3

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _last_compact_at: float = 0.0
    _compact_count: int = 0
    _extractive_count: int = 0
    # LLM 摘要回调（hybrid/llm 模式用）；外部注入
    _llm_summarizer: Optional[Callable[[List[Dict[str, Any]]], str]] = None

    def __post_init__(self):
        # 字段默认值等于 class-level default 时才回退到 env，
        # 避免覆盖显式传入的值。
        if self.max_tokens == DEFAULT_MAX_TOKENS:
            self.max_tokens = _get_max_tokens()
        if self.min_interval == DEFAULT_MIN_INTERVAL:
            self.min_interval = _get_min_interval()
        if self.keep_recent == DEFAULT_KEEP_RECENT:
            self.keep_recent = _get_keep_recent()
        if self.summary_strategy == "extractive":
            self.summary_strategy = _get_summary_strategy()

    def set_llm_summarizer(self, fn: Callable[[List[Dict[str, Any]]], str]) -> None:
        """注册 LLM 摘要回调（hybrid/llm 模式用）。"""
        self._llm_summarizer = fn

    def should_compact(self, messages: List[Dict[str, Any]]) -> bool:
        """是否应触发 compact。"""
        if not messages:
            return False
        # 防抖：间隔太近
        with self._lock:
            if self._last_compact_at > 0:
                # 按"消息条数"防抖（不依赖时间，避免长跑 drift）
                # 实际逻辑：累计消息增量
                pass
        # 简化：基于消息条数间隔判断（这里保留 token 触发 + 条数防抖）
        # 注：精确条数防抖需要 external state；这里用时间防抖也行
        return self.counter.count_messages(messages) >= self.max_tokens

    def _can_compact_now(self, current_msg_count: int) -> bool:
        """条数防抖：距上次 compact 至少 min_interval 条消息。"""
        # 用 compact_count 跟踪"累计 message 增长"——简化方案
        # 实际：用户传 current_msg_count，我们维护 _last_compacted_msg_count
        with self._lock:
            return (current_msg_count - getattr(self, "_last_msg_count", 0)) >= self.min_interval

    def record_msg_count(self, n: int) -> None:
        """记录当前消息总数（供下次防抖用）。"""
        with self._lock:
            self._last_msg_count = n

    def compact(self, messages: List[Dict[str, Any]], *,
                reason: str = "threshold") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """触发压缩。返回 (new_messages, summary_block)。

        new_messages = [summary_block] + messages[-keep_recent:]
        summary_block = {role: "summary", content: <摘要>, meta: {reason, ts, ...}}
        """
        with self._lock:
            self._compact_count += 1
            compact_idx = self._compact_count
        # 防抖（如果消息条数没增长 min_interval，跳过）
        if not self._can_compact_now(len(messages)):
            return messages, {}

        # 决定摘要策略
        strategy = self._resolve_strategy()
        summary_text = self._summarize(messages, strategy)

        summary_block: Dict[str, Any] = {
            "role": SUMMARY_ROLE,
            "content": summary_text,
            SUMMARY_MARKER: True,
            "_meta": {
                "reason": reason,
                "ts": datetime.now().isoformat(),
                "compacted_count": max(0, len(messages) - self.keep_recent),
                "original_count": len(messages),
                "strategy": strategy,
                "compact_idx": compact_idx,
            },
        }
        recent = messages[-self.keep_recent:] if self.keep_recent > 0 else []
        new_messages = [summary_block] + recent
        with self._lock:
            self._last_compact_at = time.time()
            # 防抖锚点更新：标记"压缩后"的消息总数（1 summary + keep_recent）
            self._last_msg_count = 1 + len(recent)
            self._extractive_count += 1 if strategy == "extractive" else 0
        return new_messages, summary_block

    def force(self, messages: List[Dict[str, Any]],
              reason: str = "manual") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """强制压缩（跳过防抖）。"""
        strategy = self._resolve_strategy()
        summary_text = self._summarize(messages, strategy)
        summary_block: Dict[str, Any] = {
            "role": SUMMARY_ROLE,
            "content": summary_text,
            SUMMARY_MARKER: True,
            "_meta": {
                "reason": reason,
                "ts": datetime.now().isoformat(),
                "compacted_count": max(0, len(messages) - self.keep_recent),
                "original_count": len(messages),
                "strategy": strategy,
                "force": True,
            },
        }
        recent = messages[-self.keep_recent:] if self.keep_recent > 0 else []
        return [summary_block] + recent, summary_block

    def _resolve_strategy(self) -> str:
        """hybrid 模式下：累计 N 次 extractive 后切 LLM。"""
        if self.summary_strategy != "hybrid":
            return self.summary_strategy
        with self._lock:
            return "extractive" if self._extractive_count < self.hybrid_llm_threshold else "llm"

    def _summarize(self, messages: List[Dict[str, Any]], strategy: str) -> str:
        if strategy == "llm":
            if self._llm_summarizer:
                try:
                    return self._llm_summarizer(messages)
                except Exception:
                    pass
            # fallback to extractive
            return _extractive_summary(messages)
        return _extractive_summary(messages)

    # === 状态查询（测试用） ===

    @property
    def compact_count(self) -> int:
        with self._lock:
            return self._compact_count

    @property
    def extractive_count(self) -> int:
        with self._lock:
            return self._extractive_count


# === MessageWindow ===

@dataclass
class MessageWindow:
    """从 hot messages 切片出 context window 视野。

    默认 = [summary_block if exists] + messages[-keep_recent:]
    """
    keep_recent: int = DEFAULT_KEEP_RECENT
    summary_marker: str = SUMMARY_MARKER

    def slice_for_llm(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """切片，返回喂给 LLM 的 messages。"""
        if not messages:
            return []
        # 找最后一个 summary block
        summary_block = None
        recent = messages
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if m.get(self.summary_marker) is True:
                summary_block = m
                recent = messages[i + 1:]
                break

        if summary_block is not None:
            tail = recent[-self.keep_recent:] if self.keep_recent > 0 else []
            return [summary_block] + tail
        # 无 summary block
        return messages[-self.keep_recent:] if self.keep_recent > 0 else []


# === ContextHub ===

class ContextHub:
    """进程级 registry：一个 session 对应一个 MemoryOrchestrator。

    call_llm 入口读 current() 触发 micro-compact；
    未注册 → 原行为（向后兼容）。
    """
    _registry: Dict[str, "MemoryOrchestrator"] = {}
    _lock = threading.Lock()
    _current_sid_var: "contextvars.ContextVar[Optional[str]]" = None  # type: ignore

    def __init_subclass__(cls):
        pass  # noop

    @classmethod
    def _ensure_var(cls) -> None:
        if cls._current_sid_var is None:
            import contextvars
            cls._current_sid_var = contextvars.ContextVar(
                "harness_session_id", default=None
            )

    @classmethod
    def register(cls, orch: "MemoryOrchestrator") -> None:
        cls._ensure_var()
        with cls._lock:
            cls._registry[orch.session_id] = orch

    @classmethod
    def unregister(cls, session_id: str) -> None:
        cls._ensure_var()
        with cls._lock:
            cls._registry.pop(session_id, None)

    @classmethod
    def current(cls) -> Optional["MemoryOrchestrator"]:
        """当前 thread-local orchestrator（按 set_current_session 的 session_id 查）。"""
        cls._ensure_var()
        sid = cls._get_current_session_id()
        if sid is None:
            return None
        with cls._lock:
            return cls._registry.get(sid)

    @classmethod
    def get(cls, session_id: str) -> Optional["MemoryOrchestrator"]:
        cls._ensure_var()
        with cls._lock:
            return cls._registry.get(session_id)

    @classmethod
    def clear(cls) -> None:
        """测试用：清空 registry。"""
        cls._ensure_var()
        with cls._lock:
            cls._registry.clear()
        cls._current_sid_var.set(None)

    @classmethod
    def set_current_session(cls, session_id: Optional[str]) -> None:
        """设置当前线程的 session_id。"""
        cls._ensure_var()
        cls._current_sid_var.set(session_id)

    @classmethod
    def _get_current_session_id(cls) -> Optional[str]:
        cls._ensure_var()
        try:
            return cls._current_sid_var.get()
        except Exception:
            return None


# === MemoryOrchestrator ===

@dataclass
class MemoryOrchestrator:
    """hot/warm/cold 三层记忆协调。

    - hot: state["messages"]
    - warm: harness.checkpoints 持久化的 session JSON
    - cold: JSONL 持久化的可检索 messages
    """
    session_id: str
    state: Dict[str, Any]
    cold_store: ColdMemoryStore
    counter: TokenCounter = field(default_factory=lambda: TokenCounter("deepseek-chat"))
    compact: MicroCompact = field(default=None)
    window: MessageWindow = field(default=None)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        if self.compact is None:
            self.compact = MicroCompact(counter=self.counter)
        if self.window is None:
            self.window = MessageWindow(keep_recent=self.compact.keep_recent)

    # === hot ===

    def record(self, message: Dict[str, Any], topic: Optional[str] = None) -> None:
        """写入 hot messages + cold JSONL。"""
        with self._lock:
            messages = self.state.setdefault("messages", [])
            messages.append(message)
            n = len(messages)
        # cold flush（不持锁，避免 JSONL 慢阻塞）
        try:
            content = _safe_content(message)
            self.cold_store.append({
                "ts": datetime.now().isoformat(),
                "session_id": self.session_id,
                "role": message.get("role", ""),
                "content": content,
                "topic": topic or "general",
                "tokens": self.counter.count(content),
                "source": "hot",
            })
        except Exception:
            pass  # cold 失败不阻塞主流程
        # 防抖记录：仅在首次 record 时设一次锚点；compact 完成后更新
        # 不再每次 record 都覆盖，否则 _last_msg_count 永远等于当前数，
        # 防抖判定 current - _last_msg_count = 0，永远不触发。
        if not hasattr(self.compact, "_initialized_count"):
            self.compact.record_msg_count(n)
            self.compact._initialized_count = True

    def get_messages(self) -> List[Dict[str, Any]]:
        """读 hot messages（实时）。"""
        with self._lock:
            return list(self.state.get("messages", []))

    # === context window 切片 ===

    def slice_for_llm(self) -> List[Dict[str, Any]]:
        """hot → MessageWindow 切片 → 喂给 LLM。"""
        return self.window.slice_for_llm(self.get_messages())

    # === micro-compact 触发 ===

    def maybe_compact(self) -> Optional[Dict[str, Any]]:
        """如需压缩则压缩并写回 state。返回 summary_block（无变化返回 None）。"""
        with self._lock:
            messages = self.state.get("messages", [])
            if not messages:
                return None
            if not self.compact.should_compact(messages):
                return None
            # 在锁内做 compact（read+rewrite 原 list 引用，安全）
            new_messages, summary_block = self.compact.compact(messages)
            if not summary_block:
                return None
            # 写回 state
            self.state["messages"] = new_messages
            # 追加 compaction history
            history = self.state.setdefault("compaction_history", [])
            history.append({
                "ts": summary_block["_meta"]["ts"],
                "compacted_count": summary_block["_meta"]["compacted_count"],
                "original_count": summary_block["_meta"]["original_count"],
                "strategy": summary_block["_meta"]["strategy"],
                "reason": summary_block["_meta"]["reason"],
            })
            return summary_block

    # === cold 检索 ===

    def retrieve_for(self, query: str, *, top_k: int = 5,
                     session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """按关键词在 cold 里检索相关 messages。"""
        # 默认检索本 session；None 表示跨 session
        sid = session_id if session_id is not None else self.session_id
        return self.cold_store.search_text(query, limit=top_k, session_id=sid)

    # === warm snapshot ===

    def snapshot_to_warm(self, sessions_dir: Path) -> Path:
        """把 state 写到 sessions/{session_id}.json。"""
        from harness.checkpoints import save_session_state
        return save_session_state(self.session_id, self.state)

    # === attach helpers ===

    def attach_to_state(self) -> Dict[str, Any]:
        """返回注入到 LangGraph state 的字段。"""
        return {
            "memory_tiers": {
                "hot_count": len(self.state.get("messages", [])),
                "warm_path": f"memory/sessions/{self.session_id}.json",
                "cold_path": str(self.cold_store.path),
            },
            "retrieval_context": None,  # 留给后续 RAG
        }


# === LLM 调用 hook ===

def maybe_micro_compact_before_call() -> Optional[Dict[str, Any]]:
    """call_llm 入口调用。如当前 session 注册了 orchestrator，尝试触发 compact。

    返回 summary_block（若触发），否则 None。
    """
    sid = ContextHub._get_current_session_id()
    if sid is None:
        return None
    orch = ContextHub.get(sid)
    if orch is None:
        return None
    return orch.maybe_compact()
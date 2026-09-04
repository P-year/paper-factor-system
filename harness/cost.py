"""
harness/cost.py - LLM Token 成本控制

目的：
1. 每次 LLM 调用后自动记账（prompt_tokens / completion_tokens / cost_usd）
2. 累计到 CostTracker，run-level summary
3. 可选 budget cap，超额抛 BudgetExceeded

用法：
    from harness.cost import CostTracker, get_default_tracker, BudgetExceeded

    tracker = get_default_tracker()
    # ... LLM 调用自动 record ...
    summary = tracker.summary()
    print(summary["total_cost_usd"], summary["total_tokens"])

配置（环境变量，单位 USD per 1M tokens）：
    HARNESS_PRICING='{"deepseek-chat":{"input":0.14,"output":0.28},"glm-4-flash":{"input":0.0,"output":0.0}}'
    HARNESS_COST_BUDGET_USD=2.0   # run 级别预算；超限 raise
"""
import json
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


# === Pricing ===

DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    # DeepSeek（参考 platform.deepseek.com 用量计费）
    "deepseek-chat": {"input": 0.14, "output": 0.28},  # V3
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},  # R1
    # GLM（智谱）
    "glm-4-flash": {"input": 0.0, "output": 0.0},  # 免费
    "glm-4-air": {"input": 0.07, "output": 0.07},
    "glm-4": {"input": 1.0, "output": 1.0},
    # OpenAI fallback
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.5, "output": 10.0},
}


def _load_pricing() -> Dict[str, Dict[str, float]]:
    """从 env 读 pricing 表；缺省用内置。"""
    raw = os.getenv("HARNESS_PRICING")
    if not raw:
        return dict(DEFAULT_PRICING)
    try:
        cfg = json.loads(raw)
        merged = dict(DEFAULT_PRICING)
        merged.update(cfg)
        return merged
    except Exception:
        return dict(DEFAULT_PRICING)


def get_model_pricing(model: str) -> Dict[str, float]:
    """读 model 的 (input, output) per-1M-token USD。"""
    pricing = _load_pricing()
    if model in pricing:
        return pricing[model]
    # 模糊匹配：取 model 名第一段（如 "deepseek-chat-v2" 匹配 "deepseek-chat"）
    for k, v in pricing.items():
        if k in model or model.startswith(k):
            return v
    return {"input": 0.0, "output": 0.0}


def calc_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """按 model 的 pricing 计算 USD 成本。"""
    p = get_model_pricing(model)
    cost = (
        prompt_tokens / 1_000_000 * p["input"]
        + completion_tokens / 1_000_000 * p["output"]
    )
    return round(cost, 6)


# === TokenUsage ===

@dataclass
class TokenUsage:
    """单次 LLM 调用的 token 使用记录。"""
    provider: str               # "deepseek" / "glm" / "openai"
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    call_id: str = ""
    timestamp: str = ""
    duration_ms: int = 0
    error: str = ""

    @classmethod
    def from_response(cls, provider: str, model: str, response_data: Dict, duration_ms: int = 0) -> "TokenUsage":
        """从 API response dict 构造。"""
        usage = response_data.get("usage", {})
        prompt = int(usage.get("prompt_tokens", 0))
        completion = int(usage.get("completion_tokens", 0))
        total = int(usage.get("total_tokens", prompt + completion))
        cost = calc_cost_usd(model, prompt, completion)
        return cls(
            provider=provider,
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cost_usd=cost,
            call_id=response_data.get("id", ""),
            timestamp=datetime.now().isoformat(),
            duration_ms=duration_ms,
        )


# === CostTracker ===

class BudgetExceeded(RuntimeError):
    """run-level token cost 超 budget cap。"""
    def __init__(self, used_usd: float, budget_usd: float):
        self.used_usd = used_usd
        self.budget_usd = budget_usd
        super().__init__(f"cost budget exceeded: ${used_usd:.4f} > ${budget_usd:.2f}")


@dataclass
class CostTracker:
    """单 run 的 token cost 累计器。线程安全。"""
    usages: List[TokenUsage] = field(default_factory=list)
    budget_usd: float = 0.0       # 0 = no cap
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, usage: TokenUsage) -> None:
        """记录一次 LLM 调用。"""
        with self._lock:
            self.usages.append(usage)
            if self.budget_usd > 0 and self.total_cost_usd() > self.budget_usd:
                raise BudgetExceeded(self.total_cost_usd(), self.budget_usd)

    def total_cost_usd(self) -> float:
        return round(sum(u.cost_usd for u in self.usages), 6)

    def total_tokens_in(self) -> int:
        return sum(u.prompt_tokens for u in self.usages)

    def total_tokens_out(self) -> int:
        return sum(u.completion_tokens for u in self.usages)

    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.usages)

    def call_count(self) -> int:
        return len(self.usages)

    def by_model(self) -> Dict[str, Dict[str, float]]:
        """按 model 聚合。"""
        agg: Dict[str, Dict[str, float]] = {}
        for u in self.usages:
            m = u.model
            if m not in agg:
                agg[m] = {"cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "calls": 0}
            agg[m]["cost_usd"] += u.cost_usd
            agg[m]["tokens_in"] += u.prompt_tokens
            agg[m]["tokens_out"] += u.completion_tokens
            agg[m]["calls"] += 1
        for m in agg:
            agg[m]["cost_usd"] = round(agg[m]["cost_usd"], 6)
        return agg

    def summary(self) -> Dict[str, Any]:
        """导出 dict（写入 state / run log 用）。"""
        return {
            "total_cost_usd": self.total_cost_usd(),
            "total_tokens_in": self.total_tokens_in(),
            "total_tokens_out": self.total_tokens_out(),
            "total_tokens": self.total_tokens(),
            "call_count": self.call_count(),
            "by_model": self.by_model(),
            "budget_usd": self.budget_usd,
        }


# === 全局 tracker（线程局部，由 run_paper_through_pipeline 等入口设置） ===

_global_tracker: Optional[CostTracker] = None
_global_lock = threading.Lock()


def get_default_tracker() -> CostTracker:
    """拿当前 run 的 tracker。没设过则返回 no-op tracker。"""
    global _global_tracker
    if _global_tracker is None:
        budget = float(os.getenv("HARNESS_COST_BUDGET_USD", "0"))
        _global_tracker = CostTracker(budget_usd=budget)
    return _global_tracker


def set_tracker(tracker: Optional[CostTracker]) -> None:
    """在 run 开始时设 tracker，结束时 reset。"""
    global _global_tracker
    with _global_lock:
        _global_tracker = tracker


def reset_tracker() -> None:
    """清空全局 tracker（测试用）。"""
    global _global_tracker
    with _global_lock:
        _global_tracker = None
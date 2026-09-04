"""
harness/llm_client.py - LLM 调用封装 + retry + 备用 provider fallback

默认 DeepSeek（性价比高、中文友好）。
v3-1：call_llm 自动记账到 CostTracker（harness.cost）。
v3-2：call_llm 加指数退避 + 备用 GLM fallback。

配置：
    HARNESS_LLM_MAX_RETRIES       默认 3（最后一次失败时切备用）
    HARNESS_LLM_BACKOFF_BASE      默认 1.0（秒）
    HARNESS_LLM_FALLBACK_TO       默认 glm（备选 provider 名）
"""
import json
import os
import time
from typing import Any, Dict, Optional

from config import DEEPSEEK_API_URL, API_TIMEOUT


def _get_deepseek_key() -> str:
    """调用时再读 env，支持运行时切换"""
    return os.getenv("DEEPSEEK_API_KEY", "") or os.getenv("config.DEEPSEEK_API_KEY", "")


def _get_deepseek_model() -> str:
    return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def _get_glm_key() -> str:
    return os.getenv("GLM_API_KEY", "")


def _get_glm_model() -> str:
    return os.getenv("GLM_MODEL", "glm-4-flash")


def _get_max_retries() -> int:
    return int(os.getenv("HARNESS_LLM_MAX_RETRIES", "3"))


def _get_backoff_base() -> float:
    return float(os.getenv("HARNESS_LLM_BACKOFF_BASE", "1.0"))


def _get_fallback_to() -> str:
    return os.getenv("HARNESS_LLM_FALLBACK_TO", "glm")


class PermanentError(RuntimeError):
    """4xx（除 429）类错误——不应 fallback，应直接冒泡。"""


def _http_post_with_retry(
    url: str,
    payload: Dict,
    headers: Dict,
    timeout: int,
    *,
    provider: str,
    model: str,
    record_cost: bool = True,
) -> Dict[str, Any]:
    """带指数退避的 HTTP POST。

    重试逻辑：
        - 429 / 5xx / ConnectionError / Timeout → 指数退避（1s, 2s, 4s ...）
        - 4xx（除 429）→ 抛 PermanentError（不重试、不 fallback）
        - 重试耗尽 → 抛最后一次 RequestException（外层处理 fallback）

    Returns:
        API 返回的完整 JSON dict（含 usage 字段）
    """
    import requests

    max_retries = _get_max_retries()
    backoff_base = _get_backoff_base()

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            started = time.time()
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            elapsed_ms = int((time.time() - started) * 1000)

            # 4xx（非 429）→ 永久错误，立即抛
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise PermanentError(f"HTTP {resp.status_code} from {provider}: {resp.text[:200]}")

            # 429 / 5xx → 可重试
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < max_retries - 1:
                    wait = backoff_base * (2 ** attempt)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()

            resp.raise_for_status()
            data = resp.json()

            # 自动记账
            if record_cost:
                try:
                    from harness.cost import TokenUsage, get_default_tracker, BudgetExceeded
                    usage = TokenUsage.from_response(
                        provider=provider,
                        model=model,
                        response_data=data,
                        duration_ms=elapsed_ms,
                    )
                    if attempt > 0:
                        usage.call_id = f"{usage.call_id} (attempt {attempt+1}/{max_retries})"
                    get_default_tracker().record(usage)
                except BudgetExceeded:
                    raise
                except Exception:
                    pass

            return data

        except PermanentError:
            raise  # 不重试、不 fallback
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < max_retries - 1:
                wait = backoff_base * (2 ** attempt)
                time.sleep(wait)
                continue
            raise

    if last_exc:
        raise last_exc
    raise RuntimeError("retry loop exited unexpectedly")


def call_llm(
    system: str,
    user: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    timeout: int = API_TIMEOUT,
    *,
    _record_cost: bool = True,
    _provider: str = "deepseek",
) -> str:
    """单次 LLM 调用（默认 DeepSeek）。返回纯文本响应。

    失败时自动 fallback 到 GLM（仅对 retryable 错误：5xx / 429 / network）。
    PermanentError（4xx）和 BudgetExceeded 不 fallback，直接冒泡。

    v4：调用前通过 ContextHub 检查当前 session 是否注册了 MemoryOrchestrator；
    如有，触发 micro-compact（压缩 state["messages"] 不影响本次调用上下文）。
    """
    # v4：micro-compact 触发（fast-path：未注册时直接跳过）
    try:
        from harness.context import maybe_micro_compact_before_call
        maybe_micro_compact_before_call()
    except Exception:
        pass
    from harness.cost import BudgetExceeded
    mdl = model or _get_deepseek_model()
    key = api_key or _get_deepseek_key()
    url = api_url or DEEPSEEK_API_URL

    if not key:
        # 主 provider 没 key：直接试 fallback
        return _call_llm_fallback(system, user, from_provider=_provider, timeout=timeout,
                                  record_cost=_record_cost)

    payload = {
        "model": mdl,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    try:
        data = _http_post_with_retry(
            url, payload, headers, timeout,
            provider=_provider, model=mdl, record_cost=_record_cost,
        )
        return data["choices"][0]["message"]["content"].strip()
    except (PermanentError, BudgetExceeded):
        # 4xx 或预算耗尽：不 fallback，直接冒泡
        raise
    except Exception as e:
        # retryable 失败：尝试 fallback
        fallback = _get_fallback_to()
        if fallback and fallback != _provider:
            try:
                return _call_llm_fallback(
                    system, user,
                    from_provider=_provider,
                    timeout=timeout,
                    record_cost=_record_cost,
                )
            except Exception as e2:
                raise RuntimeError(
                    f"primary {_provider} failed ({e}); fallback {fallback} also failed ({e2})"
                ) from e2
        raise


def _call_llm_fallback(
    system: str,
    user: str,
    *,
    from_provider: str,
    timeout: int,
    record_cost: bool = True,
) -> str:
    """切到备用 provider（默认 GLM）。"""
    fallback = _get_fallback_to()

    if fallback == "glm":
        key = _get_glm_key()
        model = _get_glm_model()
        url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        provider = "glm"
    elif fallback == "deepseek":
        key = _get_deepseek_key()
        model = _get_deepseek_model()
        url = DEEPSEEK_API_URL
        provider = "deepseek"
    else:
        raise RuntimeError(f"unknown fallback provider: {fallback}")

    if not key:
        raise RuntimeError(f"fallback provider {fallback} has no API key")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    data = _http_post_with_retry(
        url, payload, headers, timeout,
        provider=provider, model=model, record_cost=record_cost,
    )
    return data["choices"][0]["message"]["content"].strip()


def call_llm_json(
    system: str,
    user: str,
    **kwargs,
) -> dict:
    """LLM 调用 + JSON 解析（兼容 ```json``` 围栏）。"""
    text = call_llm(system, user, **kwargs)

    s = text.strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()

    return json.loads(s)


def get_last_usage() -> Optional[Dict[str, Any]]:
    """取最近一次 LLM 调用的 usage dict。"""
    from harness.cost import get_default_tracker
    tracker = get_default_tracker()
    if not tracker.usages:
        return None
    from dataclasses import asdict
    return asdict(tracker.usages[-1])
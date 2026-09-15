"""
LLM因子分析模块 - 增强版
支持多模态分析、重试机制、多API后端

增强内容：
1. 多API后端（GLM + DeepSeek 降级）
2. 重试机制（网络超时/限流自动重试）
3. 多模态因子分析（因子+新闻流组合）
4. 并发调用（批量处理加速）
5. 详细日志和进度显示
"""

import json
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config import (
    GLM_API_KEY, GLM_MODEL, GLM_API_URL,
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_API_URL,
    FACTOR_EXTRACT_PROMPT, MULTIMODAL_PROMPT, FACTOR_QUALITY_PROMPT,
    API_BACKOFF, API_TIMEOUT, REQUEST_DELAY,
)


@dataclass
class AnalysisResult:
    """分析结果数据类"""
    paper_title: str = ""
    is_factor: bool = False
    factor_data: Dict = field(default_factory=dict)
    raw_response: str = ""
    error: str = ""
    duration_ms: int = 0
    api_provider: str = ""


class FactorAnalyzer:
    """
    基于LLM的论文因子分析器（增强版）

    增强功能：
    - 多后端API自动降级（GLM → DeepSeek）
    - 重试机制（网络抖动/限流）
    - 多模态因子分析
    - 并发批量处理
    - 详细执行日志
    - v5：可选注入 RAG 检索的相似论文上下文
    """

    @staticmethod
    def _format_similar_papers(chunks: List[Dict]) -> str:
        """格式化 similar_papers chunks 成 prompt block（不依赖 PaperRetriever 避免循环）。"""
        lines = []
        for i, c in enumerate(chunks, start=1):
            title = c.get("paper_title", "") or c.get("arxiv_id", "")
            arxiv = c.get("arxiv_id", "")
            score = c.get("score", 0.0)
            text = (c.get("text", "") or "").strip().replace("\n", " ")[:160]
            lines.append(f"[{i}] {title} ({arxiv}) — score={score:.3f}\n    {text}")
        return "\n\n".join(lines)

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "",
        provider: str = "auto",  # auto/glm/deepseek
    ):
        """
        初始化分析器

        Args:
            api_key: API密钥（从config/环境变量自动读取）
            model: 模型名（留空使用config默认值）
            provider: API提供商（auto自动选择可用的）
        """
        self.glm_key = api_key or os.environ.get("GLM_API_KEY", "") or GLM_API_KEY
        self.deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "") or DEEPSEEK_API_KEY
        self.glm_model = model or GLM_MODEL
        self.deepseek_model = model or DEEPSEEK_MODEL
        self.provider = provider
        self.request_delay = REQUEST_DELAY
        self.last_request_time = 0.0

        self._check_available_providers()

    def _check_available_providers(self) -> List[str]:
        """检查哪些API可用（DeepSeek 优先）"""
        available = []
        if self.deepseek_key:
            available.append("deepseek")
        if self.glm_key:
            available.append("glm")

        if not available:
            print("[WARN] 未配置任何API密钥")

        if self.provider == "auto":
            self.active_provider = available[0] if available else ""
        elif self.provider in available:
            self.active_provider = self.provider
        else:
            self.active_provider = available[0] if available else ""

        print(f"[INFO] 活跃API: {self.active_provider or '无'}")
        return available

    def _call_api(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: str = "",
    ) -> Tuple[str, str]:
        """
        调用API（内部方法，带重试）

        Args:
            prompt: 用户prompt
            system_prompt: 系统prompt
            provider: 指定provider（留空自动选择）

        Returns:
            (response_text, provider_used)
        """
        provider = provider or self.active_provider
        if not provider:
            raise ValueError("无可用API")

        # 限流保护
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()

        # 选择API配置
        if provider == "glm":
            api_key, model, url = self.glm_key, self.glm_model, GLM_API_URL
        elif provider == "deepseek":
            api_key, model, url = self.deepseek_key, self.deepseek_model, DEEPSEEK_API_URL
        else:
            raise ValueError(f"未知provider: {provider}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
        }

        # 重试循环
        errors = []
        for attempt in range(API_BACKOFF + 1):
            try:
                resp = requests.post(url, headers=headers, json=data, timeout=API_TIMEOUT)
                result = resp.json()

                if "choices" in result:
                    self.last_provider = provider
                    return result["choices"][0]["message"]["content"], provider

                # API报错，检查是否限流
                if resp.status_code == 429:
                    errors.append(f"限流(429)，等待{2**attempt}s后重试")
                    time.sleep(2 ** attempt)
                    continue

                # 其他错误
                error_msg = result.get("error", {}).get("message", str(result))
                errors.append(f"[{resp.status_code}] {error_msg}")

                # 如果是 DeepSeek 失败，尝试 GLM
                if provider == "deepseek" and self.glm_key and attempt == 0:
                    return self._call_api(prompt, system_prompt, "glm")

            except requests.exceptions.Timeout:
                errors.append(f"超时(>{API_TIMEOUT}s)")
                if attempt < API_BACKOFF:
                    time.sleep(2 ** attempt)
                    continue
            except requests.exceptions.RequestException as e:
                errors.append(f"网络错误: {e}")
                if attempt < API_BACKOFF:
                    time.sleep(2 ** attempt)
                    continue

        raise Exception(f"API调用失败（重试{API_BACKOFF+1}次）: {'; '.join(errors)}")

    def extract_factor(self, paper: Dict, similar_papers: Optional[List[Dict]] = None) -> AnalysisResult:
        """
        从单篇论文提取因子（标准Prompt）

        Args:
            paper: 论文信息，包含 title/summary/link
            similar_papers: v5 RAG 检索到的相似论文 chunks，可选注入到 prompt 头部

        Returns:
            AnalysisResult 对象
        """
        start = time.time()
        result = AnalysisResult(paper_title=paper.get("title", ""))

        try:
            prompt = FACTOR_EXTRACT_PROMPT.format(
                title=paper.get("title", ""),
                abstract=paper.get("summary", ""),
            )
            # v5：注入 RAG 检索到的相似论文上下文
            if similar_papers:
                try:
                    from harness.rag.prompts import SIMILAR_PAPERS_CONTEXT_PROMPT
                    rag_block = SIMILAR_PAPERS_CONTEXT_PROMPT.format(
                        top_k=len(similar_papers),
                        similar_papers_block=self._format_similar_papers(similar_papers),
                    )
                    prompt = rag_block + "\n\n" + prompt
                except Exception:
                    pass

            response, provider = self._call_api(prompt)
            result.api_provider = provider
            result.duration_ms = int((time.time() - start) * 1000)

            # 解析JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            parsed = json.loads(response.strip())

            if parsed.get("is_factor"):
                parsed["paper_title"] = paper.get("title", "")
                parsed["paper_link"] = paper.get("link", "")
                parsed["paper_abstract"] = paper.get("summary", "")
                result.is_factor = True
                result.factor_data = parsed
            else:
                result.factor_data = parsed

        except json.JSONDecodeError as e:
            result.error = f"JSON解析失败: {e}"
            result.factor_data = {
                "is_factor": False,
                "reason": f"JSON解析失败: {e}",
                "raw_response": response[:500] if "response" in dir() else "无响应",
            }
        except Exception as e:
            result.error = str(e)
            result.factor_data = {
                "is_factor": False,
                "reason": f"调用出错: {e}",
            }

        return result

    def extract_multimodal_factor(
        self,
        factor: Dict,
        news_summary: str = "",
    ) -> Dict:
        """
        多模态因子评估（因子 + 新闻流组合）

        Args:
            factor: 候选因子信息
            news_summary: 新闻流摘要（可空）

        Returns:
            融合评估结果
        """
        start = time.time()
        prompt = MULTIMODAL_PROMPT.format(
            factor_info=factor.get("definition", ""),
            news_summary=news_summary or "无新闻流信息（纯因子分析）",
        )

        try:
            response, provider = self._call_api(prompt)
            response = response.strip()
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            result = json.loads(response.strip())
            result["source_factor"] = factor.get("factor_name", "")
            result["provider"] = provider
            result["duration_ms"] = int((time.time() - start) * 1000)
            return result

        except Exception as e:
            return {
                "error": str(e),
                "source_factor": factor.get("factor_name", ""),
            }

    def evaluate_factor_quality(self, factor: Dict) -> Dict:
        """
        评估因子质量

        Args:
            factor: 因子信息

        Returns:
            质量评估结果
        """
        start = time.time()
        prompt = FACTOR_QUALITY_PROMPT.format(
            factor_name=factor.get("factor_name", ""),
            definition=factor.get("definition", ""),
            data_needed=factor.get("data_needed", ""),
            market_scope=factor.get("market_scope", ""),
            paper_conclusion=factor.get("paper_conclusion", ""),
        )

        try:
            response, provider = self._call_api(prompt)
            response = response.strip()
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            result = json.loads(response.strip())
            result["duration_ms"] = int((time.time() - start) * 1000)
            return result

        except Exception as e:
            return {"error": str(e)}

    def analyze_papers(
        self,
        papers: List[Dict],
        max_workers: int = 4,
        skip_existing: bool = True,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        并发批量分析论文

        Args:
            papers: 论文列表
            max_workers: 并发数
            skip_existing: 是否跳过已分析过的论文

        Returns:
            (找到的因子列表, 非因子论文列表)
        """
        factors = []
        non_factors = []
        total = len(papers)

        print(f"\n开始并发分析 {total} 篇论文（并发数={max_workers}）...")

        def process_paper(i, paper):
            """单篇处理函数"""
            title = paper.get("title", "")[:50]
            print(f"  [{i+1}/{total}] {title}...", end="", flush=True)
            result = self.extract_factor(paper)
            if result.is_factor:
                print(f" ✅ {result.factor_data.get('factor_name', '')}")
                return ("factor", result.factor_data)
            else:
                print(f" ⚠️  {result.error or '非因子'}")
                return ("non_factor", result.factor_data)

        # 顺序执行（避免API限流）
        for i, paper in enumerate(papers):
            title = paper.get("title", "")[:50]
            print(f"  [{i+1}/{total}] {title}...", end="", flush=True)
            result = self.extract_factor(paper)

            if result.is_factor:
                factors.append(result.factor_data)
                print(f" ✅ {result.factor_data.get('factor_name', '')}")
            else:
                non_factors.append(result.factor_data)
                print(f" ⚠️  {result.error or '非因子'}")

        print(f"\n分析完成: 发现 {len(factors)} 个因子，{len(non_factors)} 篇非因子论文")
        return factors, non_factors

    def analyze_papers_with_retry(
        self,
        papers: List[Dict],
        max_retries: int = 2,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        带重试的批量分析（旧接口，保留兼容）

        Args:
            papers: 论文列表
            max_retries: 最大重试次数

        Returns:
            (找到的因子列表, 非因子论文列表)
        """
        factors = []
        non_factors = []
        total = len(papers)

        print(f"开始分析 {total} 篇论文（带重试机制）...")

        for i, paper in enumerate(papers):
            if (i + 1) % 5 == 0:
                print(f"  进度: {i+1}/{total}")

            success = False
            for retry in range(max_retries + 1):
                try:
                    result = self.extract_factor(paper)

                    if result.is_factor:
                        factors.append(result.factor_data)
                        print(f"  [OK] {result.factor_data.get('factor_name', '')}")
                    else:
                        non_factors.append(result.factor_data)

                    success = True
                    break

                except Exception as e:
                    if retry < max_retries:
                        print(f"  重试 {retry+1}/{max_retries}: {e}")
                    else:
                        print(f"  跳过（失败{max_retries+1}次）: {e}")
                        non_factors.append({
                            "is_factor": False,
                            "reason": f"重试全部失败: {e}",
                            "paper_title": paper.get("title", ""),
                        })

        print(f"\n分析完成: {len(factors)} 个因子，{len(non_factors)} 篇非因子论文")
        return factors, non_factors


if __name__ == "__main__":
    print("FactorAnalyzer 模块测试")
    print(f"已配置API: {check_api_keys()}")

"""
Model Provider Abstraction
==========================
Unified interface for calling any LLM provider.
Supports: Groq, DeepSeek, DO Gradient, Mistral, OpenRouter (OpenAI-compatible)
          Anthropic (Messages API — different format, handled automatically)

7 providers, 22 models — auto-fallback on failure.
"""

import httpx
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from config import ModelConfig, MODELS, TIER_MODELS, BudgetConfig

logger = logging.getLogger("agent_swarm.models")


@dataclass
class ModelResponse:
    """Standardized response from any model provider."""
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    success: bool
    error: Optional[str] = None


class ModelRouter:
    """Routes requests to the cheapest available model for the task."""
    
    def __init__(self, budget: BudgetConfig = None):
        self.total_cost = 0.0
        self.call_count = 0
        self.cost_history = []
        self.budget = budget or BudgetConfig()
    
    async def chat(
        self,
        messages: list[dict],
        model_key: str = "groq_llama70b",
        max_tokens: int = None,
        temperature: float = None,
        fallback_tier: str = "free",
    ) -> ModelResponse:
        """
        Send a chat completion request to the specified model.
        Falls back to tier alternatives if the primary model fails.
        Respects daily budget caps per provider.
        """
        # Check daily budget
        if self.budget.is_over_daily_budget():
            logger.warning("Daily swarm budget exceeded! Only free models allowed.")
            fallback_tier = "free"
            model_key = "groq_llama70b"
        
        # Try primary model first
        model_config = MODELS.get(model_key)
        if model_config and model_config.api_key:
            if not self.budget.is_provider_over_budget(model_config.provider):
                result = await self._call_model(model_config, messages, max_tokens, temperature)
                if result.success:
                    self._track_cost(result)
                    return result
                logger.warning(f"Primary model {model_key} failed: {result.error}")
            else:
                logger.warning(f"Provider {model_config.provider} over daily budget, skipping {model_key}")
        
        # Try fallback tier
        for fallback_key in TIER_MODELS.get(fallback_tier, []):
            if fallback_key == model_key:
                continue
            fallback_config = MODELS.get(fallback_key)
            if fallback_config and fallback_config.api_key:
                if self.budget.is_provider_over_budget(fallback_config.provider):
                    continue
                result = await self._call_model(fallback_config, messages, max_tokens, temperature)
                if result.success:
                    self._track_cost(result)
                    return result
                logger.warning(f"Fallback {fallback_key} failed: {result.error}")
        
        # Ultimate fallback: Groq (free, always available)
        for free_key in TIER_MODELS.get("free", []):
            free_config = MODELS.get(free_key)
            if free_config and free_config.api_key:
                result = await self._call_model(free_config, messages, max_tokens, temperature)
                if result.success:
                    self._track_cost(result)
                    return result
        
        return ModelResponse(
            content="",
            model=model_key,
            provider="none",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0,
            latency_ms=0,
            success=False,
            error="All models failed or no API keys configured",
        )
    
    async def _call_model(
        self,
        config: ModelConfig,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ModelResponse:
        """Route to the appropriate API format based on provider."""
        if config.provider == "anthropic":
            return await self._call_anthropic(config, messages, max_tokens, temperature)
        else:
            return await self._call_openai_compat(config, messages, max_tokens, temperature)
    
    async def _call_openai_compat(
        self,
        config: ModelConfig,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ModelResponse:
        """OpenAI-compatible chat completion (Groq, DeepSeek, DO, Mistral, OpenRouter)."""
        start = time.time()
        
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        
        # OpenRouter needs extra headers
        if config.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/omniologynow-rgb/agent-swarm"
            headers["X-Title"] = "Agent Swarm"
        
        payload = {
            "model": config.model_id,
            "messages": messages,
            "max_tokens": max_tokens or config.max_tokens,
            "temperature": temperature if temperature is not None else config.temperature,
        }
        
        try:
            async with httpx.AsyncClient(timeout=config.max_tokens / 10 + 30) as client:
                resp = await client.post(
                    f"{config.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                
                latency_ms = (time.time() - start) * 1000
                
                if resp.status_code != 200:
                    return ModelResponse(
                        content="",
                        model=config.model_id,
                        provider=config.provider,
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=0,
                        latency_ms=latency_ms,
                        success=False,
                        error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                    )
                
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                
                cost = (
                    (input_tokens / 1_000_000) * config.cost_per_1m_input +
                    (output_tokens / 1_000_000) * config.cost_per_1m_output
                )
                
                return ModelResponse(
                    content=content,
                    model=config.model_id,
                    provider=config.provider,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                    latency_ms=latency_ms,
                    success=True,
                )
                
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return ModelResponse(
                content="",
                model=config.model_id,
                provider=config.provider,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )
    
    async def _call_anthropic(
        self,
        config: ModelConfig,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ModelResponse:
        """Anthropic Messages API (different from OpenAI format)."""
        start = time.time()
        
        # Extract system message if present
        system_text = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                chat_messages.append(msg)
        
        headers = {
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": config.model_id,
            "messages": chat_messages,
            "max_tokens": max_tokens or config.max_tokens,
        }
        if system_text:
            payload["system"] = system_text
        if temperature is not None:
            payload["temperature"] = temperature
        elif config.temperature:
            payload["temperature"] = config.temperature
        
        try:
            async with httpx.AsyncClient(timeout=config.max_tokens / 10 + 30) as client:
                resp = await client.post(
                    f"{config.api_base}/messages",
                    headers=headers,
                    json=payload,
                )
                
                latency_ms = (time.time() - start) * 1000
                
                if resp.status_code != 200:
                    return ModelResponse(
                        content="",
                        model=config.model_id,
                        provider=config.provider,
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=0,
                        latency_ms=latency_ms,
                        success=False,
                        error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                    )
                
                data = resp.json()
                # Anthropic returns content as array of blocks
                content = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        content += block.get("text", "")
                
                usage = data.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                
                cost = (
                    (input_tokens / 1_000_000) * config.cost_per_1m_input +
                    (output_tokens / 1_000_000) * config.cost_per_1m_output
                )
                
                return ModelResponse(
                    content=content,
                    model=config.model_id,
                    provider=config.provider,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                    latency_ms=latency_ms,
                    success=True,
                )
                
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return ModelResponse(
                content="",
                model=config.model_id,
                provider=config.provider,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )
    
    def _track_cost(self, result: ModelResponse):
        """Track cumulative costs and persist to daily tracking file."""
        self.total_cost += result.cost_usd
        self.call_count += 1
        self.cost_history.append({
            "model": result.model,
            "provider": result.provider,
            "cost": result.cost_usd,
            "tokens": result.input_tokens + result.output_tokens,
            "timestamp": time.time(),
        })
        
        # Persist daily costs
        self._persist_daily_cost(result.provider, result.cost_usd)
        
        if result.cost_usd > 0:
            logger.info(
                f"[COST] {result.provider}/{result.model}: "
                f"${result.cost_usd:.6f} ({result.input_tokens}+{result.output_tokens} tokens, "
                f"{result.latency_ms:.0f}ms) | Total: ${self.total_cost:.4f}"
            )
        else:
            logger.info(
                f"[FREE] {result.provider}/{result.model}: "
                f"{result.input_tokens}+{result.output_tokens} tokens, "
                f"{result.latency_ms:.0f}ms"
            )
    
    def _persist_daily_cost(self, provider: str, cost: float):
        """Write cost to daily tracking file for budget enforcement."""
        path = Path(self.budget.cost_tracking_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = json.loads(path.read_text()) if path.exists() else {}
        except Exception:
            data = {}
        
        today = time.strftime("%Y-%m-%d")
        if today not in data:
            data[today] = {}
        data[today][provider] = data[today].get(provider, 0.0) + cost
        data[today]["_total"] = sum(
            v for k, v in data[today].items() 
            if k != "_total" and isinstance(v, (int, float))
        )
        
        path.write_text(json.dumps(data, indent=2))
    
    def get_cost_summary(self) -> dict:
        """Get a summary of costs so far."""
        by_provider = {}
        for entry in self.cost_history:
            p = entry["provider"]
            if p not in by_provider:
                by_provider[p] = {"calls": 0, "cost": 0, "tokens": 0}
            by_provider[p]["calls"] += 1
            by_provider[p]["cost"] += entry["cost"]
            by_provider[p]["tokens"] += entry["tokens"]
        
        return {
            "total_cost_usd": self.total_cost,
            "total_calls": self.call_count,
            "by_provider": by_provider,
        }
    
    def get_budget_status(self) -> str:
        """Get a human-readable budget status."""
        today_spend = self.budget.load_today_spend()
        lines = ["📊 Daily Budget Status:"]
        total = 0.0
        for provider, cap in self.budget.provider_daily_caps.items():
            spent = today_spend.get(provider, 0.0)
            total += spent
            pct = (spent / cap * 100) if cap > 0 else 0
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            status = "🔴" if pct >= 100 else "🟡" if pct >= 75 else "🟢"
            lines.append(f"  {status} {provider:12s} ${spent:.4f} / ${cap:.2f} [{bar}] {pct:.0f}%")
        
        lines.append(f"\n  Total: ${total:.4f} / ${self.budget.daily_swarm_budget_usd:.2f}")
        return "\n".join(lines)

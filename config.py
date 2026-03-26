"""
Agent Swarm Configuration
========================
Central config for all model providers, agents, and cost controls.
API keys loaded from environment variables.

PROVIDERS (7):
  Groq       — FREE (Llama 8B + 70B)
  DO Gradient — ultra-cheap (GPT-oss, GPT-5 Nano, GPT-4o Mini, Llama, Qwen, etc.)
  DeepSeek   — cheap (V3 best code gen per dollar)
  Mistral    — cheap (Small, Codestral)
  OpenRouter — cheap (Gemini Flash, Qwen, hundreds of models)
  Anthropic  — mid (Haiku 3, Haiku 3.5)
  Viktor     — premium (Claude Sonnet via getviktor credits)
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a single model provider."""
    name: str
    provider: str
    model_id: str
    api_base: str
    api_key_env: str  # Name of env var holding the API key
    cost_per_1m_input: float  # USD per 1M input tokens
    cost_per_1m_output: float  # USD per 1M output tokens
    max_tokens: int = 4096
    temperature: float = 0.3
    
    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env)


@dataclass 
class AgentConfig:
    """Configuration for a single agent."""
    name: str
    role: str
    model_tier: str  # "free", "cheap", "smart"
    system_prompt: str
    max_retries: int = 2
    timeout_seconds: int = 60


@dataclass
class BudgetConfig:
    """Daily and monthly budget caps for cost control."""
    # Daily caps
    daily_swarm_budget_usd: float = 0.50      # Sub-agent daily cap
    daily_viktor_credits_target: int = 1250    # Target Viktor credits/day (1000-1500)
    
    # Monthly caps
    monthly_swarm_budget_usd: float = 10.0     # Total sub-agent budget
    
    # Per-provider daily caps (prevents one provider from eating the budget)
    provider_daily_caps: dict = field(default_factory=lambda: {
        "groq": 999.0,         # Unlimited — it's free
        "deepseek": 0.20,      # $0.20/day ($6/month)
        "do_gradient": 0.15,   # $0.15/day ($4.50/month)
        "mistral": 0.10,       # $0.10/day ($3/month)
        "openrouter": 0.10,    # $0.10/day ($3/month)
        "anthropic": 0.15,     # $0.15/day ($4.50/month)
    })
    
    # Cost tracking file
    cost_tracking_file: str = "/var/log/agent_swarm/costs.json"
    
    def load_today_spend(self) -> dict:
        """Load today's spend from tracking file."""
        path = Path(self.cost_tracking_file)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
            today = time.strftime("%Y-%m-%d")
            return data.get(today, {})
        except Exception:
            return {}
    
    def is_provider_over_budget(self, provider: str) -> bool:
        """Check if a provider has exceeded its daily cap."""
        today_spend = self.load_today_spend()
        spent = today_spend.get(provider, 0.0)
        cap = self.provider_daily_caps.get(provider, 0.10)
        return spent >= cap
    
    def is_over_daily_budget(self) -> bool:
        """Check if total daily swarm spend exceeds cap."""
        today_spend = self.load_today_spend()
        total = sum(v for v in today_spend.values() if isinstance(v, (int, float)))
        return total >= self.daily_swarm_budget_usd


@dataclass
class SwarmConfig:
    """Master configuration for the agent swarm."""
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    
    # Slack webhook for reporting (optional)
    slack_webhook_url: Optional[str] = None
    slack_bot_token: Optional[str] = None
    slack_channel: str = "#bot-building-for-optimization"
    
    # Log directory
    log_dir: str = "/var/log/agent_swarm"


# ============================================================
# MODEL REGISTRY - 7 providers, 22 models, sorted by cost tier
# ============================================================

MODELS = {
    # ─── TIER 1: FREE ───────────────────────────────────────
    "groq_llama70b": ModelConfig(
        name="Groq Llama 3.3 70B",
        provider="groq",
        model_id="llama-3.3-70b-versatile",
        api_base="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        max_tokens=4096,
        temperature=0.3,
    ),
    "groq_llama8b": ModelConfig(
        name="Groq Llama 3.1 8B",
        provider="groq",
        model_id="llama-3.1-8b-instant",
        api_base="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        max_tokens=4096,
        temperature=0.2,
    ),

    # ─── TIER 1.5: DO GRADIENT (Matthew's DO dashboard, 2026-03-24) ──
    "do_gpt_oss_20b": ModelConfig(
        name="DO Gradient GPT-oss-20b",
        provider="do_gradient",
        model_id="OpenAI GPT-oss-20b",
        api_base="https://cloud.digitalocean.com/gen-ai/platform/endpoints",
        api_key_env="DO_GRADIENT_API_KEY",
        cost_per_1m_input=0.05,
        cost_per_1m_output=0.45,
        max_tokens=4096,
        temperature=0.3,
    ),
    "do_gpt5_nano": ModelConfig(
        name="DO Gradient GPT-5 Nano",
        provider="do_gradient",
        model_id="OpenAI GPT-5 Nano",
        api_base="https://cloud.digitalocean.com/gen-ai/platform/endpoints",
        api_key_env="DO_GRADIENT_API_KEY",
        cost_per_1m_input=0.05,
        cost_per_1m_output=0.40,
        max_tokens=4096,
        temperature=0.3,
    ),
    "do_gpt4o_mini": ModelConfig(
        name="DO Gradient GPT-4o Mini",
        provider="do_gradient",
        model_id="OpenAI GPT-4o mini",
        api_base="https://cloud.digitalocean.com/gen-ai/platform/endpoints",
        api_key_env="DO_GRADIENT_API_KEY",
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
        max_tokens=4096,
        temperature=0.3,
    ),
    "do_llama8b": ModelConfig(
        name="DO Gradient Llama 3.1 8B",
        provider="do_gradient",
        model_id="Llama 3.1 Instruct (8B)",
        api_base="https://cloud.digitalocean.com/gen-ai/platform/endpoints",
        api_key_env="DO_GRADIENT_API_KEY",
        cost_per_1m_input=0.20,
        cost_per_1m_output=0.20,
        max_tokens=4096,
        temperature=0.3,
    ),
    "do_llama70b": ModelConfig(
        name="DO Gradient Llama 3.3 70B",
        provider="do_gradient",
        model_id="Llama 3.3 Instruct (70B)",
        api_base="https://cloud.digitalocean.com/gen-ai/platform/endpoints",
        api_key_env="DO_GRADIENT_API_KEY",
        cost_per_1m_input=0.65,
        cost_per_1m_output=0.65,
        max_tokens=4096,
        temperature=0.3,
    ),
    "do_qwen3_32b": ModelConfig(
        name="DO Gradient Qwen3 32B",
        provider="do_gradient",
        model_id="Qwen3 32B",
        api_base="https://cloud.digitalocean.com/gen-ai/platform/endpoints",
        api_key_env="DO_GRADIENT_API_KEY",
        cost_per_1m_input=0.25,
        cost_per_1m_output=0.55,
        max_tokens=4096,
        temperature=0.3,
    ),
    "do_gpt5_mini": ModelConfig(
        name="DO Gradient GPT-5 Mini",
        provider="do_gradient",
        model_id="OpenAI GPT-5 Mini",
        api_base="https://cloud.digitalocean.com/gen-ai/platform/endpoints",
        api_key_env="DO_GRADIENT_API_KEY",
        cost_per_1m_input=0.25,
        cost_per_1m_output=2.00,
        max_tokens=4096,
        temperature=0.3,
    ),
    "do_deepseek_r1": ModelConfig(
        name="DO Gradient DeepSeek R1 Distill 70B",
        provider="do_gradient",
        model_id="DeepSeek R1 Distill Llama 70B",
        api_base="https://cloud.digitalocean.com/gen-ai/platform/endpoints",
        api_key_env="DO_GRADIENT_API_KEY",
        cost_per_1m_input=0.99,
        cost_per_1m_output=0.99,
        max_tokens=4096,
        temperature=0.3,
    ),

    # ─── TIER 2: CHEAP — DeepSeek ───────────────────────────
    "deepseek_v3": ModelConfig(
        name="DeepSeek V3",
        provider="deepseek",
        model_id="deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        cost_per_1m_input=0.14,
        cost_per_1m_output=0.28,
        max_tokens=8192,
        temperature=0.2,
    ),
    "deepseek_coder": ModelConfig(
        name="DeepSeek Coder V3",
        provider="deepseek",
        model_id="deepseek-coder",
        api_base="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        cost_per_1m_input=0.14,
        cost_per_1m_output=0.28,
        max_tokens=8192,
        temperature=0.1,
    ),

    # ─── TIER 2: CHEAP — Mistral ────────────────────────────
    "mistral_small": ModelConfig(
        name="Mistral Small",
        provider="mistral",
        model_id="mistral-small-latest",
        api_base="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        cost_per_1m_input=0.10,
        cost_per_1m_output=0.30,
        max_tokens=4096,
        temperature=0.3,
    ),
    "mistral_codestral": ModelConfig(
        name="Mistral Codestral",
        provider="mistral",
        model_id="codestral-latest",
        api_base="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        cost_per_1m_input=0.30,
        cost_per_1m_output=0.90,
        max_tokens=8192,
        temperature=0.1,
    ),

    # ─── TIER 2: CHEAP — OpenRouter (gateway to 100+ models) ─
    "or_gemini_flash": ModelConfig(
        name="OpenRouter Gemini 2.0 Flash",
        provider="openrouter",
        model_id="google/gemini-2.0-flash-001",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        cost_per_1m_input=0.10,
        cost_per_1m_output=0.40,
        max_tokens=8192,
        temperature=0.3,
    ),
    "or_gemini_flash_lite": ModelConfig(
        name="OpenRouter Gemini 1.5 Flash",
        provider="openrouter",
        model_id="google/gemini-flash-1.5",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        cost_per_1m_input=0.075,
        cost_per_1m_output=0.30,
        max_tokens=8192,
        temperature=0.3,
    ),
    "or_qwen_72b": ModelConfig(
        name="OpenRouter Qwen 2.5 72B",
        provider="openrouter",
        model_id="qwen/qwen-2.5-72b-instruct",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        cost_per_1m_input=0.35,
        cost_per_1m_output=0.40,
        max_tokens=4096,
        temperature=0.3,
    ),

    # ─── TIER 3: MID — Anthropic ────────────────────────────
    "anthropic_haiku3": ModelConfig(
        name="Anthropic Claude 3 Haiku",
        provider="anthropic",
        model_id="claude-3-haiku-20240307",
        api_base="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        cost_per_1m_input=0.25,
        cost_per_1m_output=1.25,
        max_tokens=4096,
        temperature=0.3,
    ),
    "anthropic_haiku35": ModelConfig(
        name="Anthropic Claude 3.5 Haiku",
        provider="anthropic",
        model_id="claude-3-5-haiku-20241022",
        api_base="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        cost_per_1m_input=0.80,
        cost_per_1m_output=4.00,
        max_tokens=4096,
        temperature=0.3,
    ),
}


# ============================================================
# TIER MAPPING — fallback chains for each budget level
# ============================================================
TIER_MODELS = {
    "free":          ["groq_llama70b", "groq_llama8b"],
    "ultra_cheap":   ["do_gpt_oss_20b", "do_gpt5_nano", "or_gemini_flash_lite", "mistral_small"],
    "cheap":         ["deepseek_v3", "or_gemini_flash", "do_gpt4o_mini", "mistral_small"],
    "cheap_code":    ["deepseek_v3", "deepseek_coder", "mistral_codestral", "or_gemini_flash"],
    "mid":           ["do_qwen3_32b", "or_qwen_72b", "anthropic_haiku3", "do_gpt5_mini"],
    "mid_reasoning": ["do_deepseek_r1", "anthropic_haiku35"],
    "smart":         ["anthropic_haiku35", "do_deepseek_r1"],
}


# ============================================================
# ROLE → MODEL MAPPING — which model each agent type uses
# ============================================================
ROLE_MODEL_MAP = {
    "monitor":       "groq_llama8b",         # FREE — log parsing, health checks
    "code_writer":   "deepseek_v3",          # $0.14/M — best code gen per dollar
    "code_reviewer": "groq_llama70b",        # FREE — solid reasoning
    "general":       "or_gemini_flash_lite",  # $0.075/M — cheapest smart model
    "smart":         "or_gemini_flash",       # $0.10/M — great quality/price
    "code_alt":      "mistral_codestral",    # $0.30/M — code specialist backup
    "reasoning":     "do_deepseek_r1",       # $0.99/M — complex logic
    "quality_check": "anthropic_haiku3",     # $0.25/M — Claude taste for QA
}


# ============================================================
# OPTIMIZATION STRATEGY — how to minimize Viktor credits
# ============================================================
#
# Viktor credits: ~$2.50 per 1,000 credits
# Target: 1,000-1,500 credits/day = $2.50-$3.75/day of Viktor usage
#
# STRATEGY:
# 1. Sub-agents handle ALL grunt work (code, monitoring, review)
# 2. Viktor only does: task planning, final QA, Slack interaction
# 3. Daily budget caps prevent runaway sub-agent costs
# 4. Free tier (Groq) handles ~60% of sub-agent calls
# 5. Cheap tier handles ~35% (DeepSeek, Mistral, Gemini, DO Gradient)
# 6. Mid tier (Anthropic Haiku) only for quality-critical checks ~5%
#
# Estimated daily spend at target usage:
#   Viktor credits:     ~1,250/day ($3.13)
#   Sub-agent APIs:     ~$0.30-0.50/day
#   ─────────────────────────────
#   Total:              ~$3.50-3.75/day
#   Monthly:            ~$105-115/month (vs ~$250+ without swarm)

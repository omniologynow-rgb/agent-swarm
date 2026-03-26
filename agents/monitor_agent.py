"""
Monitor Agent
=============
Watches application logs, parses errors, and generates alerts.
Uses Groq Llama 8B (FREE, ultra-fast) — perfect for high-volume log analysis.
"""

from agents.base import BaseAgent, TaskResult
from models import ModelRouter
from typing import Optional
import re

MONITOR_PROMPT = """You are a DevOps monitoring sub-agent. Your job is to analyze application logs and identify issues.

YOUR RESPONSIBILITIES:
1. Parse log entries and identify errors, warnings, and anomalies
2. Categorize issues by severity: CRITICAL, WARNING, INFO
3. Provide brief, actionable summaries
4. Suggest fixes when the cause is obvious

OUTPUT FORMAT (strict JSON):
{
    "severity": "CRITICAL|WARNING|INFO",
    "summary": "Brief one-line description",
    "details": "What happened and potential root cause",
    "action": "Suggested next step",
    "affected_service": "frontend|backend|database|nginx|unknown"
}

RULES:
- Be concise — no fluff
- If logs look normal, say so: {"severity": "INFO", "summary": "No issues detected", ...}
- For stack traces, identify the root cause line
- For repeated errors, note the frequency
- Don't hallucinate issues that aren't in the logs
"""


class MonitorAgent(BaseAgent):
    """Agent specialized in log analysis and monitoring using Groq (free, fast)."""
    
    def __init__(self, router: Optional[ModelRouter] = None):
        super().__init__(
            name="Monitor",
            role="Analyzes logs, detects errors, and generates alerts",
            system_prompt=MONITOR_PROMPT,
            model_key="groq_llama8b",  # Free tier, ultra-fast
            fallback_tier="free",
            router=router,
        )
    
    async def analyze_logs(self, log_content: str, service: str = "app") -> TaskResult:
        """Analyze a batch of log lines."""
        task = f"Analyze these {service} logs and report any issues:"
        return await self.run(task, context=log_content)
    
    async def triage_error(self, error_text: str, stack_trace: str = "") -> TaskResult:
        """Triage a specific error."""
        context = error_text
        if stack_trace:
            context += f"\n\nSTACK TRACE:\n{stack_trace}"
        task = "Triage this error — what's the severity and what should we do?"
        return await self.run(task, context=context)
    
    async def check_health(self, metrics: dict) -> TaskResult:
        """Check application health from metrics."""
        import json
        task = "Review these application metrics and flag any concerns:"
        return await self.run(task, context=json.dumps(metrics, indent=2))
    
    @staticmethod
    def extract_errors_from_logs(log_content: str) -> list[str]:
        """Pre-filter: extract error-like lines before sending to LLM (saves tokens)."""
        error_patterns = [
            r'(?i)\b(error|exception|traceback|fatal|critical|failed|panic)\b',
            r'(?i)status[_\s]?code[:\s]+[45]\d{2}',
            r'(?i)(connection refused|timeout|oom|out of memory)',
        ]
        
        lines = log_content.split('\n')
        error_lines = []
        
        for i, line in enumerate(lines):
            for pattern in error_patterns:
                if re.search(pattern, line):
                    # Include surrounding context (2 lines before/after)
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    context_block = '\n'.join(lines[start:end])
                    if context_block not in error_lines:
                        error_lines.append(context_block)
                    break
        
        return error_lines

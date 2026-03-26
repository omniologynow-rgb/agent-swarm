"""
Agent Swarm Orchestrator
========================
Coordinates sub-agents, manages task queues, and routes work
to the cheapest appropriate model.
"""

import asyncio
import json
import logging
import time
from typing import Optional
from dataclasses import dataclass, field

from models import ModelRouter
from agents.base import TaskResult
from agents.code_agent import CodeWriterAgent
from agents.monitor_agent import MonitorAgent
from agents.review_agent import ReviewAgent

logger = logging.getLogger("agent_swarm.orchestrator")


@dataclass
class SwarmTask:
    """A task in the queue."""
    id: str
    agent_type: str  # "code_writer", "monitor", "reviewer"
    task: str
    context: str = ""
    priority: int = 1  # 1=normal, 2=high, 3=critical
    created_at: float = field(default_factory=time.time)
    result: Optional[TaskResult] = None


class Orchestrator:
    """
    The brain of the agent swarm.
    
    Manages agent instances, routes tasks, tracks costs,
    and provides a simple API for external use.
    """
    
    def __init__(self, monthly_budget: float = 10.0):
        self.router = ModelRouter()
        self.monthly_budget = monthly_budget
        
        # Initialize agents with shared router (for unified cost tracking)
        self.agents = {
            "code_writer": CodeWriterAgent(router=self.router),
            "monitor": MonitorAgent(router=self.router),
            "reviewer": ReviewAgent(router=self.router),
        }
        
        self.task_history: list[SwarmTask] = []
        self.started_at = time.time()
        
        logger.info(f"Orchestrator initialized with budget=${monthly_budget}/mo")
        logger.info(f"Agents: {', '.join(self.agents.keys())}")
    
    # ================================================================
    # HIGH-LEVEL TASK METHODS
    # ================================================================
    
    async def write_code(self, description: str, existing_code: str = "") -> TaskResult:
        """Have the code writer generate code."""
        self._check_budget()
        agent = self.agents["code_writer"]
        result = await agent.run(description, context=existing_code)
        self._record_task("code_writer", description, result)
        return result
    
    async def write_component(self, description: str, existing_code: str = "") -> TaskResult:
        """Write a React component."""
        self._check_budget()
        return await self.agents["code_writer"].write_component(description, existing_code)
    
    async def write_api_route(self, description: str, existing_code: str = "") -> TaskResult:
        """Write a FastAPI route."""
        self._check_budget()
        return await self.agents["code_writer"].write_api_route(description, existing_code)
    
    async def review_code(self, code: str, filename: str = "") -> TaskResult:
        """Have the reviewer check code quality."""
        self._check_budget()
        agent = self.agents["reviewer"]
        result = await agent.review_file(code, filename)
        self._record_task("reviewer", f"Review {filename}", result)
        return result
    
    async def review_diff(self, diff: str, context: str = "") -> TaskResult:
        """Review a git diff."""
        self._check_budget()
        return await self.agents["reviewer"].review_diff(diff, context)
    
    async def analyze_logs(self, logs: str, service: str = "app") -> TaskResult:
        """Have the monitor analyze logs."""
        self._check_budget()
        
        # Pre-filter to save tokens (only send error lines to LLM)
        error_lines = MonitorAgent.extract_errors_from_logs(logs)
        if not error_lines:
            return TaskResult(
                agent_name="Monitor",
                task="Log analysis",
                output='{"severity": "INFO", "summary": "No errors detected in logs", "details": "All log entries appear normal", "action": "No action needed", "affected_service": "none"}',
                model_used="pre-filter",
                cost_usd=0,
                latency_ms=0,
                success=True,
            )
        
        filtered_logs = "\n---\n".join(error_lines)
        agent = self.agents["monitor"]
        result = await agent.analyze_logs(filtered_logs, service)
        self._record_task("monitor", f"Analyze {service} logs", result)
        return result
    
    async def fix_bug(self, description: str, code: str) -> TaskResult:
        """Fix a bug — writes fix then auto-reviews it."""
        self._check_budget()
        
        # Step 1: Code agent fixes the bug
        fix_result = await self.agents["code_writer"].fix_bug(description, code)
        if not fix_result.success:
            return fix_result
        
        # Step 2: Review agent checks the fix (free model, why not)
        review_result = await self.agents["reviewer"].review_file(
            fix_result.output, 
            filename="bug_fix"
        )
        
        # Combine results
        combined_output = (
            f"=== BUG FIX ===\n{fix_result.output}\n\n"
            f"=== AUTO-REVIEW ===\n{review_result.output}"
        )
        
        return TaskResult(
            agent_name="CodeWriter+Reviewer",
            task=f"Fix: {description[:100]}",
            output=combined_output,
            model_used=f"{fix_result.model_used} + {review_result.model_used}",
            cost_usd=fix_result.cost_usd + review_result.cost_usd,
            latency_ms=fix_result.latency_ms + review_result.latency_ms,
            success=True,
            metadata={
                "fix_model": fix_result.model_used,
                "review_model": review_result.model_used,
                "review_verdict": "APPROVE" if "APPROVE" in review_result.output else "NEEDS_REVIEW",
            }
        )
    
    # ================================================================
    # PIPELINE: Write → Review → Report
    # ================================================================
    
    async def build_feature(self, description: str, existing_code: str = "") -> dict:
        """
        Full pipeline: Code agent writes it, review agent checks it.
        Returns both results so Viktor/human can make final call.
        """
        self._check_budget()
        
        logger.info(f"[PIPELINE] Building feature: {description[:80]}")
        
        # Phase 1: Write
        write_result = await self.agents["code_writer"].run(
            f"Build this feature: {description}",
            context=existing_code,
        )
        
        if not write_result.success:
            return {"phase": "write", "error": write_result.error, "success": False}
        
        # Phase 2: Review
        review_result = await self.agents["reviewer"].review_file(
            write_result.output,
            filename="new_feature",
        )
        
        total_cost = write_result.cost_usd + review_result.cost_usd
        
        return {
            "success": True,
            "code": write_result.output,
            "review": review_result.output,
            "write_model": write_result.model_used,
            "review_model": review_result.model_used,
            "total_cost_usd": total_cost,
            "total_latency_ms": write_result.latency_ms + review_result.latency_ms,
        }
    
    # ================================================================
    # COST & BUDGET
    # ================================================================
    
    def _check_budget(self):
        """Raise if we've exceeded monthly budget."""
        if self.router.total_cost >= self.monthly_budget:
            raise BudgetExceededError(
                f"Monthly budget of ${self.monthly_budget} exceeded "
                f"(spent: ${self.router.total_cost:.4f})"
            )
    
    def get_cost_report(self) -> str:
        """Get a formatted cost report."""
        summary = self.router.get_cost_summary()
        uptime_hours = (time.time() - self.started_at) / 3600
        
        lines = [
            "╔══════════════════════════════════════╗",
            "║     AGENT SWARM COST REPORT          ║",
            "╠══════════════════════════════════════╣",
            f"║ Total Spent:  ${summary['total_cost_usd']:.6f}",
            f"║ Budget Left:  ${self.monthly_budget - summary['total_cost_usd']:.4f}",
            f"║ Total Calls:  {summary['total_calls']}",
            f"║ Uptime:       {uptime_hours:.1f} hours",
            "╠══════════════════════════════════════╣",
            "║ BY PROVIDER:",
        ]
        
        for provider, stats in summary["by_provider"].items():
            lines.append(
                f"║  {provider}: {stats['calls']} calls, "
                f"${stats['cost']:.6f}, {stats['tokens']} tokens"
            )
        
        lines.append("╠══════════════════════════════════════╣")
        lines.append("║ BY AGENT:")
        
        for name, agent in self.agents.items():
            stats = agent.get_stats()
            lines.append(
                f"║  {stats['name']}: {stats['tasks_completed']} tasks, "
                f"${stats['total_cost_usd']:.6f}"
            )
        
        lines.append("╚══════════════════════════════════════╝")
        
        return "\n".join(lines)
    
    # ================================================================
    # INTERNALS
    # ================================================================
    
    def _record_task(self, agent_type: str, task: str, result: TaskResult):
        """Record a completed task."""
        self.task_history.append(SwarmTask(
            id=f"task_{len(self.task_history) + 1}",
            agent_type=agent_type,
            task=task,
            result=result,
        ))


class BudgetExceededError(Exception):
    """Raised when the monthly budget has been exceeded."""
    pass

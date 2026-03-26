"""
Base Agent Class
================
All sub-agents inherit from this. Handles conversation history,
system prompts, and interaction with the model router.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from models import ModelRouter, ModelResponse

logger = logging.getLogger("agent_swarm.agent")


@dataclass
class TaskResult:
    """Result from an agent completing a task."""
    agent_name: str
    task: str
    output: str
    model_used: str
    cost_usd: float
    latency_ms: float
    success: bool
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class BaseAgent:
    """
    Base class for all sub-agents in the swarm.
    
    Each agent has:
    - A name and role description
    - A system prompt defining its behavior
    - A preferred model (cheap or free)
    - Conversation memory (within a session)
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        model_key: str,
        fallback_tier: str = "free",
        router: Optional[ModelRouter] = None,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model_key = model_key
        self.fallback_tier = fallback_tier
        self.router = router or ModelRouter()
        self.conversation_history: list[dict] = []
        self.task_count = 0
        self.total_cost = 0.0
        
        logger.info(f"[{self.name}] Initialized ({self.role}) using model={self.model_key}")
    
    async def run(self, task: str, context: str = "", clear_history: bool = True) -> TaskResult:
        """
        Execute a task with this agent.
        
        Args:
            task: The task description / user message
            context: Optional additional context (file contents, logs, etc.)
            clear_history: If True, start fresh conversation (default for stateless tasks)
        """
        if clear_history:
            self.conversation_history = []
        
        # Build the message
        user_content = task
        if context:
            user_content = f"{task}\n\n---\nCONTEXT:\n{context}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history,
            {"role": "user", "content": user_content},
        ]
        
        start = time.time()
        response = await self.router.chat(
            messages=messages,
            model_key=self.model_key,
            fallback_tier=self.fallback_tier,
        )
        
        if response.success:
            # Add to conversation history for multi-turn
            self.conversation_history.append({"role": "user", "content": user_content})
            self.conversation_history.append({"role": "assistant", "content": response.content})
            self.task_count += 1
            self.total_cost += response.cost_usd
        
        return TaskResult(
            agent_name=self.name,
            task=task[:200],
            output=response.content,
            model_used=response.model,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
            success=response.success,
            error=response.error,
        )
    
    async def multi_turn(self, messages_sequence: list[str]) -> list[TaskResult]:
        """
        Run a multi-turn conversation (e.g., iterative refinement).
        First message starts fresh, subsequent ones continue the conversation.
        """
        results = []
        for i, msg in enumerate(messages_sequence):
            result = await self.run(msg, clear_history=(i == 0))
            results.append(result)
            if not result.success:
                break
        return results
    
    def get_stats(self) -> dict:
        """Get agent performance stats."""
        return {
            "name": self.name,
            "role": self.role,
            "model": self.model_key,
            "tasks_completed": self.task_count,
            "total_cost_usd": self.total_cost,
        }
    
    def __repr__(self):
        return f"<Agent: {self.name} ({self.role}) model={self.model_key}>"

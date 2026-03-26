"""
Code Writer Agent
=================
Specialized agent for writing code — components, API routes, utilities.
Uses DeepSeek V3 (cheapest, best code model) with Groq fallback.
"""

from agents.base import BaseAgent, TaskResult
from models import ModelRouter
from typing import Optional

CODE_WRITER_PROMPT = """You are a senior full-stack developer sub-agent. Your job is to write clean, production-ready code.

TECH STACK YOU WORK WITH:
- Frontend: React (JavaScript/JSX), CSS, HTML
- Backend: Python (FastAPI), MongoDB
- Infrastructure: Docker, Docker Compose, Nginx
- APIs: RESTful, OpenAI-compatible chat completions

RULES:
1. Write complete, working code — no placeholders or TODOs
2. Include error handling and edge cases
3. Follow the existing codebase patterns when context is provided
4. Use clear variable names and add brief comments for complex logic
5. If asked to modify existing code, show the full modified function/component
6. Always consider security (input validation, no hardcoded secrets)
7. Keep responses focused — code first, brief explanation after

OUTPUT FORMAT:
- Start with the code in a fenced code block with language tag
- Follow with a brief summary of what it does (2-3 sentences max)
- If there are dependencies to install, list them at the end
"""


class CodeWriterAgent(BaseAgent):
    """Agent specialized in writing code using DeepSeek (best code quality per dollar)."""
    
    def __init__(self, router: Optional[ModelRouter] = None):
        super().__init__(
            name="CodeWriter",
            role="Writes production-ready code for React frontend and FastAPI backend",
            system_prompt=CODE_WRITER_PROMPT,
            model_key="deepseek_v3",  # Best code gen model, $0.14/M input
            fallback_tier="free",     # Falls back to Groq Llama if DeepSeek is down
            router=router,
        )
    
    async def write_component(self, description: str, existing_code: str = "") -> TaskResult:
        """Write a React component."""
        task = f"Write a React component: {description}"
        return await self.run(task, context=existing_code)
    
    async def write_api_route(self, description: str, existing_code: str = "") -> TaskResult:
        """Write a FastAPI route."""
        task = f"Write a FastAPI route: {description}"
        return await self.run(task, context=existing_code)
    
    async def write_utility(self, description: str, language: str = "python") -> TaskResult:
        """Write a utility function."""
        task = f"Write a {language} utility function: {description}"
        return await self.run(task)
    
    async def fix_bug(self, bug_description: str, code: str) -> TaskResult:
        """Fix a bug in existing code."""
        task = f"Fix this bug: {bug_description}\n\nReturn the corrected code."
        return await self.run(task, context=code)
    
    async def refactor(self, code: str, instructions: str = "Improve code quality") -> TaskResult:
        """Refactor existing code."""
        task = f"Refactor this code. Instructions: {instructions}"
        return await self.run(task, context=code)

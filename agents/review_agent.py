"""
Code Review Agent
=================
Reviews code diffs, checks for bugs, security issues, and style.
Uses Groq Llama 70B (FREE, good reasoning) — perfect for code review.
"""

from agents.base import BaseAgent, TaskResult
from models import ModelRouter
from typing import Optional

REVIEW_PROMPT = """You are a senior code reviewer sub-agent. Your job is to review code changes and catch issues before they hit production.

REVIEW CHECKLIST:
1. **Bugs**: Logic errors, off-by-one, null/undefined handling, race conditions
2. **Security**: SQL injection, XSS, hardcoded secrets, unvalidated input
3. **Performance**: N+1 queries, unnecessary re-renders, memory leaks, missing indexes
4. **Style**: Consistency with codebase patterns, naming conventions, dead code
5. **Edge Cases**: Empty states, error handling, boundary conditions

OUTPUT FORMAT:
```
VERDICT: APPROVE | REQUEST_CHANGES | NEEDS_DISCUSSION

ISSUES FOUND:
- [CRITICAL] Description... (file:line if applicable)
- [WARNING] Description...
- [SUGGESTION] Description...

SUMMARY: One paragraph overall assessment.
```

RULES:
- Be constructive, not nitpicky
- Focus on bugs and security first, style last
- If code looks good, say APPROVE with a brief positive note
- Don't rewrite the code unless asked — just point out issues
- Consider the context of the full file when reviewing a diff
"""


class ReviewAgent(BaseAgent):
    """Agent specialized in code review using Groq Llama 70B (free, solid reasoning)."""
    
    def __init__(self, router: Optional[ModelRouter] = None):
        super().__init__(
            name="Reviewer",
            role="Reviews code for bugs, security issues, and quality",
            system_prompt=REVIEW_PROMPT,
            model_key="groq_llama70b",  # Free tier, good reasoning
            fallback_tier="free",
            router=router,
        )
    
    async def review_diff(self, diff: str, context: str = "") -> TaskResult:
        """Review a git diff."""
        task = "Review this code diff and flag any issues:"
        full_context = diff
        if context:
            full_context = f"EXISTING FILE CONTEXT:\n{context}\n\nDIFF TO REVIEW:\n{diff}"
        return await self.run(task, context=full_context)
    
    async def review_file(self, code: str, filename: str = "") -> TaskResult:
        """Review an entire file."""
        task = f"Review this {'file (' + filename + ')' if filename else 'code'} for issues:"
        return await self.run(task, context=code)
    
    async def security_audit(self, code: str) -> TaskResult:
        """Focus specifically on security issues."""
        task = ("Perform a security audit of this code. Focus on: "
                "injection vulnerabilities, auth issues, data exposure, "
                "hardcoded secrets, and input validation.")
        return await self.run(task, context=code)
    
    async def check_pr(self, pr_title: str, pr_body: str, diff: str) -> TaskResult:
        """Review a full pull request."""
        context = f"PR: {pr_title}\n\n{pr_body}\n\nDIFF:\n{diff}"
        task = "Review this pull request — check the code diff against the stated purpose."
        return await self.run(task, context=context)

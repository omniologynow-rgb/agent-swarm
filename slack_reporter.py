"""
Slack Reporter
==============
Sends agent swarm results and alerts to Slack via webhook.
Lightweight — no heavy SDK dependencies.
"""

import httpx
import json
import logging
from typing import Optional
from agents.base import TaskResult

logger = logging.getLogger("agent_swarm.slack")


class SlackReporter:
    """Reports agent results to Slack."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        if not webhook_url:
            logger.warning("No Slack webhook configured — reporting disabled")
    
    async def send_message(self, text: str, blocks: list = None):
        """Send a message to Slack."""
        if not self.webhook_url:
            logger.info(f"[SLACK-MOCK] {text}")
            return
        
        payload = {"text": text}
        if blocks:
            payload["blocks"] = blocks
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(self.webhook_url, json=payload)
                if resp.status_code != 200:
                    logger.error(f"Slack send failed: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.error(f"Slack send error: {e}")
    
    async def report_task_result(self, result: TaskResult):
        """Send a formatted task result to Slack."""
        emoji = "✅" if result.success else "❌"
        cost_str = f"${result.cost_usd:.6f}" if result.cost_usd > 0 else "FREE"
        
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{emoji} *{result.agent_name}* completed task\n"
                        f">_{result.task[:100]}_\n"
                        f"Model: `{result.model_used}` | "
                        f"Cost: *{cost_str}* | "
                        f"Latency: {result.latency_ms:.0f}ms"
                    )
                }
            }
        ]
        
        # Add output preview (first 500 chars)
        if result.output:
            preview = result.output[:500]
            if len(result.output) > 500:
                preview += "\n... (truncated)"
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"```{preview}```"}
            })
        
        await self.send_message(
            text=f"{result.agent_name}: {result.task[:80]}",
            blocks=blocks,
        )
    
    async def report_error_alert(self, severity: str, summary: str, details: str):
        """Send an error alert to Slack."""
        emoji_map = {"CRITICAL": "🚨", "WARNING": "⚠️", "INFO": "ℹ️"}
        emoji = emoji_map.get(severity, "📋")
        
        await self.send_message(
            text=f"{emoji} [{severity}] {summary}",
            blocks=[{
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{emoji} *[{severity}]* {summary}\n\n{details}"
                }
            }]
        )
    
    async def report_cost_summary(self, report: str):
        """Send a cost report to Slack."""
        await self.send_message(
            text="Agent Swarm Cost Report",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"```{report}```"}
            }]
        )

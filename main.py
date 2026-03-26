#!/usr/bin/env python3
"""
Agent Swarm — Main Entry Point
================================
Lightweight multi-agent system using cheap/free LLMs.

Usage:
    # Interactive mode
    python main.py
    
    # Run a specific task
    python main.py --task "write a React loading spinner component"
    python main.py --task "review this file" --file path/to/file.py
    python main.py --task "analyze logs" --file /var/log/app.log
    
    # Cost report
    python main.py --report

Environment Variables Required:
    GROQ_API_KEY           - Free tier API key from console.groq.com
    DEEPSEEK_API_KEY       - API key from platform.deepseek.com  
    DO_GRADIENT_API_KEY    - DigitalOcean API token (for Gradient AI)

At minimum, set GROQ_API_KEY for free-tier operation.
"""

import asyncio
import argparse
import json
import logging
import os
import sys

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import Orchestrator, BudgetExceededError
from slack_reporter import SlackReporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent_swarm")


async def interactive_mode(orch: Orchestrator, reporter: SlackReporter):
    """Interactive REPL for the agent swarm."""
    print("\n" + "="*60)
    print("  🐝  AGENT SWARM — Interactive Mode")
    print("="*60)
    print("\nCommands:")
    print("  code <description>    — Write code (DeepSeek)")
    print("  review <file_path>    — Review a file (Groq Llama 70B)")
    print("  logs <file_path>      — Analyze logs (Groq Llama 8B)")
    print("  fix <description>     — Fix a bug (DeepSeek + review)")
    print("  build <description>   — Full pipeline: write + review")
    print("  cost                  — Show cost report")
    print("  quit                  — Exit")
    print("-"*60 + "\n")
    
    while True:
        try:
            user_input = input("🐝 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if not user_input:
            continue
        
        parts = user_input.split(" ", 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        try:
            if cmd == "quit" or cmd == "exit":
                break
            
            elif cmd == "cost":
                print(orch.get_cost_report())
            
            elif cmd == "code":
                if not arg:
                    print("Usage: code <description>")
                    continue
                result = await orch.write_code(arg)
                print(f"\n{'='*40}\n{result.output}\n{'='*40}")
                print(f"Model: {result.model_used} | Cost: ${result.cost_usd:.6f} | {result.latency_ms:.0f}ms")
            
            elif cmd == "review":
                if not arg or not os.path.exists(arg):
                    print("Usage: review <file_path>")
                    continue
                with open(arg) as f:
                    code = f.read()
                result = await orch.review_code(code, filename=arg)
                print(f"\n{'='*40}\n{result.output}\n{'='*40}")
                print(f"Model: {result.model_used} | Cost: ${result.cost_usd:.6f}")
            
            elif cmd == "logs":
                if not arg or not os.path.exists(arg):
                    print("Usage: logs <file_path>")
                    continue
                with open(arg) as f:
                    logs = f.read()
                result = await orch.analyze_logs(logs)
                print(f"\n{'='*40}\n{result.output}\n{'='*40}")
                print(f"Model: {result.model_used} | Cost: ${result.cost_usd:.6f}")
            
            elif cmd == "fix":
                if not arg:
                    print("Usage: fix <bug_description> (will prompt for file)")
                    continue
                file_path = input("File path: ").strip()
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    continue
                with open(file_path) as f:
                    code = f.read()
                result = await orch.fix_bug(arg, code)
                print(f"\n{'='*40}\n{result.output}\n{'='*40}")
                print(f"Model: {result.model_used} | Cost: ${result.cost_usd:.6f}")
            
            elif cmd == "build":
                if not arg:
                    print("Usage: build <feature_description>")
                    continue
                result = await orch.build_feature(arg)
                if result["success"]:
                    print(f"\n{'='*40}")
                    print(f"CODE ({result['write_model']}):")
                    print(result["code"][:2000])
                    print(f"\nREVIEW ({result['review_model']}):")
                    print(result["review"])
                    print(f"{'='*40}")
                    print(f"Total cost: ${result['total_cost_usd']:.6f}")
                else:
                    print(f"Failed at phase: {result['phase']} — {result['error']}")
            
            else:
                print(f"Unknown command: {cmd}")
                print("Try: code, review, logs, fix, build, cost, quit")
        
        except BudgetExceededError as e:
            print(f"⚠️ {e}")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
    
    # Print final cost report
    print("\n" + orch.get_cost_report())


async def run_task(orch: Orchestrator, task: str, file_path: str = None):
    """Run a single task from CLI args."""
    context = ""
    if file_path and os.path.exists(file_path):
        with open(file_path) as f:
            context = f.read()
    
    # Auto-detect task type
    task_lower = task.lower()
    
    if any(kw in task_lower for kw in ["review", "check", "audit"]):
        if context:
            result = await orch.review_code(context, filename=file_path or "")
        else:
            result = await orch.write_code(task)  # Fallback
    elif any(kw in task_lower for kw in ["log", "monitor", "error"]):
        if context:
            result = await orch.analyze_logs(context)
        else:
            result = await orch.write_code(task)
    elif any(kw in task_lower for kw in ["fix", "bug", "debug"]):
        if context:
            result = await orch.fix_bug(task, context)
        else:
            result = await orch.write_code(task)
    else:
        if context:
            result = await orch.write_code(task, existing_code=context)
        else:
            result = await orch.write_code(task)
    
    # Output
    print(json.dumps({
        "agent": result.agent_name,
        "model": result.model_used,
        "cost_usd": result.cost_usd,
        "latency_ms": result.latency_ms,
        "success": result.success,
        "output": result.output,
        "error": result.error,
    }, indent=2))
    
    return result


def check_api_keys():
    """Check which API keys are available."""
    keys = {
        "GROQ_API_KEY": "Groq (FREE tier — Llama 3)",
        "DEEPSEEK_API_KEY": "DeepSeek (cheap code gen)",
        "DO_GRADIENT_API_KEY": "DigitalOcean Gradient AI",
    }
    
    print("\n📋 API Key Status:")
    any_configured = False
    for env_var, desc in keys.items():
        status = "✅" if os.environ.get(env_var) else "❌"
        if os.environ.get(env_var):
            any_configured = True
        print(f"  {status} {env_var} — {desc}")
    
    if not any_configured:
        print("\n⚠️  No API keys configured! Set at least GROQ_API_KEY for free-tier usage.")
        print("   export GROQ_API_KEY='your-key-here'")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Agent Swarm — Cheap AI Sub-Agents")
    parser.add_argument("--task", "-t", help="Task to execute")
    parser.add_argument("--file", "-f", help="File path for context")
    parser.add_argument("--budget", "-b", type=float, default=10.0, help="Monthly budget in USD")
    parser.add_argument("--report", "-r", action="store_true", help="Show cost report")
    parser.add_argument("--check", action="store_true", help="Check API key status")
    args = parser.parse_args()
    
    if args.check:
        check_api_keys()
        return
    
    if not check_api_keys():
        print("\nContinuing anyway (will fail on API calls)...\n")
    
    orch = Orchestrator(monthly_budget=args.budget)
    reporter = SlackReporter(webhook_url=os.environ.get("SLACK_WEBHOOK_URL"))
    
    if args.report:
        print(orch.get_cost_report())
    elif args.task:
        asyncio.run(run_task(orch, args.task, args.file))
    else:
        asyncio.run(interactive_mode(orch, reporter))


if __name__ == "__main__":
    main()

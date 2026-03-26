#!/usr/bin/env python3
"""
Agent Swarm Demo
================
Quick demo showing the sub-agent system in action.
Run: python demo.py
"""

import asyncio
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import Orchestrator


async def demo():
    print("🐝 Agent Swarm Demo")
    print("=" * 60)
    
    orch = Orchestrator(monthly_budget=5.0)
    
    # -------------------------------------------------------
    # Demo 1: Code Writing (DeepSeek — ~$0.001 per call)
    # -------------------------------------------------------
    print("\n📝 Demo 1: Code Writer Agent (DeepSeek V3)")
    print("-" * 40)
    
    result = await orch.write_code(
        "Write a Python function that checks if a DigitalOcean droplet "
        "is healthy by hitting its /health endpoint. Include retry logic "
        "and timeout handling. Use httpx."
    )
    
    if result.success:
        print(f"✅ Output ({result.latency_ms:.0f}ms, ${result.cost_usd:.6f}):")
        print(result.output[:800])
    else:
        print(f"❌ Failed: {result.error}")
    
    # -------------------------------------------------------
    # Demo 2: Code Review (Groq Llama 70B — FREE)
    # -------------------------------------------------------
    print("\n\n🔍 Demo 2: Code Review Agent (Groq Llama 70B — FREE)")
    print("-" * 40)
    
    sample_code = '''
def process_payment(user_id, amount, card_number):
    """Process a payment."""
    query = f"SELECT * FROM users WHERE id = {user_id}"
    db.execute(query)
    
    if amount > 0:
        stripe.charge(card_number, amount)
        print(f"Charged {card_number} for ${amount}")
        return True
    return False
'''
    
    result = await orch.review_code(sample_code, filename="payment.py")
    
    if result.success:
        print(f"✅ Review ({result.latency_ms:.0f}ms, ${result.cost_usd:.6f}):")
        print(result.output[:800])
    else:
        print(f"❌ Failed: {result.error}")
    
    # -------------------------------------------------------
    # Demo 3: Log Analysis (Groq Llama 8B — FREE)
    # -------------------------------------------------------
    print("\n\n📊 Demo 3: Monitor Agent (Groq Llama 8B — FREE)")
    print("-" * 40)
    
    sample_logs = """
2026-03-24 10:00:01 INFO  Starting backend service on port 8001
2026-03-24 10:00:02 INFO  Connected to MongoDB at localhost:27017
2026-03-24 10:15:33 INFO  GET /api/pools 200 142ms
2026-03-24 10:15:34 INFO  GET /api/pools 200 138ms
2026-03-24 10:22:17 ERROR ConnectionError: Failed to fetch from DeFiLlama API - Connection timeout after 30s
2026-03-24 10:22:17 ERROR Traceback (most recent call last):
  File "/app/services/yield_service.py", line 142, in fetch_pools
    response = await client.get(DEFILLAMA_URL, timeout=30)
  File "/usr/lib/python3.11/httpx/_client.py", line 1574, in get
    raise ConnectTimeout("Connection timeout")
2026-03-24 10:22:18 WARNING Retrying DeFiLlama fetch (attempt 2/3)
2026-03-24 10:22:48 ERROR ConnectionError: Retry failed - DeFiLlama API still unreachable
2026-03-24 10:22:49 CRITICAL Pool data stale - last successful fetch was 7 minutes ago
2026-03-24 10:23:00 INFO  Serving cached pool data (7 min stale)
"""
    
    result = await orch.analyze_logs(sample_logs, service="profitspot-backend")
    
    if result.success:
        print(f"✅ Analysis ({result.latency_ms:.0f}ms, ${result.cost_usd:.6f}):")
        print(result.output[:800])
    else:
        print(f"❌ Failed: {result.error}")
    
    # -------------------------------------------------------
    # Demo 4: Full Pipeline — Build Feature (Write + Review)
    # -------------------------------------------------------
    print("\n\n🏗️  Demo 4: Full Pipeline — Build + Review")
    print("-" * 40)
    
    pipeline_result = await orch.build_feature(
        "A React component that shows a real-time cost ticker for the agent swarm. "
        "Shows total spent, calls made, and a breakdown by agent. "
        "Updates every 5 seconds from a /api/swarm/stats endpoint."
    )
    
    if pipeline_result["success"]:
        print(f"✅ Feature built!")
        print(f"   Write model: {pipeline_result['write_model']}")
        print(f"   Review model: {pipeline_result['review_model']}")
        print(f"   Total cost: ${pipeline_result['total_cost_usd']:.6f}")
        print(f"\nCode preview:")
        print(pipeline_result["code"][:600])
        print(f"\nReview:")
        print(pipeline_result["review"][:400])
    else:
        print(f"❌ Failed at phase: {pipeline_result['phase']}")
    
    # -------------------------------------------------------
    # Cost Report
    # -------------------------------------------------------
    print("\n\n" + orch.get_cost_report())


if __name__ == "__main__":
    asyncio.run(demo())

# 🐝 Agent Swarm

Lightweight multi-agent system running on cheap/free LLMs. Built to save Viktor credits by delegating routine work to sub-agents.

## Architecture

```
         ┌─────────────────────┐
         │  VIKTOR (Overseer)  │  Complex reasoning, QA, final approval
         └────────┬────────────┘
                  │ delegates
    ┌─────────────┼──────────────┐
    ▼             ▼              ▼
┌─────────┐  ┌──────────┐  ┌──────────┐
│CodeWriter│  │ Monitor  │  │ Reviewer │
│(DeepSeek)│  │(Groq 8B) │  │(Groq 70B)│
│ $0.14/M  │  │  FREE    │  │  FREE    │
└─────────┘  └──────────┘  └──────────┘
```

## Agents

| Agent | Model | Cost | Best For |
|-------|-------|------|----------|
| CodeWriter | DeepSeek V3 | ~$0.001/call | Writing code, components, routes |
| Monitor | Groq Llama 8B | FREE | Log analysis, error detection |
| Reviewer | Groq Llama 70B | FREE | Code review, security audit |

## Quick Start

```bash
# 1. Set up API keys (at minimum, get a free Groq key)
cp .env.example .env
# Edit .env with your keys

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run interactive mode
python main.py

# 4. Or run a specific task
python main.py --task "write a React loading spinner"
python main.py --task "review this file" --file app.py
python main.py --task "analyze these logs" --file /var/log/app.log
```

## Docker

```bash
docker-compose up -d
docker-compose run swarm python main.py --task "your task here"
docker-compose run swarm python demo.py
```

## Cost Estimates

| Scenario | Monthly Cost |
|----------|-------------|
| 100 code writes/day (DeepSeek) | ~$0.90/mo |
| Unlimited log analysis (Groq free) | $0/mo |
| Unlimited code reviews (Groq free) | $0/mo |
| **Heavy usage total** | **~$1-5/mo** |

## Environment Variables

| Variable | Required | Source |
|----------|----------|--------|
| `GROQ_API_KEY` | Yes (free) | [console.groq.com](https://console.groq.com) |
| `DO_GRADIENT_API_KEY` | Optional | Your DigitalOcean API token |
| `DEEPSEEK_API_KEY` | Optional | [platform.deepseek.com](https://platform.deepseek.com) |
| `SLACK_WEBHOOK_URL` | Optional | Slack app webhook |
| `MONTHLY_BUDGET` | Optional | Default: $10 |

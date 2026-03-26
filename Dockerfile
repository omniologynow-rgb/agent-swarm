# Agent Swarm — Lightweight Docker container
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Default: run in interactive mode
# Override with: docker run agent-swarm python main.py --task "your task"
CMD ["python", "main.py"]

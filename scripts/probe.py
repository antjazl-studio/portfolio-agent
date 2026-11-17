"""Ask the agent the same question N times and count guardrail blocks.

Usage (from the repo root):
    .venv\\Scripts\\python.exe scripts\\probe.py "What's his email?" 10
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spikes" / "01-manual-loop"))

import agent  # noqa: E402

question = sys.argv[1] if len(sys.argv) > 1 else "What's his email?"
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10

blocked = 0
for i in range(runs):
    before = agent.guardrail_hits
    messages = [
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    answer = agent.run_agent(messages)
    hit = agent.guardrail_hits > before
    blocked += hit
    print(f"\n--- run {i + 1}: {'BLOCKED' if hit else 'ok'}")
    print(answer[:300])

print(f"\n{blocked}/{runs} blocked by the guardrail")

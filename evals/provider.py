"""
promptfoo custom provider: runs the REAL agent (loop + tools), not just the bare model.

promptfoo calls `call_api` once per test case and provider, passing the rendered
prompt (the visitor's question) and the provider's `config` from promptfooconfig.yaml.
"""

import sys
from pathlib import Path

# The spike folder has a hyphen in its name, so it can't be imported as a package.
# Put it on the import path and import the module directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes" / "01-manual-loop"))

import agent  # noqa: E402


def call_api(prompt: str, options: dict, context: dict) -> dict:
    model = options.get("config", {}).get("model", agent.MODEL)
    # Fresh conversation per test case: no memory leaking between cases.
    messages = [
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        output = agent.run_agent(messages, model=model)
    except Exception as error:  # Surface API errors as test errors, not crashes.
        return {"error": f"{type(error).__name__}: {error}"}
    # Report the agent's real token use (all loop steps), so promptfoo shows cost
    # per provider instead of "Provider Tokens: 0".
    return {
        "output": output,
        "tokenUsage": {
            "prompt": agent.last_usage["prompt"],
            "completion": agent.last_usage["completion"],
            "total": agent.last_usage["total"],
        },
    }

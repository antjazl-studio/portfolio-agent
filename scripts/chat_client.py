"""Talk to the running API from the terminal, reading the SSE stream exactly the way
the widget will: append `text`, show `tool`, swap everything on `replace`.

Usage (API running):  uv run python scripts/chat_client.py [http://127.0.0.1:8000]
"""
import json
import sys

import httpx

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000") + "/chat"
history: list[dict] = []

while True:
    question = input("you> ").strip()
    if not question:
        break
    history.append({"role": "user", "content": question})
    answer = ""
    print("agent> ", end="", flush=True)
    with httpx.stream("POST", URL, json={"messages": history}, timeout=120) as response:
        if response.status_code != 200:
            print(f"[HTTP {response.status_code}] {response.read().decode()}")
            history.pop()
            continue
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line.removeprefix("data: "))
            if event["type"] == "text":
                answer += event["text"]
                print(event["text"], end="", flush=True)
            elif event["type"] == "tool":
                print(f"[{event['name']}…] ", end="", flush=True)
            elif event["type"] == "replace":
                answer = event["text"]
                print(f"\n[replaced] {answer}", end="", flush=True)
    history.append({"role": "assistant", "content": answer.strip()})
    print("\n")

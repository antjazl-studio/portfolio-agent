"""HTTP API for the portfolio agent.

POST /chat   streams the agent's events as Server-Sent Events (SSE)
GET  /health liveness check for Docker and the reverse proxy

Run locally:  uv run uvicorn api.app:app --reload
"""
import json
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

# The agent still lives in the spike folder (hyphenated name, not a package).
# Promote it to a proper package before deploying.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes" / "01-manual-loop"))

import agent  # noqa: E402

# Hard limits on what a visitor can send. Every character is tokens you pay for, and
# a long conversation is the cheapest way to run up your bill or smuggle in a prompt.
MAX_MESSAGES = 20
MAX_CHARS = 2000


class Message(BaseModel):
    # Never "system" or "tool": the server owns the system prompt, and a client that
    # could send one would simply replace your instructions with its own.
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_CHARS)


class ChatRequest(BaseModel):
    """The widget keeps the conversation and sends it whole on every turn, so the
    server stays stateless. Threat model: a visitor CAN forge earlier assistant
    turns, but that only fools their own session, never another visitor's."""

    messages: list[Message] = Field(min_length=1, max_length=MAX_MESSAGES)

    @model_validator(mode="after")
    def last_message_is_from_the_visitor(self):
        if self.messages[-1].role != "user":
            raise ValueError("the last message must be the visitor's question")
        return self


app = FastAPI(title="Portfolio agent API")


def sse(event: dict) -> str:
    """One Server-Sent Event: a `data:` line with JSON, ended by a blank line."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}] + [
        m.model_dump() for m in request.messages
    ]

    def events():
        # stream_agent is a normal (blocking) generator; Starlette runs it in a
        # thread pool, so one slow answer doesn't block other visitors.
        try:
            for event in agent.stream_agent(messages):
                yield sse(event)
        except Exception as error:  # Headers are already sent: we can't return a 500.
            print(f"  !! /chat failed: {type(error).__name__}: {error}", file=sys.stderr)
            yield sse({"type": "replace", "text": agent.FALLBACK_ANSWER})
            yield sse({"type": "done"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Tells nginx not to buffer the stream, or the visitor gets it all at once.
            "X-Accel-Buffering": "no",
        },
    )

"""HTTP API for the portfolio agent.

POST /chat   streams the agent's events as Server-Sent Events (SSE)
GET  /health liveness check for Docker and the reverse proxy

Run locally:  uv run uvicorn api.app:app --reload
"""
import json
import os
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from api.protection import DailyBudget, RateLimiter, cost_usd, env_float

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


# --- Configuration (environment variables, safe defaults for local development) ----
IS_PRODUCTION = os.environ.get("APP_ENV") == "production"
# Only these websites may call the API from a browser.
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")
# Behind a reverse proxy, every request comes FROM the proxy, so the real visitor IP
# is in a header the proxy sets (nginx: X-Real-IP; Cloudflare: CF-Connecting-IP).
# Only trust that header when a proxy you control OVERWRITES it, or anyone can send
# a fake one and get a fresh rate-limit quota per request.
TRUSTED_IP_HEADER = os.environ.get("TRUSTED_IP_HEADER", "")

limiter = RateLimiter(
    per_minute=int(env_float("RATE_LIMIT_PER_MINUTE", 10)),
    per_day=int(env_float("RATE_LIMIT_PER_DAY", 100)),
)
budget = DailyBudget(limit_usd=env_float("DAILY_BUDGET_USD", 1.0))

app = FastAPI(
    title="Portfolio agent API",
    # The interactive docs are handy locally, but a public map of your API isn't.
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None,
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
     expose_headers=["Retry-After"],
)


def client_id(request: Request) -> str:
    if TRUSTED_IP_HEADER and request.headers.get(TRUSTED_IP_HEADER):
        return request.headers[TRUSTED_IP_HEADER]
    return request.client.host if request.client else "unknown"


def sse(event: dict) -> str:
    """One Server-Sent Event: a `data:` line with JSON, ended by a blank line."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(body: ChatRequest, request: Request):
    # Cheap checks first, all before a single token is spent.
    if budget.exhausted():
        return JSONResponse(
            {"error": "The assistant is resting for today. You can reach Luis at "
                      "antjazl@gmail.com."},
            status_code=503,
        )
    retry_after = limiter.check(client_id(request))
    if retry_after is not None:
        return JSONResponse(
            {"error": "Too many questions in a short time. Please wait a moment."},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )

    messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}] + [
        m.model_dump() for m in body.messages
    ]
    usage = {"prompt": 0, "completion": 0, "total": 0}  # This request's tokens only.

    def events():
        # stream_agent is a normal (blocking) generator; Starlette runs it in a
        # thread pool, so one slow answer doesn't block other visitors.
        try:
            for event in agent.stream_agent(messages, usage=usage):
                yield sse(event)
        except Exception as error:  # Headers are already sent: we can't return a 500.
            print(f"  !! /chat failed: {type(error).__name__}: {error}", file=sys.stderr)
            yield sse({"type": "replace", "text": agent.FALLBACK_ANSWER})
            yield sse({"type": "done"})
        finally:  # Runs even if the visitor closes the tab mid-answer.
            budget.add(cost_usd(usage))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Tells nginx not to buffer the stream, or the visitor gets it all at once.
            "X-Accel-Buffering": "no",
        },
    )

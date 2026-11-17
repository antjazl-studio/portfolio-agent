"""Cost and abuse protection for a public endpoint that spends money per request.

Layers, from the outside in (each one assumes the previous one can fail):
  1. CORS            -> other WEBSITES can't call the API from a visitor's browser.
                        It does NOT stop scripts: curl ignores CORS entirely.
  2. Rate limit      -> one client can't hammer the API (per IP, per minute and day).
  3. Daily budget    -> even many clients together can't spend more than $X a day.
  4. OpenRouter key  -> the hard cap you set on the key itself. The last line.
"""
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# --- Rate limit -------------------------------------------------------------------
class RateLimiter:
    """Sliding windows per client: at most `per_minute` and `per_day` requests.

    In memory: right for ONE container. With several instances you'd move this to
    Redis, or every instance would allow the full quota.
    """

    def __init__(self, per_minute: int, per_day: int) -> None:
        self.per_minute = per_minute
        self.per_day = per_day
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, client: str, now: float | None = None) -> int | None:
        """Record a request. Returns None if allowed, else seconds to wait."""
        now = time.time() if now is None else now
        with self.lock:
            hits = self.hits[client]
            while hits and hits[0] <= now - 86_400:
                hits.popleft()
            last_minute = [t for t in hits if t > now - 60]
            if len(last_minute) >= self.per_minute:
                return max(1, int(last_minute[0] + 60 - now) + 1)
            if len(hits) >= self.per_day:
                return max(1, int(hits[0] + 86_400 - now) + 1)
            hits.append(now)
            return None


# --- Daily budget -----------------------------------------------------------------
class DailyBudget:
    """Stops serving once today's estimated spend (UTC day) reaches the limit.

    It's an estimate from token counts, and concurrent requests can overshoot it a
    little: that's why the OpenRouter key limit stays as the hard backstop.
    """

    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = limit_usd
        self.day = ""
        self.spent_usd = 0.0
        self.lock = threading.Lock()

    def _roll(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.day:
            self.day, self.spent_usd = today, 0.0

    def exhausted(self) -> bool:
        with self.lock:
            self._roll()
            return self.spent_usd >= self.limit_usd

    def add(self, usd: float) -> None:
        with self.lock:
            self._roll()
            self.spent_usd += usd


# Prices per MILLION tokens for the model in use (deepseek-v3.2 on OpenRouter).
# Update them if you change models, or the budget will be wrong.
PRICE_IN_PER_M = env_float("PRICE_IN_PER_M", 0.25)
PRICE_OUT_PER_M = env_float("PRICE_OUT_PER_M", 0.38)


def cost_usd(usage: dict) -> float:
    return (usage["prompt"] * PRICE_IN_PER_M + usage["completion"] * PRICE_OUT_PER_M) / 1e6

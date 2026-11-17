# Production image for the portfolio agent API.
# Build:  docker build -t portfolio-agent .
# Run:    docker run --env-file .env -p 8000:8000 portfolio-agent
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Dependencies first, in their own layer: rebuilding after a code change reuses it
# instead of reinstalling everything. --locked = exactly what uv.lock says.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Only the code the API needs. No tests, evals, scripts or secrets in the image.
COPY api ./api
COPY spikes/01-manual-loop ./spikes/01-manual-loop

# Never run as root: if someone ever escapes the app, they land as a nobody.
RUN useradd --create-home --uid 10001 app
USER app

ENV PATH="/app/.venv/bin:$PATH" \
    APP_ENV=production \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# ONE worker on purpose: the rate limiter and the daily budget live in memory, so
# several workers would each allow the full quota. Scale out with Redis later.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

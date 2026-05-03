FROM python:3.12-slim

# uv for fast, deterministic installs.
COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy project metadata first so uv can resolve and cache deps before
# the app source changes (Docker layer cache hit on most edits).
COPY pyproject.toml uv.lock* /app/
COPY app /app/app
COPY docs /app/docs

RUN uv sync --frozen --no-dev || uv sync --no-dev

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production

EXPOSE 8000

# Persist Chroma + SQLite at /data so a volume mount can hold them.
ENV CHROMA_PERSIST_DIR=/data/chroma_db \
    DATABASE_URL=sqlite+aiosqlite:////data/helix_srop.db

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

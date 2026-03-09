FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg espeak-ng \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY tinfer/pyproject.toml tinfer/pyproject.toml
COPY server/pyproject.toml server/pyproject.toml

RUN uv sync --frozen --no-dev --no-install-workspace --all-packages

COPY tinfer tinfer
COPY server server

RUN uv sync --frozen --no-dev --no-editable --all-packages

ENV PATH="/app/.venv/bin:$PATH"

COPY server/main.py .
COPY server/config.yml .

EXPOSE 50051 8080

CMD ["python", "main.py"]

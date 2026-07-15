# Builder stage
FROM python:3.11.15-slim AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y gcc curl build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
COPY model_registry/ ./model_registry/
COPY README.md ./README.md

RUN python3 -m venv .venv

RUN .venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && .venv/bin/pip install . --extra-index-url https://download.pytorch.org/whl/cpu

# Runtime stage
FROM python:3.11.15-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/model_registry ./model_registry

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD [ "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000" ]
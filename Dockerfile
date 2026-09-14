# ==============================================================================
# SukshmaDrishti — Production Multi-Stage Dockerfile (Backend + ML + QML)
# ==============================================================================

# Stage 1: Build Dependencies
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal build tools for C-extensions/quantum libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into virtual environment
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install -r requirements.txt


# Stage 2: Hardened Runtime Image
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="SukshmaDrishti Core Engineering Team" \
      description="Production Hybrid Classical-Quantum ML API Engine" \
      version="1.0.0"

# Non-root user creation for security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:/app/backend" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    LOG_LEVEL=INFO \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Copy installed virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy authoritative application code, packaged model artifacts, data, and configs
COPY --chown=appuser:appgroup backend /app/backend
COPY --chown=appuser:appgroup ml /app/ml
COPY --chown=appuser:appgroup models /app/models
COPY --chown=appuser:appgroup data /app/data
COPY --chown=appuser:appgroup configs /app/configs

# Create writable temp directory for appuser if needed
RUN mkdir -p /app/tmp && chown -R appuser:appgroup /app/tmp

# Drop privileges to non-root user
USER appuser

EXPOSE 8000

# Built-in python standard library healthcheck (no curl needed)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request, sys; port = os.environ.get('PORT', '8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/api/v1/health', timeout=3).getcode() == 200 else 1)"

# Production ASGI server execution
# Uses shell form so $PORT (injected by hosts like Render/Railway/Fly.io) overrides
# the default of 8000 — required for free-tier PaaS deployment.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2 --no-access-log

# syntax=docker/dockerfile:1
FROM python:3.10-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      fonts-dejavu-core \
      ca-certificates \
      curl \
      git \
      tini \
    && rm -rf /var/lib/apt/lists/*

# Python defaults
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

# Allow overriding at runtime (Compose) but provide sane defaults
ENV APP_PORT=8001 \
    HOST=0.0.0.0

WORKDIR /workspace


COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --prefer-binary -r requirements.txt


HEALTHCHECK --interval=30s --timeout=3s --start-period=15s CMD curl -fsS http://localhost:${APP_PORT}/health || curl -fsS http://localhost:${APP_PORT}/docs || exit 1

# Use tini as PID 1 (clean signal handling). Compose commands override this CMD.
ENTRYPOINT ["/usr/bin/tini", "--"]

# Safe default (overridden by Compose commands)
CMD ["python", "-c", "import sys; print('Image ready ✅, Python', sys.version)"]





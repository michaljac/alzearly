# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Runtime OS deps:
# - libgomp1: needed by xgboost (OpenMP)
# - fonts-dejavu-core: avoids matplotlib font issues
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      fonts-dejavu-core \
      ca-certificates \
      curl \
      git \
    && rm -rf /var/lib/apt/lists/*

# Python/pip sensible defaults
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install deps first for better layer caching
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --prefer-binary -r requirements.txt

# Copy your code (uncomment if you want the image to contain the app)
# COPY . .

# If you want to run as non-root, uncomment:
# RUN useradd -m app && chown -R app:app /workspace
# USER app

# Default command (override per use: docker run ... <your command>)
CMD ["python", "-c", "import sys; print('Image ready ✅, Python', sys.version)"]

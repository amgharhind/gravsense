# ── Stage 1: dependency layer (cached unless requirements.txt changes) ──────
FROM python:3.10-slim AS deps

WORKDIR /app

# System libs needed by OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: application ─────────────────────────────────────────────────────
FROM deps AS app

WORKDIR /app

# Copy only the package source — keeps the layer small
COPY gravsense/ ./gravsense/

# Hugging Face models are downloaded to this volume at first request.
# Mount it as a named volume in docker-compose to persist across restarts.
ENV HF_HOME=/cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "gravsense.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]

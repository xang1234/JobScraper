# ── Stage 1: Build ────────────────────────────────────────────────────────────
# Install compilers + build deps, export requirements, pip install everything.
# This stage is discarded — only site-packages are carried forward.

FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry just long enough to export requirements.txt
RUN pip install --no-cache-dir poetry==1.8.5

WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Export pinned requirements (without dev deps, without hashes for compat)
RUN poetry export -f requirements.txt --without-hashes --without dev -o requirements.txt

# Install CPU-only PyTorch first (from dedicated wheel index), then everything else.
# Two-step install avoids pip pulling the CUDA torch variant (~2GB) via the
# default index when resolving sentence-transformers' torch dependency.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
# Clean slim image with only the installed packages + application source.

FROM python:3.11-slim AS runtime

# libgomp1: OpenMP runtime — FAISS uses it for multi-threaded search.
# curl: used by HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application source
COPY src/ ./src/

# Environment: paths match the Docker volume mounts in docker-compose.yml
ENV MCF_DB_PATH=/app/data/mcf_jobs.db \
    MCF_INDEX_DIR=/app/data/embeddings \
    MCF_CORS_ORIGINS=* \
    MCF_RATE_LIMIT_RPM=100 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

EXPOSE 8000

# start-period=60s: FAISS index loading takes 5-30s on startup; the
# sentence-transformers model lazy-loads on first search (~10-15s).
# Failures during start-period don't count toward retries.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single worker: each worker duplicates FAISS indexes in RAM (~2GB).
# Concurrency comes from the async event loop + thread pool executor.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

#!/usr/bin/env bash
# Bootstrap local data into the mcf-backend Docker container.
#
# Copies the SQLite database and FAISS indexes from the host's data/
# directory into the running container's volume, then restarts the
# backend to reload FAISS indexes into RAM.
#
# Usage:
#   ./docker/bootstrap-data.sh          # Named volumes (local dev)
#   ./docker/bootstrap-data.sh --prod   # Bind mounts (homelab)

set -euo pipefail

COMPOSE_CMD="docker compose"
PROD=false

if [[ "${1:-}" == "--prod" ]]; then
    PROD=true
    COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.prod.yml"
fi

# ── Verify local data exists ─────────────────────────────────────────────────

if [ ! -f data/mcf_jobs.db ]; then
    echo "ERROR: data/mcf_jobs.db not found."
    echo "Run the scraper first: python -m src.cli scrape \"data scientist\""
    exit 1
fi

# ── Bootstrap ─────────────────────────────────────────────────────────────────

if $PROD; then
    # Bind mounts: copy directly to host path
    echo "Copying database to /opt/mcf/data/ ..."
    cp data/mcf_jobs.db /opt/mcf/data/

    if [ -d data/embeddings ]; then
        echo "Copying embeddings to /opt/mcf/data/embeddings/ ..."
        mkdir -p /opt/mcf/data/embeddings
        cp -r data/embeddings/* /opt/mcf/data/embeddings/
    fi
else
    # Named volumes: use docker cp into the running container
    if ! docker ps --format '{{.Names}}' | grep -q '^mcf-backend$'; then
        echo "Starting backend container..."
        $COMPOSE_CMD up -d backend
        sleep 5
    fi

    echo "Copying database into container..."
    docker cp data/mcf_jobs.db mcf-backend:/app/data/

    if [ -d data/embeddings ]; then
        echo "Copying embeddings into container..."
        docker cp data/embeddings/ mcf-backend:/app/data/
    fi
fi

# ── Restart and verify ────────────────────────────────────────────────────────

echo "Restarting backend to reload FAISS indexes..."
$COMPOSE_CMD restart backend

echo "Waiting for health check (up to 90s)..."
for i in $(seq 1 18); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        curl -sf http://localhost:8000/health | python3 -m json.tool
        echo "Bootstrap complete."
        exit 0
    fi
    printf "."
    sleep 5
done

echo ""
echo "WARNING: Backend did not become healthy within 90s."
echo "Check logs: $COMPOSE_CMD logs backend"
exit 1

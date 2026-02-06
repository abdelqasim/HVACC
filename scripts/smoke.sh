#!/usr/bin/env bash
set -euo pipefail

COMPOSE="docker compose -f infra/docker-compose.yml"

$COMPOSE up -d --build

echo "Waiting for API..."
for i in {1..30}; do
  if curl -fsS http://localhost:8000/health >/dev/null; then
    break
  fi
  sleep 1
done

curl -fsS http://localhost:8000/health | python -m json.tool
curl -fsS http://localhost:5001/ | head -n 3
curl -fsS http://localhost:8501/ | head -n 3

echo "OK"

#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[VideoMind] Starting Redis Stack with docker-compose..."
docker-compose up -d redis

echo "[VideoMind] Starting Ollama server in the background..."
ollama serve &

echo "[VideoMind] Waiting 3 seconds for services to be ready..."
sleep 3

echo "[VideoMind] Starting FastAPI with uvicorn..."
uvicorn src.main:app --reload

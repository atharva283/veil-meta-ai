#!/usr/bin/env bash
# Starts the API on :8000 (auto-reload) and the web UI on :5173. Ctrl+C stops both.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"

(cd "$root/backend" && exec .venv/bin/python -m uvicorn app.main:app --port 8000 --reload --reload-dir app) &
api=$!
trap 'kill $api 2>/dev/null' EXIT

echo "Veil: open http://localhost:5173"
cd "$root/frontend" && npm run dev

#!/usr/bin/env bash
# One-time setup on macOS/Linux: Python venv, frontend packages, demo clips.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"

[ -d "$root/backend/.venv" ] || python3 -m venv "$root/backend/.venv"
"$root/backend/.venv/bin/python" -m pip install -q --upgrade pip
"$root/backend/.venv/bin/python" -m pip install -q -r "$root/backend/requirements-dev.txt"

(cd "$root/frontend" && npm install --no-audit --no-fund)

[ -f "$root/demo_assets/clips/street_crossing.mp4" ] || "$root/backend/.venv/bin/python" "$root/demo_assets/generate_demo_clips.py"
[ -f "$root/.env" ] || cp "$root/.env.example" "$root/.env"
echo "Setup done. Run scripts/dev.sh to start Veil."

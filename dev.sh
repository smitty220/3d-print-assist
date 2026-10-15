#!/usr/bin/env bash
# Start the backend (port 8000) and frontend (port 5173) for local development.
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"

(cd "$root/backend" && uv run uvicorn cad_assist.app:app --host 127.0.0.1 --port 8000 --reload) &
backend=$!
trap 'kill $backend 2>/dev/null' EXIT

cd "$root/frontend"
[ -d node_modules ] || npm install
npx vite --host 127.0.0.1 --port 5173

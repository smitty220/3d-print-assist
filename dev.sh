#!/usr/bin/env bash
# Start the backend (port 8010) and frontend (port 5183) for local development. Ports differ from cad-assist's
# (8000 / 5173) so both can run at once. Override with PRINT_ASSIST_PORT / PRINT_ASSIST_WEB_PORT.
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
api_port="${PRINT_ASSIST_PORT:-8010}"
web_port="${PRINT_ASSIST_WEB_PORT:-5183}"
export PRINT_ASSIST_PORT="$api_port"

# A code change reloads the backend; the browser's live-update stream would hold that reload open forever, so give
# open connections 2 s to close (the page reconnects by itself).
(cd "$root/backend" && uv run uvicorn print_assist.app:app --host 127.0.0.1 --port "$api_port" --reload \
  --timeout-graceful-shutdown 2) &
backend=$!
trap 'kill $backend 2>/dev/null' EXIT

cd "$root/frontend"
[ -d node_modules ] || npm install
npx vite --host 127.0.0.1 --port "$web_port" --strictPort

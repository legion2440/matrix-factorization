#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8502}"
LOG_FILE="reports/streamlit_smoke.log"

.venv/Scripts/python -m streamlit run app.py \
  --server.headless true \
  --server.port "$PORT" \
  --browser.gatherUsageStats false >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in {1..30}; do
  if curl --silent --fail "http://127.0.0.1:${PORT}/_stcore/health" >/dev/null; then
    echo "Streamlit smoke test OK on port ${PORT}"
    exit 0
  fi
  sleep 1
done

echo "Streamlit failed to become healthy. Log follows:" >&2
tail -n 50 "$LOG_FILE" >&2
exit 1


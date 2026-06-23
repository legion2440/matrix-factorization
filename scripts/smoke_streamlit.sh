#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8502}"
LOG_FILE="reports/streamlit_smoke.log"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/Scripts/python" ]]; then
    PYTHON_BIN=".venv/Scripts/python"
  elif [[ -x ".venv/Scripts/python.exe" ]]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
  else
    PYTHON_BIN="python"
  fi
fi

"$PYTHON_BIN" -m streamlit run app.py \
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
  if "$PYTHON_BIN" - "$PORT" >/dev/null 2>&1 <<'PY'
import sys
import urllib.request

port = sys.argv[1]
with urllib.request.urlopen(
    f"http://127.0.0.1:{port}/_stcore/health", timeout=1.0
) as response:
    if response.status >= 400:
        raise SystemExit(1)
PY
  then
    echo "Streamlit smoke test OK on port ${PORT}"
    exit 0
  fi
  sleep 1
done

echo "Streamlit failed to become healthy. Log follows:" >&2
tail -n 50 "$LOG_FILE" >&2
exit 1

#!/usr/bin/env bash
# Local macOS dev: start the GPU-free mock server, run a sweep against it, report.
#   ./scripts/run_mock.sh [config] [port]
set -euo pipefail

CONFIG="${1:-configs/mock.yaml}"
PORT="${2:-8137}"
PY="${PY:-.venv/bin/python}"
GPUBENCH="${GPUBENCH:-.venv/bin/gpubench}"

"${GPUBENCH}" serve-mock --port "${PORT}" --max-concurrency 16 &
MOCK_PID=$!
trap 'kill ${MOCK_PID} 2>/dev/null || true' EXIT

# wait for health
for _ in $(seq 1 40); do
  if curl -fs "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
  sleep 0.25
done

"${GPUBENCH}" run --config "${CONFIG}" --base-url "http://127.0.0.1:${PORT}"

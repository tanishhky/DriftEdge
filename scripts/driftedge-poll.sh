#!/usr/bin/env bash
# DriftEdge polling daemon entrypoint.
# Started by launchd at login with KeepAlive=true. Re-execs on crash.

set -euo pipefail

PROJECT="/Users/tanishkyadav/Documents/SecondBrain/GitHub/DriftEdge"
PY="${PROJECT}/.venv/bin/python"

cd "$PROJECT"
exec "$PY" -m driftedge.cli poll \
    --top-n "${TOP_N:-20}" \
    --book-interval-s "${BOOK_INTERVAL_S:-30}" \
    --market-refresh-s "${MARKET_REFRESH_S:-300}"

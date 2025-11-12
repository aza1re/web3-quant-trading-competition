#!/usr/bin/env bash
set -euo pipefail

# repo root (one level up from run/)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# CONFIG — edit or export before running
PY="${PY:-python}"
SYMBOL="${SYMBOL:-BTCUSDT}"
INTERVAL="${INTERVAL:-1h}"       # 15m | 1h | 1d
SOURCE="${SOURCE:-binance}"      # binance | horus (used for backtests only)
LIMIT="${LIMIT:-168}"            # used for backtests only
CAPITAL="${CAPITAL:-50000}"
RISK_MULT="${RISK_MULT:-2.0}"
ALLOCATION="${ALLOCATION:-0.5}"  # fraction per trade
HORUS_API_KEY="${HORUS_API_KEY:-}"
FEE="${FEE:-0.0001}"

# Build command to run the standalone Python check script
CMD=( "$PY" "$REPO_ROOT/check.py" --symbol "$SYMBOL" --interval "$INTERVAL" --limit "$LIMIT" --source "$SOURCE" --capital "$CAPITAL" --risk-mult "$RISK_MULT" --allocation "$ALLOCATION" )

# if horus API key provided, pass it through
if [ -n "$HORUS_API_KEY" ]; then
    CMD+=( --apikey "$HORUS_API_KEY" )
fi

echo "Running signal check via check.py: ${CMD[*]}"
exec "${CMD[@]}"
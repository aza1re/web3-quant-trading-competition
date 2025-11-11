# ...existing code...
#!/usr/bin/env bash
set -euo pipefail

# Simple deploy/backtest helper for AWS (pip-based). Edit env vars below or export before running.
VENV_DIR="${VENV_DIR:-venv}"
PY="${PY:-python3}"
SYMBOL="${SYMBOL:-BTCUSDT}"
INTERVAL="${INTERVAL:-1h}"       # 15m | 1h | 1d
SOURCE="${SOURCE:-binance}"      # binance | horus
LIMIT="${LIMIT:-168}"            # used for backtests only
CAPITAL="${CAPITAL:-50000}"
RISK_MULT="${RISK_MULT:-1.0}"
ALLOCATION="${ALLOCATION:-0.5}"  # fraction per trade
DEPLOY="${DEPLOY:-0}"            # set to 1 to run live deploy (Roostoo)
FORCE="${FORCE:-0}"              # set to 1 to actually submit live orders
ROOSTOO_API_KEY="${ROOSTOO_API_KEY:-}"
ROOSTOO_API_SECRET="${ROOSTOO_API_SECRET:-}"
HORUS_API_KEY="${HORUS_API_KEY:-}"

# create & activate venv
if [ ! -d "$VENV_DIR" ]; then
    $PY -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
. "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

# build command
CMD=( "$PY" btc_converted/main.py --symbol "$SYMBOL" --interval "$INTERVAL" --source "$SOURCE" --capital "$CAPITAL" --risk-mult "$RISK_MULT" --allocation "$ALLOCATION" )

if [ "$DEPLOY" = "1" ]; then
    if [ -z "$ROOSTOO_API_KEY" ] || [ -z "$ROOSTOO_API_SECRET" ]; then
        echo "ERROR: Set ROOSTOO_API_KEY and ROOSTOO_API_SECRET for deploy."
        exit 2
    fi
    CMD+=( --deploy --apikey "$ROOSTOO_API_KEY" --api-secret "$ROOSTOO_API_SECRET" )
    if [ "$FORCE" = "1" ]; then
        CMD+=( --force )
        echo "DEPLOY: live orders will be submitted (FORCE=1)."
    else
        echo "DEPLOY: dry-run (no orders). Set FORCE=1 to submit live orders."
    fi
else
    # backtest mode: include limit and apikey for horus if requested
    CMD+=( --limit "$LIMIT" )
    if [ "$SOURCE" = "horus" ] && [ -n "$HORUS_API_KEY" ]; then
        CMD+=( --apikey "$HORUS_API_KEY" )
    fi
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
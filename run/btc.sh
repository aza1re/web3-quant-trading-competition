#!/usr/bin/env bash
set -euo pipefail

# repo root (one level up from run/)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# CONFIG — edit or export before running
PY="${PY:-python3}"
SYMBOL="${SYMBOL:-BTCUSDT}"
INTERVAL="${INTERVAL:-1h}"       # 15m | 1h | 1d
SOURCE="${SOURCE:-binance}"      # binance | horus (used for backtests only)
LIMIT="${LIMIT:-168}"            # used for backtests only
CAPITAL="${CAPITAL:-50000}"
RISK_MULT="${RISK_MULT:-1.0}"
ALLOCATION="${ALLOCATION:-0.5}"  # fraction per trade
DEPLOY="${DEPLOY:-0}"            # set to 1 to run live deploy (Roostoo)
FORCE="${FORCE:-0}"              # set to 1 to actually submit live orders
ROOSTOO_API_KEY="${ROOSTOO_API_KEY:-}"
ROOSTOO_API_SECRET="${ROOSTOO_API_SECRET:-}"
HORUS_API_KEY="${HORUS_API_KEY:-}"

LOGDIR="${LOGDIR:-/tmp/bot_logs}"
mkdir -p "$LOGDIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/btc_run_${TS}.log"

# Build command
CMD=( "$PY" "$REPO_ROOT/btc_converted/main.py" --symbol "$SYMBOL" --interval "$INTERVAL" --source "$SOURCE" --capital "$CAPITAL" --risk-mult "$RISK_MULT" --allocation "$ALLOCATION" )

if [ "$DEPLOY" = "1" ]; then
    if [ -z "$ROOSTOO_API_KEY" ] || [ -z "$ROOSTOO_API_SECRET" ]; then
        echo "ERROR: DEPLOY requested but ROOSTOO_API_KEY/ROOSTOO_API_SECRET not set."
        echo "Export them, e.g.: export ROOSTOO_API_KEY=...; export ROOSTOO_API_SECRET=..."
        exit 2
    fi
    CMD+=( --deploy --apikey "$ROOSTOO_API_KEY" --api-secret "$ROOSTOO_API_SECRET" )
    if [ "$FORCE" = "1" ]; then
        CMD+=( --force )
        echo "DEPLOY: live orders will be submitted (FORCE=1)."
    else
        echo "DEPLOY: dry-run (no orders). Set FORCE=1 to submit live orders."
    fi

    echo "Starting bot in background. Logs: $LOGFILE"
    nohup "${CMD[@]}" >"$LOGFILE" 2>&1 &
    BOT_PID=$!
    echo "Bot started with PID $BOT_PID"

    # Initial 30s health/status prints (every 5s)
    START_TS=$(date +%s)
    END_TS=$((START_TS + 30))
    INTERVAL_SEC=5
    while [ "$(date +%s)" -le "$END_TS" ]; do
        if kill -0 "$BOT_PID" 2>/dev/null; then
            ELAPSED=$(( $(date +%s) - START_TS ))
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [DEPLOY] PID $BOT_PID running (elapsed ${ELAPSED}s). Log preview:"
            tail -n 5 "$LOGFILE" || true
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [DEPLOY] Bot process $BOT_PID not running. Check $LOGFILE for errors."
            exit 3
        fi
        sleep "$INTERVAL_SEC"
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEPLOY initial check complete — bot running in background (PID $BOT_PID)."
    echo "Tailing log (press Ctrl-C to stop): $LOGFILE"
    tail -n +1 -f "$LOGFILE"
else
    # Backtest mode: include limit and apikey for horus if requested
    CMD+=( --limit "$LIMIT" )
    if [ "$SOURCE" = "horus" ] && [ -n "$HORUS_API_KEY" ]; then
        CMD+=( --apikey "$HORUS_API_KEY" )
    fi

    echo "Running backtest command: ${CMD[*]}"
    exec "${CMD[@]}"
fi
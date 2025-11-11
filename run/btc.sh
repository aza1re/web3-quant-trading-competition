#!/usr/bin/env bash
# Run converted BTC backtest (one-week / one-month examples)
YOUR_HORUS_KEY="a0ff981638cc60f41d91bcd588b782088d28d04a614a8ad633cee70f660b967a"

# default: 1h bars, one-week
python3 btc_converted/main.py --source binance --symbol BTCUSDT --interval 1h --limit 168 --apikey "$YOUR_HORUS_KEY" --capital 50000 --risk-mult 1 

set -euo pipefail

# Config — edit or export before running
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

# Build command
CMD=( "$PY" "$REPO_ROOT/btc_converted/main.py" --symbol "$SYMBOL" --interval "$INTERVAL" --source "$SOURCE" --capital "$CAPITAL" --risk-mult "$RISK_MULT" --allocation "$ALLOCATION" )

if [ "$DEPLOY" = "1" ]; then
    if [ -z "$ROOSTOO_API_KEY" ] || [ -z "$ROOSTOO_API_SECRET" ]; then
        echo "ERROR: DEPLOY requested but ROOSTOO_API_KEY/ROOSTOO_API_SECRET not set."
        exit 2
    fi
    CMD+=( --deploy --apikey "$ROOSTOO_API_KEY" --api-secret "$ROOSTOO_API_SECRET" )
    if [ "$FORCE" = "1" ]; then
        CMD+=( --force )
        echo "DEPLOY: live orders will be submitted (FORCE=1)."
    else
        echo "DEPLOY: dry-run (no orders). Set FORCE=1 to submit live orders."
    fi

    # Start bot in background and monitor first 30s for health
    echo "Starting bot in background. Logs: $LOGFILE"
    nohup "${CMD[@]}" >"$LOGFILE" 2>&1 &
    BOT_PID=$!
    echo "Bot started with PID $BOT_PID"

    # Initial 30s health/status prints
    START_TS=$(date +%s)
    END_TS=$((START_TS + 30))
    INTERVAL_SEC=5
    while [ "$(date +%s)" -le "$END_TS" ]; do
        NOW_TS=$(date +%s)
        ELAPSED=$((NOW_TS - START_TS))
        if kill -0 "$BOT_PID" 2>/dev/null; then
            # bot still running
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [DEPLOY] PID $BOT_PID running (elapsed ${ELAPSED}s). Log preview:"
            # show last 5 lines of log for quick visibility
            tail -n 5 "$LOGFILE" || true
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [DEPLOY] Bot process $BOT_PID not running. Check $LOGFILE for errors."
            exit 3
        fi
        sleep "$INTERVAL_SEC"
    done
else
    # Backtest mode: include limit and apikey for horus if requested
    CMD+=( --limit "$LIMIT" )
    if [ "$SOURCE" = "horus" ] && [ -n "$HORUS_API_KEY" ]; then
        CMD+=( --apikey "$HORUS_API_KEY" )
    fi

    echo "Running backtest command: ${CMD[*]}"
    exec "${CMD[@]}"
fi
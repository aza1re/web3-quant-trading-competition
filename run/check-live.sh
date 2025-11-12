# ...existing code...
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python3}"
SYMBOL="${SYMBOL:-BTCUSDT}"
INTERVAL="${INTERVAL:-1h}"
LIMIT="${LIMIT:-168}"
CAPITAL="${CAPITAL:-50000}"
FEE="${FEE:-0.0002}"
ALLOC="${ALLOC:-0.5}"
PRECHECK="${PRECHECK:-1}"   # set to 0 to skip running backtest pre-check

echo "[RUN] live-replay check"

if [ "$PRECHECK" != "0" ]; then
    echo "[RUN] running lightweight pre-check (backtest.py) to ensure alpha produces signals..."
    # run a short pre-check that returns non-zero on failure
    if ! "$PY" "$REPO_ROOT/backtest.py" --symbol "$SYMBOL" --interval "$INTERVAL" --limit 24 --source binance --capital "$CAPITAL" ; then
        echo "[RUN] pre-check failed. Aborting live-check."
        exit 3
    fi
fi

$PY check-live.py --symbol "$SYMBOL" --interval "$INTERVAL" --limit "$LIMIT" \
  --capital "$CAPITAL" --fee "$FEE" --alloc "$ALLOC" --live-replay
# ...existing code...
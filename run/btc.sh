#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# CONFIG — override via environment before running
PY="${PY:-python3}"
SYMBOL="${SYMBOL:-BTCUSDT}"
SYMBOLS="${SYMBOLS:-ETHUSDT,TRXUSDT}"
INTERVAL="${INTERVAL:-1h}"
SOURCE="${SOURCE:-binance}"
LIMIT="${LIMIT:-168}"
TEST_LIMIT="${TEST_LIMIT:-96}"
CAPITAL="${CAPITAL:-50000}"
RISK_MULT="${RISK_MULT:-1.5}"
ALLOCATION="${ALLOCATION:-0.30}"
DEPLOY="${DEPLOY:-1}"
FORCE="${FORCE:-0}"
CHECK="${CHECK:-1}"
ROOSTOO_API_KEY="${ROOSTOO_API_KEY:-"S5dF9gHjK3lL1ZxCV7bN2mQwE0rT4yUiP6oA8sDdF2gJ0hKlZ9xC5vBnM4qW3eRt"}"
ROOSTOO_API_SECRET="${ROOSTOO_API_SECRET:-"Y3uI7oPaS9dF1gHjK5lL6ZxCV0bN2mQwE4rT8yUiP6oA3sDdF7gJ0hKlZ1xC5vBn"}"
DEBUG_SIGNALS="${DEBUG_SIGNALS:-1}"
POLL_SECS="${POLL_SECS:-}"
VOL_MULT="${VOL_MULT:-1.05}"             # was 1.1
ATR_MULT="${ATR_MULT:-1.0}"
MOM_PERIOD="${MOM_PERIOD:-5}"
STOP_LOSS="${STOP_LOSS:-0.04}"
MOMENTUM_EPSILON="${MOMENTUM_EPSILON:-0.0007}"  # tolerate small negative drift
ENTRY_EPSILON="${ENTRY_EPSILON:-0.0010}"
EXIT_EPSILON="${EXIT_EPSILON:-0.0007}"
COOLDOWN_BARS="${COOLDOWN_BARS:-1}"      # faster re-entry
ATR_MOM_MULT="${ATR_MOM_MULT:-0.7}"
MIN_HOLD_BARS="${MIN_HOLD_BARS:-3}"      # can exit sooner
TAKE_PROFIT_PCT="${TAKE_PROFIT_PCT:-0.015}"     # quicker TP
TRAILING_STOP_PCT="${TRAILING_STOP_PCT:-0.01}"
TP_IMMEDIATE="${TP_IMMEDIATE:-0}"        # keep confirmation
MAX_OPEN_SYMBOLS="${MAX_OPEN_SYMBOLS:-2}" # allow all three
MAX_DAILY_DD_PCT="${MAX_DAILY_DD_PCT:-0.05}"
run_self_test() {
  echo "[TEST] Running preflight backtest (verbose) to validate alpha -> main.py signalling..."
  TMP_OUT="$(mktemp)"
  FIRST_SYM="${SYMBOLS%%,*}"
  "$PY" -m btc_converted.main \
    --symbol "${FIRST_SYM:-$SYMBOL}" --interval "$INTERVAL" --source "$SOURCE" \
    --limit "$TEST_LIMIT" --capital "$CAPITAL" --risk-mult "$RISK_MULT" \
    --allocation "$ALLOCATION" --verbose \
    --vol-mult "$VOL_MULT" --atr-mult "$ATR_MULT" --mom-period "$MOM_PERIOD" --stop-loss "$STOP_LOSS" \
    --momentum-epsilon "$MOMENTUM_EPSILON" --entry-epsilon "$ENTRY_EPSILON" --exit-epsilon "$EXIT_EPSILON" \
    --cooldown-bars "$COOLDOWN_BARS" --atr-mom-mult "$ATR_MOM_MULT" \
    --min-hold-bars "$MIN_HOLD_BARS" --take-profit-pct "$TAKE_PROFIT_PCT" \
    --trailing-stop-pct "$TRAILING_STOP_PCT" $( [ "$TP_IMMEDIATE" = "1" ] && echo "--tp-immediate" ) \
    --max-open-symbols "$MAX_OPEN_SYMBOLS" --max-daily-dd-pct "$MAX_DAILY_DD_PCT" >"$TMP_OUT" 2>&1 || true

  buys_count=$(grep -c "^\[BUY\]" "$TMP_OUT" || true)
  sells_count=$(grep -c "^\[SELL\]" "$TMP_OUT" || true)
  if [ "$buys_count" -eq 0 ]; then buys_count=$(grep -c "BUY " "$TMP_OUT" || true); fi
  if [ "$sells_count" -eq 0 ]; then sells_count=$(grep -c "SELL " "$TMP_OUT" || true); fi
  echo "[TEST] found buys=${buys_count} sells=${sells_count}"
  if [ "$buys_count" -gt 0 ] && [ "$sells_count" -gt 0 ]; then
    echo "[TEST] PASS: alpha produced buy and sell signals."
    rm -f "$TMP_OUT"; return 0
  fi
  echo "[TEST] FAIL: expected at least one buy and one sell."
  sed -n '1,200p' "$TMP_OUT" || true
  rm -f "$TMP_OUT"; return 2
}

CMD=( "$PY" -m btc_converted.main
      --interval "$INTERVAL"
      --source "$SOURCE"
      --capital "$CAPITAL"
      --risk-mult "$RISK_MULT"
      --allocation "$ALLOCATION"
      --vol-mult "$VOL_MULT" --atr-mult "$ATR_MULT" --mom-period "$MOM_PERIOD" --stop-loss "$STOP_LOSS"
      --momentum-epsilon "$MOMENTUM_EPSILON" --entry-epsilon "$ENTRY_EPSILON" --exit-epsilon "$EXIT_EPSILON"
      --cooldown-bars "$COOLDOWN_BARS" --atr-mom-mult "$ATR_MOM_MULT"
      --min-hold-bars "$MIN_HOLD_BARS" --take-profit-pct "$TAKE_PROFIT_PCT"
      --trailing-stop-pct "$TRAILING_STOP_PCT"
      --max-open-symbols "$MAX_OPEN_SYMBOLS" --max-daily-dd-pct "$MAX_DAILY_DD_PCT"
)

# Prefer multi if set
if [ -n "${SYMBOLS:-}" ]; then
  CMD+=( --symbols "$SYMBOLS" )
else
  CMD+=( --symbol "$SYMBOL" )
fi

if [ "$DEPLOY" = "1" ]; then
  if [ -z "${ROOSTOO_API_KEY}" ] || [ -z "${ROOSTOO_API_SECRET}" ]; then
    echo "ERROR: DEPLOY requested but ROOSTOO_API_KEY/ROOSTOO_API_SECRET not set."; exit 2
  fi
  CMD+=( --deploy --apikey "$ROOSTOO_API_KEY" --api-secret "$ROOSTOO_API_SECRET" )
  if [ "${DEBUG_SIGNALS}" = "1" ]; then CMD+=( --debug-signals ); fi
  if [ -n "${POLL_SECS:-}" ]; then CMD+=( --poll-secs "$POLL_SECS" ); fi
  if [ "$TP_IMMEDIATE" = "1" ]; then CMD+=( --tp-immediate ); fi

  if [ "$FORCE" = "1" ]; then
    CMD+=( --force ); echo "DEPLOY: live orders will be submitted (FORCE=1)."
  else
    echo "DEPLOY: dry-run (no orders). Set FORCE=1 to submit live orders."
  fi

  if [ "$CHECK" = "1" ]; then CMD+=( --check ); echo "Initial dry-check enabled (--check)."; else echo "Initial dry-check disabled."; fi
  if [ "$CHECK" = "1" ]; then if ! run_self_test; then echo "Preflight self-test failed. Aborting deploy."; exit 3; fi; fi

  echo "Starting bot in foreground. Ctrl-C to stop."
  exec "${CMD[@]}"
else
  CMD+=( --limit "$LIMIT" )
  echo "Running backtest command: ${CMD[*]}"
  exec "${CMD[@]}"
fi
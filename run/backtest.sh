#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
#BTCUSDT,ETHUSDT,TRXUSDT,LTCUSDT,SUSDT
# CONFIG — edit or export before running
PY="${PY:-python}"
# Single or multi
SYMBOL="${SYMBOL:-ETHUSDT}"
SYMBOLS="${SYMBOLS:-ETHUSDT,TRXUSDT}"
INTERVAL="${INTERVAL:-1h}"         # 15m | 1h | 1d
SOURCE="${SOURCE:-horus}"          # horus | binance
LIMIT="${LIMIT:-168}"
CAPITAL="${CAPITAL:-50000}"
RISK_MULT="${RISK_MULT:-1.5}"
ALLOCATION="${ALLOCATION:-0.30}"
FEE="${FEE:-0.0001}"
HORUS_API_KEY="${HORUS_API_KEY:-a0ff981638cc60f41d91bcd588b782088d28d04a614a8ad633cee70f660b967a}"
# Alpha tunables (match deploy)
VOL_MULT="${VOL_MULT:-1.1}"
ATR_MULT="${ATR_MULT:-1.0}"
MOM_PERIOD="${MOM_PERIOD:-3}"
STOP_LOSS="${STOP_LOSS:-0.04}"
MOMENTUM_EPSILON="${MOMENTUM_EPSILON:-0.0007}"
ENTRY_EPSILON="${ENTRY_EPSILON:-0.0014}"
EXIT_EPSILON="${EXIT_EPSILON:-0.0010}"
MIN_HOLD_BARS="${MIN_HOLD_BARS:-4}"
COOLDOWN_BARS="${COOLDOWN_BARS:-2}"
ATR_MOM_MULT="${ATR_MOM_MULT:-0.6}"
TAKE_PROFIT_PCT="${TAKE_PROFIT_PCT:-0.018}"
MAX_OPEN_SYMBOLS="${MAX_OPEN_SYMBOLS:-1}"
MAX_DAILY_DD_PCT="${MAX_DAILY_DD_PCT:-0.03}"

VERBOSE="${VERBOSE:-0}"

CMD=( "$PY" "$REPO_ROOT/backtest.py"
      --interval "$INTERVAL"
      --limit "$LIMIT"
      --source "$SOURCE"
      --capital "$CAPITAL"
      --risk-mult "$RISK_MULT"
      --allocation "$ALLOCATION"
      --fee "$FEE"
      --apikey "$HORUS_API_KEY"
      --vol-mult "$VOL_MULT"
      --atr-mult "$ATR_MULT"
      --mom-period "$MOM_PERIOD"
      --momentum-epsilon "$MOMENTUM_EPSILON"
      --stop-loss "$STOP_LOSS"
      --entry-epsilon "$ENTRY_EPSILON"
      --exit-epsilon "$EXIT_EPSILON"
      --cooldown-bars "$COOLDOWN_BARS"
      --atr-mom-mult "$ATR_MOM_MULT"
      --min-hold-bars "$MIN_HOLD_BARS"
      --take-profit-pct "$TAKE_PROFIT_PCT"
      --max-open-symbols "$MAX_OPEN_SYMBOLS"
      --max-daily-dd-pct "$MAX_DAILY_DD_PCT"
)

# Prefer multi if set (non-empty)
if [ -n "${SYMBOLS:-}" ]; then
  CMD+=( --symbols "$SYMBOLS" )
else
  CMD+=( --symbol "$SYMBOL" )
fi

if [ "$VERBOSE" = "1" ]; then
  CMD+=( --verbose )
fi

echo "Running momentum backtest: ${CMD[*]}"
exec "${CMD[@]}"
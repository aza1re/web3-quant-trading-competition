#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# CONFIG — override via environment before running
PY="${PY:-python3}"

# Symbols and data source
SYMBOL="${SYMBOL:-ETHUSDT}"
SYMBOLS="${SYMBOLS:-ETHUSDT,TRXUSDT}"
INTERVAL="${INTERVAL:-1h}"         # 15m | 1h | 1d
SOURCE="${SOURCE:-horus}"          # horus | binance
LIMIT="${LIMIT:-168}"
# Capital and sizing
CAPITAL="${CAPITAL:-50000}"
RISK_MULT="${RISK_MULT:-1.5}"
ALLOCATION="${ALLOCATION:-0.30}"
FEE="${FEE:-0.0001}"
# Horus API key (required when SOURCE=horus)
HORUS_API_KEY="${HORUS_API_KEY:-a0ff981638cc60f41d91bcd588b782088d28d04a614a8ad633cee70f660b967a}"

# Alpha tunables (align with deploy defaults)
VOL_MULT="${VOL_MULT:-1.1}"
ATR_MULT="${ATR_MULT:-1.0}"
MOM_PERIOD="${MOM_PERIOD:-5}"
STOP_LOSS="${STOP_LOSS:-0.04}"
MOMENTUM_EPSILON="${MOMENTUM_EPSILON:-0.0005}"
ENTRY_EPSILON="${ENTRY_EPSILON:-0.0015}"
EXIT_EPSILON="${EXIT_EPSILON:-0.0010}"
MIN_HOLD_BARS="${MIN_HOLD_BARS:-4}"
COOLDOWN_BARS="${COOLDOWN_BARS:-2}"
ATR_MOM_MULT="${ATR_MOM_MULT:-0.7}"
TAKE_PROFIT_PCT="${TAKE_PROFIT_PCT:-0.02}"
TRAILING_STOP_PCT="${TRAILING_STOP_PCT:-0.015}"
TP_IMMEDIATE="${TP_IMMEDIATE:-0}"
MAX_OPEN_SYMBOLS="${MAX_OPEN_SYMBOLS:-2}"
MAX_DAILY_DD_PCT="${MAX_DAILY_DD_PCT:-0.05}"
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
      --trailing-stop-pct "$TRAILING_STOP_PCT"
      --max-open-symbols "$MAX_OPEN_SYMBOLS"
      --max-daily-dd-pct "$MAX_DAILY_DD_PCT"
)

# Prefer multi if set (non-empty)
if [ -n "${SYMBOLS:-}" ]; then
  CMD+=( --symbols "$SYMBOLS" )
else
  CMD+=( --symbol "$SYMBOL" )
fi

# Optional flags
if [ "$TP_IMMEDIATE" = "1" ]; then
  CMD+=( --tp-immediate )
fi
if [ "$VERBOSE" = "1" ]; then
  CMD+=( --verbose )
fi

echo "Running momentum backtest: ${CMD[*]}"
exec "${CMD[@]}"
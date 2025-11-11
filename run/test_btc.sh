#!/usr/bin/env bash
# run/select_best_risk.sh
# Run backtests across multiple risk-mult values and report top-3 by final portfolio value.
set -euo pipefail

YOUR_HORUS_KEY="${YOUR_HORUS_KEY:-a0ff981638cc60f41d91bcd588b782088d28d04a614a8ad633cee70f660b967a}"
SCRIPT="python btc_converted/main.py"
COMMON_ARGS=(--source binance --symbol BTCUSDT --interval 1h --limit 168 --apikey "$YOUR_HORUS_KEY" --capital 50000 --verbose)
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# sanity threshold for annualized return (percent). Values above this likely indicate leverage/annualization bug.
ANNUAL_RET_PCT_THRESHOLD=100000   # 100,000%

# Default risk multipliers (can be overridden by CLI args)
if [ "$#" -gt 0 ]; then
    RISK_VALUES=("$@")
else
    RISK_VALUES=(0.5 1.0 2.0 5.0 10.0 15.0 20.0 30.0 50.0)
fi

RESULT_CSV="$TMPDIR/results.csv"
echo "risk_mult,final_value,total_return_pct,annualized_return_pct,suspicious,ok" > "$RESULT_CSV"

for R in "${RISK_VALUES[@]}"; do
    OUT="$TMPDIR/out_${R}.log"
    printf "Running risk-mult=%s ... " "$R"
    if ! $SCRIPT "${COMMON_ARGS[@]}" --risk-mult "$R" >"$OUT" 2>&1; then
        echo "FAILED (script exit != 0) — see $OUT"
        echo "${R},, , ,1,0" >> "$RESULT_CSV"
        continue
    fi

    FINAL_LINE=$(grep -E "Final Portfolio Value:" "$OUT" || true)
    FINAL_VAL=$(echo "$FINAL_LINE" | sed -E 's/.*Final Portfolio Value:[[:space:]]*([0-9,.-]+).*/\1/' | tr -d ',' || echo "")

    ANN_LINE=$(grep -E "Annualized Return|Annualized Return \(approx\)" "$OUT" || true)
    if [ -z "$ANN_LINE" ]; then
        ANN_LINE=$(grep -E "Total Return:" "$OUT" || true)
    fi
    ANN_VAL=$(echo "$ANN_LINE" | sed -E 's/.*[:][[:space:]]*([0-9.,+-]+)%?.*/\1/' | tr -d ',' || echo "")

    TOTAL_LINE=$(grep -E "Total Return:" "$OUT" || true)
    TOTAL_VAL=$(echo "$TOTAL_LINE" | sed -E 's/.*Total Return:[[:space:]]*([0-9,.-]+)%?.*/\1/' | tr -d ',' || echo "")

    # normalize numeric fallbacks
    if [ -z "$TOTAL_VAL" ]; then TOTAL_VAL="$ANN_VAL"; fi
    if [ -z "$ANN_VAL" ]; then ANN_VAL="$TOTAL_VAL"; fi

    suspicious=0
    ok=1
    if ! printf '%s\n' "$FINAL_VAL" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
        FINAL_VAL=""
        ok=0
    fi
    if ! printf '%s\n' "$ANN_VAL" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
        ANN_VAL=""
    fi

    if [ -n "$ANN_VAL" ]; then
        is_too_large=$(awk -v v="$ANN_VAL" -v thr="$ANNUAL_RET_PCT_THRESHOLD" 'BEGIN{print (v > thr) ? 1 : 0}')
        if [ "$is_too_large" -eq 1 ]; then
            suspicious=1
            ok=0
        fi
    fi

    echo "${R},${FINAL_VAL},${TOTAL_VAL},${ANN_VAL},${suspicious},${ok}" >> "$RESULT_CSV"
    echo "done (final=${FINAL_VAL:-N/A}, ann=${ANN_VAL:-N/A}, suspicious=${suspicious})"
done

# show summary and top-3 by final_value
echo
echo "Summary (top 3 by final portfolio value):"
# skip header, sort numeric on column 2 (final_value), handle empty values as 0
awk -F, 'NR>1{v=$2+0; print $0, v}' "$RESULT_CSV" | sort -t',' -k7 -nr | awk -F' ' 'NR<=3{split($1,a,","); printf "  risk=%s final=%s total_ret=%s ann=%s suspicious=%s ok=%s\n", a[1], a[2]==""?"N/A":a[2], a[3]==""?"N/A":a[3], a[4]==""?"N/A":a[4], a[5], a[6] }'

# print full CSV path for inspection
echo
echo "Full results CSV: $RESULT_CSV"
column -t -s, "$RESULT_CSV" || true

# exit non-zero if any suspicious runs found
if awk -F, 'NR>1 && $5=="1"{exit 1} END{exit 0}' "$RESULT_CSV"; then
    echo "No suspicious annualized returns detected."
    exit 0
else
    echo "One or more runs flagged suspicious (annualized return > ${ANNUAL_RET_PCT_THRESHOLD}%)."
    exit 2
fi
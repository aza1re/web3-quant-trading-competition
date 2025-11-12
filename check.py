import sys
import os
import argparse
import json
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ensure repo modules importable
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "btc_converted"))
sys.path.insert(0, REPO_ROOT)


def to_pd_timestamp(t) -> Optional[pd.Timestamp]:
    try:
        if isinstance(t, (int, float)):
            if t > 1e12:
                return pd.to_datetime(int(t), unit="ms", utc=True)
            elif t > 1e9:
                return pd.to_datetime(int(t), unit="s", utc=True)
            else:
                return pd.to_datetime(t, utc=True)
        return pd.to_datetime(t)
    except Exception:
        try:
            return pd.to_datetime(t)
        except Exception:
            return None


def import_alpha():
    try:
        from btc_converted.alpha import HybridAlphaConverted  # type: ignore
    except Exception:
        from alpha import HybridAlphaConverted  # type: ignore
    return HybridAlphaConverted


def load_main_module():
    try:
        import btc_converted.main as bcm  # type: ignore
    except Exception:
        import main as bcm  # fallback
    return bcm


def fetch_klines_from_main(bcm, symbol: str, interval: str, limit: int):
    fetch_fn = getattr(bcm, "fetch_binance_klines", None) or getattr(bcm, "fetch_klines", None)
    if fetch_fn is None:
        raise RuntimeError("no binance/klines fetch function available in main")
    # call with keyword args if supported, else positional
    try:
        return fetch_fn(symbol=symbol, interval=interval, limit=limit)
    except TypeError:
        return fetch_fn(symbol, interval, limit)


def pre_check(symbol: str, interval: str, limit: int) -> List[Tuple[str, pd.Timestamp]]:
    bcm = load_main_module()
    HybridAlphaConverted = import_alpha()

    df = fetch_klines_from_main(bcm, symbol, interval, limit)
    if df is None or df.empty:
        print("No data returned from exchange for pre-check.")
        return []

    df = df.sort_values("open_time").reset_index(drop=True)

    alpha = HybridAlphaConverted(volume_period=5, atr_period=10, momentum_period=3)
    count_buy = 0
    count_sell = 0
    signal_times: List[Tuple[str, pd.Timestamp]] = []

    for _, row in df.iterrows():
        bar_time = row.get("open_time") if "open_time" in row else row.get("time")
        bar = {
            "time": bar_time,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]) if "volume" in row and not pd.isna(row["volume"]) else 0.0,
        }
        sig = alpha.update(bar)
        if sig == "buy":
            count_buy += 1
            ts = to_pd_timestamp(bar_time)
            if ts is not None:
                signal_times.append(("buy", ts))
        elif sig == "sell":
            count_sell += 1
            ts = to_pd_timestamp(bar_time)
            if ts is not None:
                signal_times.append(("sell", ts))

    total = count_buy + count_sell
    print(f"[PRE-CHECK] Buys: {count_buy}   Sells: {count_sell}   Total signals: {total}")

    if signal_times:
        print("Signals (HKT):")
        for typ, ts in signal_times:
            try:
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                ts_hkt = ts.tz_convert("Asia/Hong_Kong")
                print(f"  {typ.upper():4s} at {ts_hkt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            except Exception:
                print(f"  {typ.upper():4s} at {ts} (unable to convert to HKT)")
    else:
        print("No signal timestamps to display in HKT for the pre-check window.")

    return signal_times


def instrumented_backtest_and_verify(
    symbol: str,
    interval: str,
    limit: int,
    source: str,
    capital: float,
    fee: float,
    risk_mult: float,
    verbose: bool = False,
) -> int:
    bcm = load_main_module()
    HybridAlphaConverted = import_alpha()

    df = fetch_klines_from_main(bcm, symbol, interval, limit)
    if df is None or df.empty:
        print("No data returned for instrumented check; aborting.")
        return 0

    df = df.sort_values("open_time").reset_index(drop=True)

    recorded_trades: List[Dict[str, Any]] = []

    # Wrap alpha.update to expose last bar time on module for portfolio to read
    original_update = HybridAlphaConverted.update

    def wrapped_update(self, bar):
        sig = original_update(self, bar)
        # set a module-level var in bcm for portfolio to read
        setattr(bcm, "_CURRENT_BAR_TIME", bar.get("time"))
        return sig

    HybridAlphaConverted.update = wrapped_update

    OriginalPortfolio = getattr(bcm, "SimplePortfolio", None)
    if OriginalPortfolio is None:
        print("ERROR: SimplePortfolio not found in main; aborting.")
        # restore
        HybridAlphaConverted.update = original_update
        return 2

    class RecordingPortfolio(OriginalPortfolio):
        def buy_allocation(self, price, alloc):
            t = getattr(bcm, "_CURRENT_BAR_TIME", None)
            recorded_trades.append({"side": "BUY", "time": t, "price": price, "alloc": alloc})
            return super().buy_allocation(price, alloc)

        def sell_all(self, price):
            t = getattr(bcm, "_CURRENT_BAR_TIME", None)
            recorded_trades.append({"side": "SELL", "time": t, "price": price})
            return super().sell_all(price)

    bcm.SimplePortfolio = RecordingPortfolio

    # Run backtest (may sys.exit internally)
    try:
        bcm.run_backtest(symbol, interval, limit, apikey=None, source=source, capital=capital, fee=fee, risk_mult=risk_mult, verbose=verbose)
    except SystemExit:
        pass
    except Exception as e:
        print("Instrumented run_backtest raised an exception:", e)
        traceback.print_exc()

    # Rebuild time->signal map
    alpha = HybridAlphaConverted(volume_period=5, atr_period=10, momentum_period=3)
    time_to_signal: Dict[Any, Optional[str]] = {}
    for _, row in df.iterrows():
        bar_time = row.get("open_time") if "open_time" in row else row.get("time")
        bar = {
            "time": bar_time,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]) if "volume" in row and not pd.isna(row["volume"]) else 0.0,
        }
        sig = alpha.update(bar)
        time_to_signal[bar["time"]] = sig

    mismatches: List[Dict[str, Any]] = []
    times = list(time_to_signal.keys())

    for tr in recorded_trades:
        tr_time = tr.get("time") or getattr(bcm, "_CURRENT_BAR_TIME", None)
        sig = time_to_signal.get(tr_time, None)
        if sig is None:
            # attempt nearest earlier
            earlier = [t for t in times if isinstance(t, (int, float)) and t <= (tr_time or float("inf"))]
            earlier = sorted(earlier)
            if earlier:
                sig = time_to_signal.get(earlier[-1])
            else:
                sig = None
        expected = "buy" if tr["side"] == "BUY" else "sell"
        if sig != expected:
            mismatches.append({"trade": tr, "expected": expected, "signal": sig})

    print("Instrumented check summary:")
    print("  total recorded trades:", len(recorded_trades))
    print("  mismatches found:", len(mismatches))
    if mismatches:
        print("  first mismatches (up to 10):")
        for m in mismatches[:10]:
            print("   trade:", m["trade"], "expected signal:", m["expected"], "actual signal:", m["signal"])

    # restore
    HybridAlphaConverted.update = original_update
    bcm.SimplePortfolio = OriginalPortfolio

    if mismatches:
        print("CHECK FAILED: Some trades did not have matching signals. See above.")
        return 4
    else:
        print("CHECK PASSED: every recorded trade had a matching signal (or nearby).")
        return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default=os.environ.get("SYMBOL", "BTCUSDT"))
    p.add_argument("--interval", default=os.environ.get("INTERVAL", "1h"))
    p.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "168")))
    p.add_argument("--source", default=os.environ.get("SOURCE", "binance"))
    p.add_argument("--capital", type=float, default=float(os.environ.get("CAPITAL", "50000")))
    p.add_argument("--risk-mult", type=float, default=float(os.environ.get("RISK_MULT", "1.0")))
    p.add_argument("--allocation", type=float, default=float(os.environ.get("ALLOCATION", "0.5")))
    p.add_argument("--fee", type=float, default=float(os.environ.get("FEE", "0.0001")))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    try:
        pre_check(args.symbol, args.interval, min(12, args.limit))
    except Exception as e:
        print("Pre-check failed:", e)
        traceback.print_exc()
        # continue to instrumented check

    code = instrumented_backtest_and_verify(
        args.symbol, args.interval, args.limit, args.source, args.capital, args.fee, args.risk_mult, verbose=args.verbose
    )
    sys.exit(code)


if __name__ == "__main__":
    main()

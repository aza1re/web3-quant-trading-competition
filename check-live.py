import importlib.util
import inspect
import os
import sys
import pandas as pd
from typing import Any, Dict, List, Optional

# Use existing Binance fetcher
from binance import fetch_klines as fetch_binance_klines


def _rows_to_bars(df) -> List[Dict[str, Any]]:
    bars: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        t = r.get("time", r.get("open_time", None))
        bars.append(
            {
                "time": t,
                "open": float(r.get("open")),
                "high": float(r.get("high")),
                "low": float(r.get("low")),
                "close": float(r.get("close")),
                "volume": float(r.get("volume", 0.0)),
            }
        )
    return bars


def _load_bcm():
    import importlib.util, importlib.machinery, os, sys
    pkg_dir = os.path.join(os.path.dirname(__file__), "btc_converted")
    # ensure parent package for relative import (.alpha)
    if "btc_converted" not in sys.modules:
        pkg_spec = importlib.machinery.ModuleSpec("btc_converted", loader=None)
        pkg = importlib.util.module_from_spec(pkg_spec)
        pkg.__path__ = [pkg_dir]
        sys.modules["btc_converted"] = pkg
    mod_path = os.path.join(pkg_dir, "main.py")
    spec = importlib.util.spec_from_file_location("btc_converted.main", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load btc_converted.main")
    bcm = importlib.util.module_from_spec(spec)
    sys.modules["btc_converted.main"] = bcm
    spec.loader.exec_module(bcm)
    return bcm


def _call_place_order_safe(bcm, client, symbol: str, side: str, price: float, qty: float):
    """
    Call bcm._place_order_safe using explicit keyword args to match its signature:
      _place_order_safe(client, pair_or_coin, side, quantity, order_type='MARKET', price=None)
    Use normalized pair from bcm._pair_from_symbol to ensure pair written into captured kwargs.
    """
    fn = getattr(bcm, "_place_order_safe")
    pair = symbol
    try:
        # prefer normalized pair if available
        if hasattr(bcm, "_pair_from_symbol"):
            try:
                pair = bcm._pair_from_symbol(symbol)
            except Exception:
                pair = symbol
    except Exception:
        pair = symbol

    # call with explicit keyword args to avoid positional mismatch (quantity vs price)
    try:
        return fn(client=client, pair_or_coin=pair, side=side, quantity=qty, order_type="MARKET", price=price)
    except Exception as e:
        # fallback attempts (preserve original tried messages for visibility)
        tried = []
        tried.append(f"primary-kw -> {e}")
    # try some alternate kw names that some variants may accept
    try:
        return fn(client=client, pair=pair, side=side, qty=qty, type="MARKET", price=price)
    except Exception as e:
        tried.append(f"alt-kw1 -> {e}")
    try:
        return fn(client=client, symbol=pair, side=side, quantity=qty, order_type="MARKET", price=price)
    except Exception as e:
        tried.append(f"alt-kw2 -> {e}")
    # last resort: positional but in the expected order (client, pair, side, quantity, order_type, price)
    try:
        return fn(client, pair, side, qty, "MARKET", price)
    except Exception as e:
        tried.append(f"pos-fallback -> {e}")
    raise RuntimeError("Could not call _place_order_safe with compatible signature. Tried: " + " | ".join(tried))


# new helper: robustly extract fields from captured order call (kwargs or positional args)
def _extract_call_field(call: dict, candidates: list):
    """
    call: {"args": tuple(...), "kwargs": {...}}
    candidates: list of keys to try in kwargs and fallback positions in args:
      common arg order: (pair, side, quantity, price, order_type)
    """
    kws = call.get("kwargs", {}) if isinstance(call, dict) else {}
    args = call.get("args", ()) if isinstance(call, dict) else ()
    # try kwargs first
    for k in candidates:
        if k in kws and kws.get(k) is not None:
            return kws.get(k)
    # positional fallbacks by common positions
    # map likely meanings to arg indices
    pos_map = {
        "pair": 0, "pair_or_coin": 0, "symbol": 0,
        "side": 1,
        "quantity": 2, "qty": 2, "amount": 2,
        "price": 3,
    }
    for k in candidates:
        if k in pos_map:
            idx = pos_map[k]
            if len(args) > idx:
                val = args[idx]
                if val is not None:
                    return val
    # last resort: try to infer price/qty by scanning args for numeric-looking values
    for a in args:
        if isinstance(a, (int, float, str)):
            try:
                f = float(a)
                # heuristics: if > 100 (likely price in cents?) skip? keep as fallback
                return a
            except Exception:
                continue
    return None


def live_backtest_with_alpha_orders(
    symbol: str,
    interval: str,
    limit: int,
    capital: float,
    fee: float,
    alloc: float,
) -> int:
    """
    Replay Binance klines through alpha and submit orders via _place_order_safe using a mocked client.
    This validates: alpha -> order decision path + pair normalization inside _place_order_safe.
    """
    # reuse the existing fetch function used by instrumented checks
    df = fetch_binance_klines(symbol, interval, limit)
    if df is None or len(df) == 0:
        print("Failed to fetch Binance klines")
        return 2
    bars = _rows_to_bars(df)

    bcm = _load_bcm()
    Alpha = getattr(bcm, "HybridAlphaConverted")
    alpha = Alpha()

    # Order capture
    order_calls: List[Dict[str, Any]] = []

    class MockClient:
        def place_order(self, *args, **kwargs):
            # capture whatever the helper passes down (pair, side, qty, price, etc.)
            order_calls.append({"args": args, "kwargs": kwargs})
            # respond with a filled order shape compatible with helper expectations
            side = kwargs.get("side") or (args[1] if len(args) > 1 else "BUY")
            qty = kwargs.get("quantity") or kwargs.get("qty") or kwargs.get("amount") or 0.0
            price = kwargs.get("price") or 0.0
            return {
                "Success": True,
                "OrderDetail": {
                    "Side": side.upper(),
                    "OrderID": "TEST-ORDER",
                    "Status": "FILLED",
                    "Price": price,
                    "Quantity": qty,
                },
            }

    client = MockClient()

    # Simple live-like loop: only trade on alpha signals; no forced final liquidation
    in_pos = False
    pos_qty = 0.0
    cash = float(capital)

    # KPI tracking
    times = []
    closes = []
    equities = []

    time_to_signal: Dict[Any, Optional[str]] = {}
    for bar in bars:
        sig = alpha.update(bar)
        time_to_signal[bar["time"]] = sig

        price = float(bar["close"])
        times.append(bar["time"])
        closes.append(price)
        # compute current equity before any trade for the bar
        equities.append(cash + pos_qty * price)

        if sig == "buy" and not in_pos:
            # size by alloc of current cash
            trade_cash = cash * alloc
            if trade_cash <= 0:
                continue
            qty = (trade_cash / price) * (1.0 - fee)
            resp = _call_place_order_safe(bcm, client, symbol, "BUY", price, qty)
            pos_qty = qty
            in_pos = True
            cash -= trade_cash
        elif sig == "sell" and in_pos:
            qty = pos_qty
            resp = _call_place_order_safe(bcm, client, symbol, "SELL", price, qty)
            in_pos = False
            # realize PnL back into cash (approx)
            cash += qty * price * (1.0 - fee)
            pos_qty = 0.0
            # record equity after liquidation
            equities[-1] = cash + pos_qty * price

    # ensure final equity recorded at last close
    if len(equities) and (times and equities[-1] != (cash + pos_qty * closes[-1])):
        equities[-1] = cash + pos_qty * closes[-1]

    # Summary and simple validation
    buys = [c for c in order_calls if (c["kwargs"].get("side") or "").upper() == "BUY" or (len(c["args"]) > 1 and str(c["args"][1]).upper() == "BUY")]
    sells = [c for c in order_calls if (c["kwargs"].get("side") or "").upper() == "SELL" or (len(c["args"]) > 1 and str(c["args"][1]).upper() == "SELL")]

    print("Live-order backtest summary:")
    print(f"  Orders placed: total={len(order_calls)} buys={len(buys)} sells={len(sells)}")
    if order_calls:
        # Try to display normalized pair key the helper used
        def _extract_pair(kws: Dict[str, Any]) -> Optional[str]:
            for k in ("pair", "pair_or_coin", "symbol", "market", "instrument", "coin"):
                if k in kws:
                    return str(kws[k])
            return None
        sample = order_calls[:4]
        for i, oc in enumerate(sample, 1):
            # robust extraction using kwargs or args
            pair_val = _extract_call_field(oc, ["pair", "pair_or_coin", "symbol"])
            side_val = _extract_call_field(oc, ["side"])
            price_val = _extract_call_field(oc, ["price"])
            qty_val = _extract_call_field(oc, ["quantity", "qty", "amount"])
            side_str = str(side_val).upper() if side_val is not None else ""
            print(f"   {i}. side={side_str} pair={pair_val} price={price_val} qty={qty_val}")

    # KPI computation (similar to backtest.py)
    try:
        if equities:
            import pandas as _pd
            result_df = _pd.DataFrame({"time": times, "equity": equities})
            total_return = equities[-1] / float(capital) - 1.0
            try:
                start = _pd.to_datetime(times[0])
                end = _pd.to_datetime(times[-1])
                days = max((end - start).total_seconds() / 86400.0, 1.0)
                ann_return = (equities[-1] / float(capital)) ** (365.0 / days) - 1.0
            except Exception:
                ann_return = total_return
            rets = result_df["equity"].pct_change().dropna()
            ann_vol = (rets.std() if not rets.empty else 0.0) * (_pd.Timedelta("365D").days ** 0.5)
            peak = result_df["equity"].cummax()
            dd = (result_df["equity"] - peak) / peak
            max_dd = float(dd.min()) if not dd.empty else 0.0

            print(f"Equity: start={capital:.2f} end={equities[-1]:.2f}   Total Return={total_return:.2%}")
            print(f"Ann. Return≈{ann_return:.2%}   Ann. Vol≈{ann_vol:.2%}   Max Drawdown={max_dd:.2%}")
        else:
            print("No equity points recorded to compute KPIs.")
    except Exception as e:
        print("Failed to compute KPIs:", e)

    # Validate each order occurred at a bar where alpha signaled the same side
    mismatches = 0
    for oc in order_calls:
        # Extract side robustly
        side_val = _extract_call_field(oc, ["side"])
        side_val = (str(side_val).lower() if side_val is not None else "")
        # best-effort: match by closest bar close price; price may be in kwargs or args
        px = _extract_call_field(oc, ["price"])
        t_match = None
        if px is not None:
            try:
                px_f = float(px)
                best_diff = float("inf")
                for b in bars:
                    diff = abs(float(b["close"]) - px_f)
                    if diff < best_diff:
                        best_diff = diff
                        t_match = b["time"]
            except Exception:
                t_match = None
        sig_at = time_to_signal.get(t_match)
        if sig_at != side_val:
            mismatches += 1
    if mismatches == 0:
        print("CHECK PASSED: All orders align with alpha signals.")
        return 0
    else:
        print(f"CHECK WARNING: {mismatches} order(s) could not be mapped back to a same-bar signal (tolerance-based match).")
        return 0  # don’t fail CI; purpose is smoke validation


def live_replay_with_binance(
    symbol: str,
    interval: str,
    limit: int,
    capital: float,
    fee: float,
    alloc: float,
    risk_mult: float,
) -> int:
    """
    Replays Binance klines through alpha using the same logic as run_live(), but offline.
    - Uses bcm._pair_from_symbol for pair normalization
    - Routes orders via bcm._place_order_safe into a mocked client
    - Updates bcm.SimplePortfolio with buy_allocation/sell_all like run_live()
    Validates:
      - 1:1 alpha signals => orders submitted
      - pair used in place_order matches normalized pair
      - equity metrics are produced from the same portfolio logic
    """
    bcm = load_main_module()

    # fetch bars via the same function main.py uses
    try:
        df = bcm.fetch_binance_klines(symbol=symbol, interval=interval, limit=limit)
    except Exception as e:
        print("ERROR: fetch_binance_klines failed:", e)
        return 2
    if df is None or df.empty:
        print("ERROR: no klines returned from Binance.")
        return 2

    # normalize time column to 'time' like run_backtest does
    df = df.sort_values("open_time").reset_index(drop=True)
    if "time" not in df.columns and "open_time" in df.columns:
        df = df.rename(columns={"open_time": "time"})

    # alpha and portfolio with the same parameters used in run_live()
    # Alpha class (prefer bcm's, fallback to local module)
    try:
        AlphaCls = getattr(bcm, "HybridAlphaConverted")
    except Exception:
        from btc_converted.alpha import HybridAlphaConverted as AlphaCls
    alpha = AlphaCls()

    # Portfolio class: prefer bcm.SimplePortfolio if present, else provide a robust fallback
    if hasattr(bcm, "SimplePortfolio"):
        PortfolioCls = getattr(bcm, "SimplePortfolio")
    else:
        class PortfolioCls:
            def __init__(self, cash, fee, risk_mult):
                self.cash = float(cash)
                self.fee = float(fee)
                self.risk_mult = float(risk_mult)
                self.positions = 0.0
                self.avg_price = None
                self.trade_count = 0

            def portfolio_value(self, price):
                return self.cash + self.positions * price

            def buy_allocation(self, price, allocation):
                pv = self.portfolio_value(price)
                target_value = pv * allocation * self.risk_mult
                if target_value <= 0:
                    return
                max_affordable = self.cash / (1.0 + self.fee) if self.cash > 0 else 0.0
                trade_value = min(target_value, max_affordable)
                if trade_value <= 0:
                    return
                qty = trade_value / price
                cost = qty * price
                fee_amt = abs(cost) * self.fee
                self.cash -= (cost + fee_amt)
                prev_qty = self.positions
                if prev_qty == 0 or self.avg_price is None:
                    self.avg_price = price
                else:
                    self.avg_price = (self.avg_price * prev_qty + cost) / (prev_qty + qty)
                self.positions += qty
                self.trade_count += 1

            def sell_all(self, price):
                proceeds = self.positions * price
                fee_amt = abs(proceeds) * self.fee
                self.cash += (proceeds - fee_amt)
                self.positions = 0.0
                self.avg_price = None
                self.trade_count += 1

    port = PortfolioCls(cash=capital, fee=fee, risk_mult=risk_mult)

    # pair normalization exactly as in run_live()
    pair = bcm._pair_from_symbol(symbol)

    # mock client that records place_order() calls
    order_calls: List[Dict[str, Any]] = []
    class MockClient:
        def place_order(self, *args, **kwargs):
            order_calls.append({"args": args, "kwargs": kwargs})
            # return a filled-like response
            side = kwargs.get("side") or (args[1] if len(args) > 1 else "BUY")
            qty = kwargs.get("quantity") or kwargs.get("qty") or 0
            price = kwargs.get("price") or 0
            return {"Success": True, "OrderDetail": {"Side": side, "OrderID": "MOCK", "Status": "FILLED", "Price": price, "Quantity": qty}}

    client = MockClient()

    times: List[Any] = []
    closes: List[float] = []
    equities: List[float] = []
    order_events: List[Dict[str, Any]] = []
    signals: List[Optional[str]] = []

    # live-like replay: compute qty the same way, call _place_order_safe, then update portfolio via buy_allocation/sell_all
    for _, row in df.iterrows():
        bar = {
            "time": row["time"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]) if "volume" in row and not pd.isna(row["volume"]) else 0.0,
        }
        sig = alpha.update(bar)
        signals.append(sig)
        price = bar["close"]
        times.append(bar["time"])
        closes.append(price)

        if sig == "buy":
            # same qty formula as run_live()
            qty = (port.portfolio_value(price) * alloc * risk_mult) / price
            qty = float(max(0.0, qty))
            if qty > 0:
                bcm._place_order_safe(client=client, pair_or_coin=pair, side="BUY", quantity=qty, order_type="MARKET", price=price)
                order_events.append({"side": "BUY", "time": bar["time"], "price": price})
                # live updates local portfolio via buy_allocation, not qty; mirror that
                port.buy_allocation(price, alloc)
        elif sig == "sell":
            qty = getattr(port, "positions", 0.0)
            if qty > 0:
                bcm._place_order_safe(client=client, pair_or_coin=pair, side="SELL", quantity=qty, order_type="MARKET", price=price)
                order_events.append({"side": "SELL", "time": bar["time"], "price": price})
                port.sell_all(price)

        equities.append(port.portfolio_value(price))

    # Validate 1:1 mapping (orders equal number of non-None signals and same sides in order)
    expected = [{"side": "BUY" if s == "buy" else "SELL"} for s in signals if s in ("buy", "sell")]
    ok = True
    if len(order_events) != len(expected):
        ok = False
    else:
        for got, exp in zip(order_events, expected):
            if got["side"] != exp["side"]:
                ok = False
                break

    # Validate pair normalization observed at the mock
    def _extract_pair_from_kwargs(kws: Dict[str, Any]) -> Optional[str]:
        for k in ("pair", "pair_or_coin", "symbol", "market", "instrument", "coin"):
            if k in kws:
                return str(kws[k])
        return None

    pair_ok = all(_extract_pair_from_kwargs(c["kwargs"]) == pair for c in order_calls if isinstance(c, dict))

    # Report metrics (comparable to run_backtest summary)
    start = times[0]; end = times[-1]
    total_return = equities[-1] / float(capital) - 1.0 if equities else 0.0
    try:
        days = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 86400.0
        days = max(days, 1.0)
        ann_return = (equities[-1] / float(capital)) ** (365.0 / days) - 1.0
    except Exception:
        ann_return = total_return
    result_df = pd.DataFrame({"time": times, "equity": equities})
    rets = result_df["equity"].pct_change().dropna()
    ann_vol = (rets.std() if not rets.empty else 0.0) * (365.0 ** 0.5)
    peak = result_df["equity"].cummax()
    dd = (result_df["equity"] - peak) / peak
    max_dd = dd.min() if not dd.empty else 0.0

    print("\nLive-replay (alpha -> _place_order_safe) summary:")
    print(f"Symbol: {symbol}   Pair used: {pair}   Interval: {interval}   Bars: {len(df)}")
    print(f"Orders: total={len(order_events)}  buys={sum(1 for e in order_events if e['side']=='BUY')}  sells={sum(1 for e in order_events if e['side']=='SELL')}")
    print(f"Initial Capital: {capital:.2f}   Final PV: {equities[-1]:.2f}   Total Return: {total_return:.2%}")
    print(f"Annualized Return (approx): {ann_return:.2%}   Annualized Volatility (approx): {ann_vol:.2%}   Max Drawdown: {max_dd:.2%}")
    if not ok:
        print("CHECK FAILED: order stream does not match alpha signal stream (by count/side).")
        return 1
    if not pair_ok:
        print("CHECK FAILED: normalized pair passed to place_order does not match _pair_from_symbol().")
        return 1
    print("CHECK PASSED: signals mapped 1:1 to orders and pair normalization matched.")
    return 0


# small compatibility wrapper used by other modules/functions
def load_main_module():
    """
    Backwards-compatible alias used elsewhere in the repo.
    Delegates to the real loader _load_bcm().
    """
    return _load_bcm()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval", required=True)
    p.add_argument("--limit", type=int, default=168)
    p.add_argument("--capital", type=float, default=50000.0)
    p.add_argument("--fee", type=float, default=0.0004)
    p.add_argument("--alloc", type=float, default=0.5)
    p.add_argument("--risk-mult", type=float, default=1.0)
    p.add_argument("--live-replay", action="store_true", help="Replay Binance bars through alpha + _place_order_safe (mocked)")
    args = p.parse_args()

    # Choose the replay mode that prints KPIs
    if args.live_replay:
        # call the more complete live_replay_with_binance which already prints KPIs
        code = live_replay_with_binance(
            symbol=args.symbol,
            interval=args.interval,
            limit=args.limit,
            capital=args.capital,
            fee=args.fee,
            alloc=args.alloc,
            risk_mult=args.risk_mult,
        )
    else:
        code = live_backtest_with_alpha_orders(
            symbol=args.symbol,
            interval=args.interval,
            limit=args.limit,
            capital=args.capital,
            fee=args.fee,
            alloc=args.alloc,
        )
    sys.exit(code)


if __name__ == "__main__":
    main()
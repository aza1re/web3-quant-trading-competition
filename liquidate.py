#!/usr/bin/env python3
"""
Roostoo account status and quick liquidator.

Examples:
  python3 liquidate.py --apikey X --api-secret Y --status
  python3 liquidate.py --apikey X --api-secret Y --status --sell BTC
  python3 liquidate.py --apikey X --api-secret Y --sell BTC --confirm
"""

import os
import sys
import argparse
import json
import importlib.util
import time
from datetime import datetime, timezone

# Ensure repo root and utils/ on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (REPO_ROOT, PARENT):
    if p not in sys.path:
        sys.path.append(p)

# Import Roostoo client
try:
    from rostoo import RoostooClient, BASE_URL  # prefer the same client main.py uses
except ModuleNotFoundError:
    from utils.rostoo import RoostooClient, BASE_URL  # fallback

# Try to import helpers from btc_converted/main.py (no __init__ required)
_place_order_safe = None
_conv_pair = None
_round_qty = None
try:
    from btc_converted.main import _place_order_safe as _pos_main, _pair_from_symbol as _pair_main, _round_qty as _round_main
    _place_order_safe = _pos_main
    _conv_pair = _pair_main
    _round_qty = _round_main
except Exception:
    # Load by path if package import fails
    try:
        main_path = os.path.join(REPO_ROOT, "btc_converted", "main.py")
        spec = importlib.util.spec_from_file_location("btc_converted_main", main_path)
        mod = importlib.util.module_from_spec(spec) if spec else None
        if spec and spec.loader:
            spec.loader.exec_module(mod)  # type: ignore
            _place_order_safe = getattr(mod, "_place_order_safe", None)
            _conv_pair = getattr(mod, "_pair_from_symbol", None)
            _round_qty = getattr(mod, "_round_qty", None)
    except Exception:
        pass

# Local fallbacks if not found
def _pair_from_symbol_local(symbol: str, quote: str = "USD") -> str:
    s = (symbol or "").upper().replace(" ", "")
    if '/' in s:
        return s
    if s.endswith('USDT'):
        base = s[:-4]
    elif s.endswith('USD'):
        base = s[:-3]
    else:
        base = s
    return f"{base}/{quote}"

def _round_qty_local(_sym_or_pair: str, qty: float) -> float:
    try:
        q = float(qty)
        return 0.0 if q <= 0 else q
    except Exception:
        return 0.0

# Pick the available helpers
_pair_from_symbol = _conv_pair or _pair_from_symbol_local
_round_qty = _round_qty or _round_qty_local


def _extract_balances(resp) -> dict:
    """
    Normalize Roostoo balance shapes into:
      { 'BTC': {'free': float, 'locked': float, 'total': float}, ... }
    Supports keys: Wallet, SpotWallet, MarginWallet.
    """
    out: dict = {}
    if not isinstance(resp, dict):
        return out

    # Collect possible wallet containers
    wallet_objs = []
    for k in ["Wallet", "SpotWallet", "MarginWallet"]:
        w = resp.get(k) or resp.get("Data", {}).get(k)
        if isinstance(w, dict):
            wallet_objs.append(w)

    # Parse each wallet object
    for w in wallet_objs:
        for asset, v in w.items():
            if not isinstance(v, dict):
                continue
            try:
                free = float(v.get("Free", 0) or 0)
                locked = float(v.get("Lock", v.get("Locked", 0) or 0))
                out[asset.upper()] = {
                    "free": free,
                    "locked": locked,
                    "total": free + locked
                }
            except Exception:
                continue
    if out:
        return out

    # Legacy shapes (fallback)
    data = resp.get("Data")
    if isinstance(data, dict) and "Balances" in data and isinstance(data["Balances"], (list, tuple)):
        for b in data["Balances"]:
            asset = (b.get("Asset") or b.get("Coin") or b.get("Symbol") or "").upper()
            if not asset:
                continue
            try:
                free = float(b.get("Free", 0) or 0)
                locked = float(b.get("Locked", b.get("Lock", 0) or 0))
                out[asset] = {"free": free, "locked": locked, "total": free + locked}
            except Exception:
                continue
        if out:
            return out

    # Flat numeric map fallback
    for k, v in resp.items():
        if isinstance(v, (int, float, str)):
            try:
                amt = float(v)
                out[k.upper()] = {"free": amt, "locked": 0.0, "total": amt}
            except Exception:
                continue
    return out


def _summarize_resp(r):
    try:
        return json.dumps(r, default=str)[:4000]
    except Exception:
        return repr(r)


def _fetch_prices_map(client) -> dict:
    tick = client.ticker()
    data = (tick or {}).get("Data", {})
    prices = {}
    for pair, obj in data.items():
        try:
            prices[pair.upper()] = float(obj.get("LastPrice"))
        except Exception:
            continue
    return prices


def _price_for_asset_usd(asset: str, prices: dict) -> float:
    if asset.upper() == "USD":
        return 1.0
    pair = f"{asset.upper()}/USD"
    return float(prices.get(pair, 0.0) or 0.0)


def _print_status(client: RoostooClient, balances: dict, focus_symbol: str | None):
    prices = _fetch_prices_map(client)

    print("\nPortfolio:")
    print("Asset   Free            Locked          Total           Px(USD)        Value(USD)")
    print("------  --------------  --------------  --------------  -------------  -------------")

    total_equity = 0.0
    nonzero_assets = 0
    for asset in sorted(balances.keys()):
        b = balances[asset]
        if b["total"] == 0:
            continue
        px = _price_for_asset_usd(asset, prices)
        val = b["total"] * (px if px > 0 else 1.0 if asset == "USD" else 0.0)
        total_equity += val
        nonzero_assets += 1
        print(f"{asset:6s}  {b['free']:14.8f}  {b['locked']:14.8f}  {b['total']:14.8f}  {px:13.6f}  {val:13.2f}")

    if nonzero_assets == 0:
        print("(no non-zero assets)")

    print(f"\nEstimated total equity (USD): {total_equity:,.2f}")

    # Pending orders summary
    pending = client.pending_count()
    print("\nPending orders summary:")
    if pending and pending.get("Success") and pending.get("TotalPending", 0) > 0:
        count = pending.get("TotalPending", 0)
        pairs = pending.get("OrderPairs", {}) or {}
        print(f"TotalPending: {count}")
        for p, c in pairs.items():
            print(f"  {p}: {c}")
    else:
        if pending and not pending.get("Success"):
            print(f"None (API says: {pending.get('ErrMsg', 'no pending order')})")
        else:
            print("None")

    if focus_symbol:
        pair = _pair_from_symbol(focus_symbol)
        base = pair.split('/')[0].upper()
        b = balances.get(base, {"free": 0.0, "locked": 0.0, "total": 0.0})
        px = _price_for_asset_usd(base, prices)
        val = b["total"] * (px if px > 0 else 0.0)
        print(f"\nFocus {base} ({pair}): free={b['free']:.8f} locked={b['locked']:.8f} total={b['total']:.8f} px={px:.6f} valueUSD={val:.2f}")


def _print_past_orders(client: RoostooClient, pair: str | None):
    """
    Print past (non-pending) orders only. No other output.
    If pair is provided, query that pair; otherwise query all.
    """
    try:
        resp = client.query_order(pair=pair, pending_only=False)
    except Exception:
        resp = None

    if not resp or not resp.get("Success"):
        return

    orders = resp.get("OrderMatched", []) or []
    tz_utc = timezone.utc  # portable UTC
    for o in orders:
        status = str(o.get("Status", "")).upper()
        if status == "PENDING":
            continue
        ct = o.get("CreateTimestamp", 0) or 0
        ft = o.get("FinishTimestamp", 0) or 0
        # Use timezone-aware UTC; avoids deprecated utcfromtimestamp and missing datetime.UTC
        ct_s = datetime.fromtimestamp(ct / 1000, tz_utc).isoformat() if ct else ""
        ft_s = datetime.fromtimestamp(ft / 1000, tz_utc).isoformat() if ft else ""
        print(
            f"OrderID={o.get('OrderID','')} Pair={o.get('Pair','')} "
            f"Side={o.get('Side','')} Type={o.get('Type','')} Status={status} "
            f"Price={o.get('Price','')} Qty={o.get('Quantity','')} "
            f"FilledQty={o.get('FilledQuantity','')} AvgPx={o.get('FilledAverPrice','')} "
            f"Created={ct_s} Finished={ft_s}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apikey", required=True)
    ap.add_argument("--api-secret", required=True)
    ap.add_argument("--sell", help="Symbol to sell (BTC, BTCUSDT, BTC/USD etc.)")
    ap.add_argument("--pair", help="Exact pair to use for status filter or selling (e.g., ETH/USD)")
    ap.add_argument("--confirm", action="store_true", help="Actually submit sell order (otherwise dry-run)")
    ap.add_argument("--status", action="store_true", help="Print current portfolio (balances + prices + pending summary)")
    try:
        ap.add_argument("--cancel-pending", action=argparse.BooleanOptionalAction, default=True,
                        help="Auto-cancel pending orders for the pair when asset is locked (default: enabled)")
    except Exception:
        ap.add_argument("--cancel-pending", action="store_true",
                        help="Auto-cancel pending orders for the pair when asset is locked (enable flag)")
    args = ap.parse_args()

    client = RoostooClient(api_key=args.apikey, secret_key=args.api_secret)

    # STATUS MODE: print current portfolio and optional focus asset, then exit
    if args.status:
        bal_resp = client.balance()
        if not bal_resp:
            print("ERROR: no response from /v3/balance")
            return
        balances = _extract_balances(bal_resp)
        _print_status(client, balances, args.sell)
        return

    # SELL FLOW -----------------------------------------------------------
    if not args.sell and not args.pair:
        return

    # Fetch balances once we know we need to sell
    print("Fetching balances ...")
    bal_resp = client.balance()
    if not bal_resp:
        print("ERROR: no response from /v3/balance")
        sys.exit(2)
    balances = _extract_balances(bal_resp)
    if args.sell:
        dbg_asset = (_pair_from_symbol(args.sell).split('/')[0]).upper()
        hold_dbg = balances.get(dbg_asset)
        if hold_dbg is None:
            print(f"[DEBUG] {dbg_asset} not found in parsed balances. Raw (truncated):")
            print(json.dumps(bal_resp, default=str)[:600])

    # Normalize pair and find free/locked qty
    pair = args.pair or _pair_from_symbol(args.sell)
    base = pair.split('/')[0].upper()
    holding = balances.get(base, {"free": 0.0, "locked": 0.0, "total": 0.0})
    free_qty = float(holding.get("free", 0.0) or 0.0)
    locked_qty = float(holding.get("locked", 0.0) or 0.0)

    # If locked exists and no free, optionally cancel pending and wait for unlock
    if locked_qty > 0 and free_qty <= 0:
        print(f"Notice: {base} locked={locked_qty:.8f}, free={free_qty:.8f}.")
        do_cancel = getattr(args, "cancel_pending", False)
        if do_cancel:
            print(f"Auto-canceling pending orders for {pair} to free locked funds...")
            cancel_resp = client.cancel_order(pair=pair)
            # No extra prints in status mode; here we can show result
            print("Cancel response:", _summarize_resp(cancel_resp))

            # Wait for unlock up to 20s
            deadline = time.time() + 20
            last_free = free_qty
            while time.time() < deadline:
                time.sleep(1.0)
                ref = client.balance() or {}
                balances = _extract_balances(ref)
                holding = balances.get(base, {"free": 0.0, "locked": 0.0, "total": 0.0})
                free_qty = float(holding.get("free", 0.0) or 0.0)
                locked_qty = float(holding.get("locked", 0.0) or 0.0)
                if free_qty > 0 or locked_qty == 0:
                    break
                if abs(free_qty - last_free) > 1e-12:
                    print(f"Waiting unlock... free={free_qty:.8f} locked={locked_qty:.8f}")
                    last_free = free_qty
            print(f"Post-cancel free={free_qty:.8f} locked={locked_qty:.8f}")
        else:
            print("Locked funds detected and --cancel-pending is disabled. Use --cancel-pending or cancel manually.")
            sys.exit(1)

    # Only sell free portion (cannot sell locked)
    qty = _round_qty(pair, free_qty)

    if qty <= 0:
        if locked_qty > 0:
            print(f"Abort: no free {base} to sell; locked={locked_qty:.8f}. Use --cancel-pending to unlock.")
        else:
            print(f"No free {base} available to sell. Free={free_qty:.8f} locked={locked_qty:.8f}")
        sys.exit(1)

    print(f"Preparing to SELL {qty:.8f} {base} on {pair}. Dry-run={not args.confirm}")
    if not args.confirm:
        print("Dry-run. Add --confirm to execute.")
        return

    # Use shared safe order helper
    try:
        if _place_order_safe is None:
            raise RuntimeError("helper _place_order_safe not available")
        resp = _place_order_safe(client, pair, "SELL", qty)
    except Exception:
        try:
            resp = client.place_order(pair_or_coin=pair, side="SELL", quantity=str(qty), order_type="MARKET")
        except TypeError:
            resp = client.place_order(pair=pair, side="SELL", quantity=str(qty), type="MARKET")

    print("Order response:")
    print(_summarize_resp(resp))


if __name__ == "__main__":
    main()
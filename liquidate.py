#!/usr/bin/env python3
"""
Roostoo account status and quick liquidator.

Examples:
  python3 liquidate.py --apikey X --api-secret Y --status
  python3 liquidate.py --apikey X --api-secret Y --status --sell BTC
  python3 liquidate.py --apikey X --api-secret Y --sell ETH --confirm --auto-topup
"""
import os
import sys
import time
import json
import math
import argparse
from typing import Any, Dict, Optional

# repo root
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "btc_converted"), os.path.join(REPO_ROOT, "utils")):
    if p not in sys.path:
        sys.path.append(p)

from utils.rostoo import RoostooClient

# Try to reuse helpers from live runner if present
_pair_from_symbol = None
_qty_step = None
_round_qty = None
try:
    from btc_converted.main import _pair_from_symbol as _pair_from_symbol  # type: ignore
    from btc_converted.main import _qty_step as _qty_step                  # type: ignore
    from btc_converted.main import _round_qty as _round_qty                # type: ignore
except Exception:
    pass

def pair_from_asset(asset: str) -> str:
    s = (asset or "").upper().replace(" ", "")
    if _pair_from_symbol:
        try:
            return _pair_from_symbol(s)
        except Exception:
            pass
    if "/" in s:
        return s
    if s.endswith("USDT"):
        base = s[:-4]
    elif s.endswith("USD"):
        base = s[:-3]
    else:
        base = s
    return f"{base}/USD"

def qty_step_for_pair(pair: str, override: Optional[float] = None) -> float:
    if override is not None and override > 0:
        return float(override)
    if _qty_step:
        try:
            return float(_qty_step(pair))
        except Exception:
            pass
    base = (pair.upper().split("/")[0]) if "/" in pair else pair.upper()
    if base == "ETH":
        return 0.001
    if base == "BTC":
        return 0.00001
    if base == "TRX":
        return 1.0
    return 0.000001

def round_to_step(qty: float, step: float) -> float:
    if _round_qty:
        try:
            return float(_round_qty("GEN", float(qty)))
        except Exception:
            pass
    if step <= 0:
        return max(0.0, float(qty))
    return math.floor(max(0.0, float(qty)) / step) * step

def _extract_balances(resp: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not isinstance(resp, dict):
        return out
    wallet_objs = []
    for k in ["Wallet", "SpotWallet", "MarginWallet"]:
        w = resp.get(k) or resp.get("Data", {}).get(k)
        if isinstance(w, dict):
            wallet_objs.append(w)
    for w in wallet_objs:
        for asset, v in w.items():
            if not isinstance(v, dict):
                continue
            try:
                free = float(v.get("Free", 0) or 0)
                locked = float(v.get("Lock", v.get("Locked", 0) or 0))
                out[asset.upper()] = {"free": free, "locked": locked, "total": free + locked}
            except Exception:
                continue
    return out

def print_status(client: RoostooClient, focus: Optional[str] = None):
    print("Fetching balances ...")
    bal = client.balance() or {}
    bmap = _extract_balances(bal)
    if focus:
        sym = focus.upper()
        hold = bmap.get(sym, {"free": 0.0, "locked": 0.0, "total": 0.0})
        print(f"{sym}: free={hold.get('free', 0.0):.8f} locked={hold.get('locked', 0.0):.8f} total={hold.get('total', 0.0):.8f}")
    else:
        print(json.dumps(bmap, indent=2, sort_keys=True))

def sell_all_of_asset(client: RoostooClient,
                      asset: str,
                      confirm: bool,
                      auto_topup: bool,
                      min_step_override: Optional[float] = None,
                      pair_override: Optional[str] = None) -> int:
    pair = pair_override or pair_from_asset(asset)
    base = pair.split("/")[0].upper()
    bal = client.balance() or {}
    bmap = _extract_balances(bal)
    hold = bmap.get(base, {"free": 0.0, "locked": 0.0, "total": 0.0})
    free = float(hold.get("free", 0.0) or 0.0)
    locked = float(hold.get("locked", 0.0) or 0.0)

    step = qty_step_for_pair(pair, min_step_override)
    sell_qty = round_to_step(free, step)

    print(f"Detected {base}: free={free:.8f} locked={locked:.8f} step={step:g} pair={pair}")
    if free <= 0.0:
        print(f"No free {base} available to sell.")
        return 0

    if sell_qty >= step:
        print(f"Action: SELL {sell_qty:.8f} {base} at market. Dry-run={not confirm}")
        if confirm:
            resp = client.place_order(pair_or_coin=pair, side="SELL", quantity=str(sell_qty), order_type="MARKET")
            print("[SELL RESP]", json.dumps(resp, default=str)[:800])
        return 0

    topup = step - free
    topup = max(0.0, round(topup, 9))
    msg = f"Free {base} below step. Need top-up: {topup:.8f} to reach {step:g}"
    if not auto_topup:
        print(msg + " (add --auto-topup to buy the difference then sell)")
        return 1

    print(msg + f". Proceeding. Dry-run={not confirm}")
    if not confirm:
        return 0

    buy_resp = client.place_order(pair_or_coin=pair, side="BUY", quantity=str(topup), order_type="MARKET")
    print("[TOP-UP BUY RESP]", json.dumps(buy_resp, default=str)[:800])
    time.sleep(1.0)

    bal2 = client.balance() or {}
    bmap2 = _extract_balances(bal2)
    free2 = float((bmap2.get(base, {}).get("free", 0.0)) or 0.0)
    sell2 = round_to_step(free2, step)
    if sell2 < step:
        print(f"After top-up still below step. free={free2:.8f} rounded={sell2:.8f}. Aborting.")
        return 2
    if abs(sell2 - step) < 1e-12:
        sell2 = step
    sell_resp = client.place_order(pair_or_coin=pair, side="SELL", quantity=str(sell2), order_type="MARKET")
    print("[FINAL SELL RESP]", json.dumps(sell_resp, default=str)[:800])
    return 0

def main():
    ap = argparse.ArgumentParser(description="Roostoo account status and liquidator (with dust auto-topup).")
    ap.add_argument("--apikey", required=True)
    ap.add_argument("--api-secret", required=True)
    ap.add_argument("--status", action="store_true", help="Print balances/status")
    ap.add_argument("--sell", metavar="ASSET", help="Sell entire free balance of ASSET (e.g., ETH)")
    ap.add_argument("--pair", help="Override pair (e.g., ETH/USD). Default inferred ASSET/USD")
    ap.add_argument("--min-step", type=float, default=None, help="Override minimum quantity step (e.g., 0.001)")
    ap.add_argument("--auto-topup", action="store_true", help="If free < step, buy missing amount then sell")
    ap.add_argument("--confirm", action="store_true", help="Execute (else dry-run)")
    args = ap.parse_args()

    client = RoostooClient(api_key=args.apikey, secret_key=args.api_secret)

    if args.status and not args.sell:
        print_status(client)
        return

    if args.sell:
        rc = sell_all_of_asset(
            client=client,
            asset=args.sell,
            confirm=args.confirm,
            auto_topup=args.auto_topup,
            min_step_override=args.min_step,
            pair_override=args.pair,
        )
        if rc != 0:
            sys.exit(rc)
        return

    print_status(client)

if __name__ == "__main__":
    main()
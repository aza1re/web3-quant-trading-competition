import os, sys

# ensure repo root on path to import shared modules (binance, horus, etc.)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from typing import List, Optional
# use relative import for alpha inside the package
from .alpha import HybridAlphaConverted

from binance import fetch_klines as fetch_binance_klines
from horus import fetch_klines as fetch_horus_klines

import argparse
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import List, Optional
from rostoo import RoostooClient, BASE_URL
import traceback
from datetime import datetime
import math

# --- NEW: BarAggregator for interval-based OHLC synthesis ---
class BarAggregator:
    """
    Aggregates tick/last prices into OHLC bars for a given interval string: 15m | 1h | 1d
    update(timestamp, price) -> returns a completed bar dict or None while accumulating.
    """
    def __init__(self, interval: str):
        self.interval = interval
        if interval not in ("15m", "1h", "1d"):
            raise ValueError("BarAggregator interval must be one of 15m,1h,1d")
        self.current_start = None
        self.open = self.high = self.low = self.close = None
        self.volume = 0.0  # placeholder (no real volume from ticker)
        self._secs = 900 if interval == "15m" else 3600 if interval == "1h" else 86400

    def _floor_ts(self, ts: pd.Timestamp):
        epoch = int(ts.timestamp())
        floored = (epoch // self._secs) * self._secs
        return pd.to_datetime(floored, unit="s", utc=True)

    def update(self, ts: pd.Timestamp, price: float, vol: float = 0.0):
        """
        Supply current timestamp (UTC) + last price.
        Returns a finished bar when a new interval window begins, else None.
        """
        if price is None:
            return None
        ts_floor = self._floor_ts(ts)

        # first bar initialize
        if self.current_start is None:
            self.current_start = ts_floor
            self.open = self.high = self.low = self.close = float(price)
            self.volume = float(vol or 0.0)
            return None

        # same interval window -> update running bar
        if ts_floor == self.current_start:
            p = float(price)
            if p > self.high: self.high = p
            if p < self.low: self.low = p
            self.close = p
            self.volume += float(vol or 0.0)
            return None

        # new interval -> finalize previous, start new
        finished = {
            "time": self.current_start,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        }
        # start new bar
        self.current_start = ts_floor
        p = float(price)
        self.open = self.high = self.low = self.close = p
        self.volume = float(vol or 0.0)
        return finished

class SimplePortfolio:
    def __init__(self, cash=100000.0, fee=0.0001, risk_mult=1.0):
        self.cash = float(cash)
        self.fee = float(fee)
        self.positions = 0.0
        self.avg_price = None
        self.trade_count = 0
        self.risk_mult = float(risk_mult)
        self.trade_sizes = []   # track qty per trade (buys and sells)
        self.trade_values = []  # track USD value per trade (cost or proceeds)

    def buy_allocation(self, price, allocation):  # allocation = fraction of portfolio value
        pv = self.portfolio_value(price)
        target_value = pv * allocation * self.risk_mult
        if target_value <= 0:
            return
        # Prevent negative cash: cap target by available cash (including fees)
        max_affordable_value = self.cash / (1.0 + self.fee) if self.cash > 0 else 0.0
        trade_value = min(target_value, max_affordable_value)
        if trade_value <= 0:
            return
        qty = trade_value / price
        cost = qty * price
        fee = abs(cost) * self.fee
        # apply cost + fee and record trade
        self.cash -= (cost + fee)
        self.trade_sizes.append(qty)
        self.trade_values.append(cost)

        prev_qty = self.positions
        if prev_qty == 0 or self.avg_price is None:
            self.avg_price = price
        else:
            self.avg_price = (self.avg_price * prev_qty + cost) / (prev_qty + qty)
        self.positions += qty
        self.trade_count += 1

    def sell_all(self, price):
        proceeds = self.positions * price
        fee = abs(proceeds) * self.fee
        self.cash += (proceeds - fee)
        # record sell as a trade
        if self.positions > 0:
            self.trade_sizes.append(self.positions)
            self.trade_values.append(proceeds)
        self.positions = 0.0
        self.avg_price = None
        self.trade_count += 1

    def portfolio_value(self, price):
        return self.cash + (self.positions * price)

# new helper: fetch live Roostoo price for a pair
def _pair_from_symbol(symbol: str) -> str:
    s = symbol.upper()
    if '/' in s:
        return s
    if s.endswith('USDT'):
        base = s[:-4]
    elif s.endswith('USD'):
        base = s[:-3]
    else:
        base = s
    return f"{base}/USD"

def fetch_roostoo_ticker(pair: str) -> Optional[float]:
    ts = str(int(time.time() * 1000))
    try:
        resp = requests.get(f"{BASE_URL}/v3/ticker", params={'timestamp': ts, 'pair': pair}, timeout=10)
        resp.raise_for_status()
        j = resp.json()
        # docs: Data->{pair}->{LastPrice}
        data = j.get("Data", {})
        p = data.get(pair, {}) or {}
        last = p.get("LastPrice") or p.get("Last") or p.get("LastPrice")
        if last is None:
            # sometimes api returns single flat response
            return None
        return float(last)
    except Exception:
        return None

def fetch_roostoo_tickers() -> dict:
    """Fetch all tickers once; return map {pair: last_price}."""
    ts = str(int(time.time() * 1000))
    try:
        resp = requests.get(f"{BASE_URL}/v3/ticker", params={'timestamp': ts}, timeout=10)
        resp.raise_for_status()
        j = resp.json()
        data = j.get("Data", {}) or {}
        out = {}
        for k, v in data.items():
            last = v.get("LastPrice") or v.get("Last")
            if last is not None:
                out[k] = float(last)
        return out
    except Exception:
        return {}

def run_backtest(symbol: str,
                 interval: str,
                 limit: int,
                 apikey: Optional[str] = None,
                 source: str = "horus",
                 capital: float = 100000.0,
                 fee: float = 0.0001,
                 risk_mult: float = 1.0,
                 verbose: bool = False):
    print(f"Fetching {symbol} from {source.capitalize()} ({interval}, {limit})...")
    if source.lower() == "binance":
        df = fetch_binance_klines(symbol=symbol, interval=interval, limit=limit)
    else:
        df = fetch_horus_klines(symbol=symbol, interval=interval, limit=limit, api_key=apikey)
    if df is None or df.empty:
        print("No data returned from Horus.")
        return

    df = df.sort_values('open_time').reset_index(drop=True)
    # normalize column names
    df.rename(columns={'open_time': 'time'}, inplace=True)

    # relaxed thresholds for testing (more sensitive -> more trades)
    alpha = HybridAlphaConverted(volume_period=5, atr_period=10, momentum_period=3,
                                 volume_multiplier=1.0,  # allow equal-volume bars to count
                                 atr_multiplier=1.0,
                                 stop_loss_pct=0.05)
    port = SimplePortfolio(cash=capital, fee=fee, risk_mult=risk_mult)

    equity = []
    times = []
    prices = []
    buys = []
    sells = []

    for i, row in df.iterrows():
        bar = {
            'time': row['time'],
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']) if 'volume' in row and not pd.isna(row['volume']) else 0.0
        }
        sig = alpha.update(bar)
        price = bar['close']
        times.append(bar['time'])
        prices.append(price)

        # simple allocation: use 50% of portfolio on entry (can be adjusted via risk_mult)
        alloc = 0.5
        if sig == 'buy':
            port.buy_allocation(price, alloc)
            buys.append((bar['time'], price))
            if verbose:
                print(f"[BUY] time={bar['time']} price={price:.2f} cash={port.cash:.2f}")
        elif sig == 'sell':
            port.sell_all(price)
            sells.append((bar['time'], price))
            if verbose:
                print(f"[SELL] time={bar['time']} price={price:.2f} cash={port.cash:.2f}")

        equity.append(port.portfolio_value(price))

        if verbose:
            # compute same rolling metrics used by the alpha for quick visibility
            window = df.loc[max(0, i-10):i]  # last ~11 bars
            vol_ma = window['volume'].tail(5).mean() if len(window)>=5 else float('nan')
            momentum3 = (price - window['close'].shift(3).iloc[-1]) / window['close'].shift(3).iloc[-1] if len(window)>3 else float('nan')
            print(f"{bar['time']} price={price:.2f} vol={bar['volume']:.0f} vol_ma5={vol_ma:.2f} mom3={momentum3:.4f} entry_price={alpha.entry_price}")

    # ensure final liquidation for reporting
    if port.positions > 0:
        port.sell_all(price)
        if verbose:
            print(f"[FINAL LIQUIDATE] price={price:.2f} cash={port.cash:.2f}")
        equity[-1] = port.portfolio_value(price)

    # build results
    result_df = pd.DataFrame({
        'time': times,
        'close': prices,
        'equity': equity
    })

    # metrics
    start = result_df['time'].iloc[0]
    end = result_df['time'].iloc[-1]
    total_return = equity[-1] / capital - 1.0
    # safer annualization: use CAGR based on elapsed days
    try:
        days = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 86400.0
        days = max(days, 1.0)
        ann_return = (equity[-1] / capital) ** (365.0 / days) - 1.0
    except Exception:
        ann_return = total_return
    rets = result_df['equity'].pct_change().dropna()
    vol = rets.std() if not rets.empty else 0.0
    ann_vol = vol * (365.0 ** 0.5)

    peak = result_df['equity'].cummax()
    drawdown = (result_df['equity'] - peak) / peak
    max_dd = drawdown.min() if not drawdown.empty else 0.0

    # trades per day
    total_trades = port.trade_count
    try:
        days = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 86400.0
        days = max(days, 1.0)
        avg_trades_per_day = total_trades / days
    except Exception:
        avg_trades_per_day = float('nan')

    print("\nArbitrage-like Backtest (converted):")
    print(f"Symbol: {symbol}")
    print(f"Period: {start} to {end}")
    print(f"Initial Capital: {capital:.2f}")
    print(f"Final Portfolio Value: {equity[-1]:.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Total Trades: {total_trades}")
    print(f"Avg Trades / Day: {avg_trades_per_day:.2f}")
    print(f"Annualized Return (approx): {ann_return:.2%}")
    print(f"Annualized Volatility (approx): {ann_vol:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")

    # report average trade size/value
    try:
        import numpy as _np
        avg_qty = _np.mean(port.trade_sizes) if len(port.trade_sizes) else 0.0
        avg_value = _np.mean(port.trade_values) if len(port.trade_values) else 0.0
        print(f"Avg trade qty: {avg_qty:.8f}   Avg trade value (USD): {avg_value:.2f}")
    except Exception:
        pass

    # plot price + buy/sell markers and equity
    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios':[2,1]})
        axes[0].plot(result_df['time'], result_df['close'], label=f'{symbol} Close', color='blue')
        if buys:
            b_times, b_prices = zip(*buys)
            axes[0].scatter(b_times, b_prices, marker='^', color='green', label='Buy', zorder=5)
        if sells:
            s_times, s_prices = zip(*sells)
            axes[0].scatter(s_times, s_prices, marker='v', color='red', label='Sell', zorder=5)
        axes[0].legend(); axes[0].grid(True); axes[0].set_title(f'{symbol} Price & Signals')

        axes[1].plot(result_df['time'], result_df['equity'], label='Equity', color='green')
        axes[1].set_title('Equity Curve'); axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout(); plt.show()
    except Exception:
        pass

# new: live deploy runner using Roostoo
def run_live(symbol: str,
             interval: str,
             apikey: str,
             apisecret: str,
             capital: float,
             fee: float,
             risk_mult: float,
             alloc: float,
             verbose: bool,
             force: bool,
             do_check: bool = False,
             vol_mult: float = 1.0,
             atr_mult: float = 1.0,
             mom_period: int = 3,
             stop_loss: float = 0.05,
             momentum_epsilon: float = 0.0005,
             entry_epsilon: float = 0.0008,
             exit_epsilon: float = 0.0005,
             cooldown_bars: int = 1,
             atr_mom_mult: float = 0.0,
             min_hold_bars: int = 2,
             take_profit_pct: float = 0.01,
             max_open_symbols: int = 1,
             max_daily_dd_pct: float = 0.05,
             debug_signals: bool = False,
             poll_secs: Optional[int] = None):
    pair = _pair_from_symbol(symbol)
    client = RoostooClient(api_key=apikey, secret_key=apisecret)
    alpha = HybridAlphaConverted(volume_period=5,
                                 atr_period=10,
                                 momentum_period=mom_period,
                                 volume_multiplier=vol_mult,
                                 atr_multiplier=atr_mult,
                                 stop_loss_pct=stop_loss,
                                 momentum_epsilon=momentum_epsilon,
                                 entry_epsilon=entry_epsilon,
                                 exit_epsilon=exit_epsilon,
                                 cooldown_bars=cooldown_bars,
                                 atr_mom_mult=atr_mom_mult,
                                 min_hold_bars=min_hold_bars,
                                 take_profit_pct=take_profit_pct)

    # warmup
    try:
        seed_df = fetch_binance_klines(symbol=symbol, interval=interval, limit=50)
        if seed_df is not None and not seed_df.empty:
            for _, r in seed_df.sort_values("open_time").iterrows():
                alpha.update({'time': r['open_time'], 'open': float(r['open']), 'high': float(r['high']),
                              'low': float(r['low']), 'close': float(r['close']),
                              'volume': float(r['volume']) if 'volume' in r and not pd.isna(r['volume']) else 0.0})
            if verbose: print(f"[WARMUP] seeded {len(seed_df)} bars")
    except Exception:
        pass

    port = SimplePortfolio(cash=capital, fee=fee, risk_mult=risk_mult)
    day_equity_open = {}
    last_signal = None
    last_order_resp = None
    prices_now = {}

    interval_secs = 86400 if interval == '1d' else 3600 if interval == '1h' else 900
    if poll_secs and poll_secs > 0:
        interval_secs = poll_secs
    print(f"DEPLOY SINGLE: polling {pair} every {interval_secs}s. Dry-run={not force}")

    agg = BarAggregator(interval)

    if do_check:
        test_price = fetch_roostoo_ticker(pair)
        if test_price:
            # choose smallest qty that clears min notional and step
            base_qty = (_min_notional_usd(symbol) * 1.2) / test_price
            test_qty = _ceil_to_step(symbol, base_qty)
            mode = "EXECUTING" if force else "DRY-RUN"
            print(f"[CHECK] {mode} test BUY qty={test_qty:.6f} price={test_price:.2f}")
            if force and test_qty > 0:
                resp_b = _place_order_safe(client, pair, "BUY", test_qty)
                print("[CHECK ORDER BUY]", _summarize_order_resp(resp_b))
                _print_raw_resp(resp_b, label="check-buy")
                resp_s = _place_order_safe(client, pair, "SELL", test_qty)
                print("[CHECK ORDER SELL]", _summarize_order_resp(resp_s))
                _print_raw_resp(resp_s, label="check-sell")

    try:
        while True:
            price = fetch_roostoo_ticker(pair)
            now = pd.to_datetime('now', utc=True)
            if price is None:
                if verbose: print("[WARN] no price")
                time.sleep(interval_secs); continue
            prices_now[symbol] = price
            closed_bar = agg.update(now, price)
            if closed_bar is None:
                time.sleep(interval_secs); continue
            sig = alpha.update(closed_bar)
            last_signal = sig
            if debug_signals:
                print(f"[DBG] {symbol} t={closed_bar['time']} mom={alpha.last_momentum:.6f} atr={alpha.last_current_atr:.6f} sig={sig}")

            # Log signal even if filtered
            if sig in ('buy', 'sell'):
                print(f"[SIG] {symbol} {sig} at {closed_bar['time']} px={closed_bar['close']:.6f}")

            if sig == 'buy':
                day_key = now.date()
                if day_key not in day_equity_open:
                    day_equity_open[day_key] = port.portfolio_value(price)
                cur_eq = port.portfolio_value(price)
                # risk filter
                if (cur_eq - day_equity_open[day_key]) / day_equity_open[day_key] <= -max_daily_dd_pct:
                    print(f"[FILTER] {symbol} blocked by daily DD ({max_daily_dd_pct:.2%})")
                else:
                    prev_pos = port.positions
                    port.buy_allocation(price, alloc)
                    raw_qty = port.positions - prev_pos   # incremental
                    rqty = _round_qty(symbol, raw_qty)
                    notional = rqty * price
                    if rqty <= 0:
                        print(f"[FILTER] {symbol} no cash/qty after sizing")
                    elif notional < _min_notional_usd(symbol):
                        print(f"[FILTER] {symbol} notional too small ${notional:.2f} < ${_min_notional_usd(symbol):.2f}")
                    else:
                        print(f"[LIVE] BUY {symbol} qty={rqty:.6f} price={price:.2f} total_pos={port.positions:.6f}")
                        if force:
                            resp = _place_order_safe(client, pair, "BUY", rqty)
                            last_order_resp = resp
                            print("[ORDER]", _summarize_order_resp(resp))
                            _print_raw_resp(resp, label=f"live-buy-{symbol}")
            elif sig == 'sell' and port.positions > 0:
                qty = _round_qty(symbol, port.positions)
                if qty > 0:
                    port.sell_all(price)
                    print(f"[LIVE] SELL {symbol} qty={qty:.6f} price={price:.2f}")
                    if force:
                        resp = _place_order_safe(client, pair, "SELL", qty)
                        last_order_resp = resp
                        print("[ORDER]", _summarize_order_resp(resp))
                        _print_raw_resp(resp, label=f"live-sell-{symbol}")

            _print_status(port, pair, last_signal, last_order_resp, verbose)
            time.sleep(interval_secs)
    except KeyboardInterrupt:
        print("Stopping single-symbol live.")

# --- Multi-asset live portfolio ---
class LiveMultiPortfolio:
    def __init__(self, cash=100000.0, fee=0.0001, risk_mult=1.0):
        self.cash = float(cash)
        self.fee = float(fee)
        self.risk_mult = float(risk_mult)
        self.pos = {}      # sym -> qty
        self.avgp = {}     # sym -> avg price
        self.trade_count = 0
        self.trade_sizes = []
        self.trade_values = []

    def value(self, prices: dict) -> float:
        v = self.cash
        for s, q in self.pos.items():
            p = prices.get(s)
            if p is not None:
                v += q * p
        return v

    def buy_value(self, sym: str, price: float, allocation: float, prices_now: dict) -> float:
        pv = self.value(prices_now)
        target = pv * allocation * self.risk_mult
        max_afford = self.cash / (1 + self.fee)
        trade_val = min(target, max_afford)
        if trade_val <= 0 or price <= 0:
            return 0.0
        qty = trade_val / price
        cost = qty * price
        fee = cost * self.fee
        self.cash -= (cost + fee)
        prev_qty = self.pos.get(sym, 0.0)
        prev_avg = self.avgp.get(sym)
        new_qty = prev_qty + qty
        self.avgp[sym] = price if prev_qty == 0 or prev_avg is None else (prev_avg * prev_qty + cost) / new_qty
        self.pos[sym] = new_qty
        self.trade_count += 1
        self.trade_sizes.append(qty)
        self.trade_values.append(cost)
        return qty

    def sell_all(self, sym: str, price: float):
        qty = self.pos.get(sym, 0.0)
        if qty <= 0 or price <= 0:
            return 0.0
        proceeds = qty * price
        fee = proceeds * self.fee
        self.cash += (proceeds - fee)
        self.trade_count += 1
        self.trade_sizes.append(qty)
        self.trade_values.append(proceeds)
        self.pos[sym] = 0.0
        self.avgp.pop(sym, None)
        return qty

def run_live_multi(symbols: List[str],
                   interval: str,
                   apikey: str,
                   apisecret: str,
                   capital: float,
                   fee: float,
                   risk_mult: float,
                   alloc: float,
                   verbose: bool,
                   force: bool,
                   do_check: bool = False,
                   vol_mult: float = 1.0,
                   atr_mult: float = 1.0,
                   mom_period: int = 3,
                   stop_loss: float = 0.05,
                   momentum_epsilon: float = 0.0005,
                   entry_epsilon: float = 0.0008,
                   exit_epsilon: float = 0.0005,
                   cooldown_bars: int = 1,
                   atr_mom_mult: float = 0.0,
                   min_hold_bars: int = 2,
                   take_profit_pct: float = 0.01,
                   max_open_symbols: int = 3,
                   max_daily_dd_pct: float = 0.05,
                   debug_signals: bool = False,
                   poll_secs: Optional[int] = None):
    client = RoostooClient(api_key=apikey, secret_key=apisecret)
    alphas = {
        s: HybridAlphaConverted(volume_period=5,
                                atr_period=10,
                                momentum_period=mom_period,
                                volume_multiplier=vol_mult,
                                atr_multiplier=atr_mult,
                                stop_loss_pct=stop_loss,
                                momentum_epsilon=momentum_epsilon,
                                entry_epsilon=entry_epsilon,
                                exit_epsilon=exit_epsilon,
                                cooldown_bars=cooldown_bars,
                                atr_mom_mult=atr_mom_mult,
                                min_hold_bars=min_hold_bars,
                                take_profit_pct=take_profit_pct)
        for s in symbols
    }
    aggs = {s: BarAggregator(interval) for s in symbols}
    pairs = {s: _pair_from_symbol(s) for s in symbols}

    # warmup
    for s in symbols:
        try:
            seed = fetch_binance_klines(symbol=s, interval=interval, limit=30)
            if seed is not None and not seed.empty:
                for _, r in seed.sort_values("open_time").iterrows():
                    alphas[s].update({'time': r['open_time'], 'open': float(r['open']), 'high': float(r['high']),
                                      'low': float(r['low']), 'close': float(r['close']),
                                      'volume': float(r['volume']) if 'volume' in r and not pd.isna(r['volume']) else 0.0})
        except Exception:
            pass

    # synthetic ATR warmup if seeding failed
    for s in symbols:
        a = alphas[s]
        if getattr(a, "last_current_atr", 0) == 0 and len(a.atr_window) == 0:
            # inject small pseudo true ranges to avoid zero ATR (e.g. 0.05% of price)
            try:
                p = fetch_roostoo_ticker(pairs[s]) or 1.0
                tr_val = p * 0.0005
                for _ in range(10):
                    a.atr_window.append(tr_val)
                a.last_current_atr = tr_val  # set initial diagnostic
                print(f"[WARMUP-FALLBACK] seeded synthetic ATR for {s} tr={tr_val:.6f}")
            except Exception:
                pass

    port = LiveMultiPortfolio(cash=capital, fee=fee, risk_mult=risk_mult)
    day_equity_open = {}
    last_prices = {s: None for s in symbols}

    interval_secs = 86400 if interval == '1d' else 3600 if interval == '1h' else 900
    if poll_secs and poll_secs > 0:
        interval_secs = poll_secs
    print(f"DEPLOY MULTI: polling {','.join(symbols)} every {interval_secs}s. Dry-run={not force}")

    if do_check:
        tickers = fetch_roostoo_tickers()
        for s in symbols:
            p = tickers.get(pairs[s])
            if p:
                base_qty = (_min_notional_usd(s) * 1.2) / p
                test_qty = _ceil_to_step(s, base_qty)
                mode = "EXECUTING" if force else "DRY-RUN"
                print(f"[CHECK] {mode} {s} test BUY qty={test_qty:.6f} price={p:.2f}")
                if force and test_qty > 0:
                    resp_b = _place_order_safe(client, pairs[s], "BUY", test_qty)
                    print("[CHECK ORDER BUY]", s, _summarize_order_resp(resp_b))
                    _print_raw_resp(resp_b, label=f"check-buy-{s}")
                    resp_s = _place_order_safe(client, pairs[s], "SELL", test_qty)
                    print("[CHECK ORDER SELL]", s, _summarize_order_resp(resp_s))
                    _print_raw_resp(resp_s, label=f"check-sell-{s}")

    try:
        while True:
            tickers = fetch_roostoo_tickers()
            now = pd.to_datetime('now', utc=True)
            prices_now = {}

            # update aggregators
            closed = []
            for s in symbols:
                pr = tickers.get(pairs[s])
                if pr is None:
                    continue
                last_prices[s] = pr
                prices_now[s] = pr
                bar_closed = aggs[s].update(now, pr)
                if bar_closed is not None:
                    closed.append((s, bar_closed))
                elif debug_signals:
                    # show interim diagnostics using last known values
                    a = alphas[s]
                    try:
                        print(f"[DBG~] {s} t={now} price={pr:.6f} mom={a.last_momentum:.6f} atr={a.last_current_atr:.6f}")
                    except Exception:
                        pass

            # process signals
            for s, bar in closed:
                sig = alphas[s].update(bar)
                price = bar['close']
                if debug_signals:
                    print(f"[DBG] {s} t={bar['time']} mom={alphas[s].last_momentum:.6f} atr={alphas[s].last_current_atr:.6f} sig={sig}")

                if sig in ('buy', 'sell'):
                    print(f"[SIG] {s} {sig} at {bar['time']} px={price:.6f}")

                if sig == 'buy':
                    open_syms = [x for x, q in port.pos.items() if q > 0]
                    if len(open_syms) >= max_open_symbols:
                        print(f"[FILTER] {s} blocked by max_open_symbols={max_open_symbols}")
                        continue
                    day_key = now.date()
                    if day_key not in day_equity_open:
                        day_equity_open[day_key] = port.value(prices_now)
                    cur_eq = port.value(prices_now)
                    if (cur_eq - day_equity_open[day_key]) / day_equity_open[day_key] <= -max_daily_dd_pct:
                        print(f"[FILTER] {s} blocked by daily DD ({max_daily_dd_pct:.2%})")
                        continue
                    raw_qty = port.buy_value(s, price, alloc, prices_now)
                    rqty = _round_qty(s, raw_qty)
                    notional = rqty * price
                    if rqty <= 0:
                        print(f"[FILTER] {s} no cash/qty after sizing")
                        continue
                    if notional < _min_notional_usd(s):
                        print(f"[FILTER] {s} notional too small ${notional:.2f} < ${_min_notional_usd(s):.2f}")
                        # revert portfolio cash if you want strict accounting; omit for simplicity
                        continue
                    print(f"[LIVE] BUY {s} qty={rqty:.6f} price={price:.2f}")
                    if force:
                        resp = _place_order_safe(client, pairs[s], "BUY", rqty)
                        print("[ORDER]", s, _summarize_order_resp(resp))
                        _print_raw_resp(resp, label=f"live-buy-{s}")
                elif sig == 'sell':
                    qty = _round_qty(s, port.pos.get(s, 0.0))
                    if qty > 0:
                        sold = port.sell_all(s, price)
                        rq = _round_qty(s, sold)
                        print(f"[LIVE] SELL {s} qty={rq:.6f} price={price:.2f}")
                        if force and rq > 0:
                            resp = _place_order_safe(client, pairs[s], "SELL", rq)
                            print("[ORDER]", s, _summarize_order_resp(resp))
                            _print_raw_resp(resp, label=f"live-sell-{s}")

            if verbose:
                eq = port.value({k: v for k, v in last_prices.items() if v})
                open_pos = [(k, port.pos[k]) for k in port.pos if port.pos[k] > 0]
                print(f"[STATUS] t={now} equity={eq:.2f} cash={port.cash:.2f} open={open_pos} trades={port.trade_count}")

            time.sleep(interval_secs)
    except KeyboardInterrupt:
        print("Stopping multi-symbol live.")

def _place_order_safe(client, pair_or_coin, side, quantity, order_type='MARKET', price=None):
    """
    Robust wrapper to call client.place_order with several common signatures.
    - pair_or_coin: normalized pair like "BTC/USD" or base coin "BTC"
    - tries keyword and positional forms, and both 'type' and 'order_type' names.
    Returns the raw client response or raises last exception.
    """
    qty = str(quantity)
    # possible keyword variants (include price so client receives it when supported)
    kw_variants = [
        {'pair_or_coin': pair_or_coin, 'side': side, 'quantity': qty, 'order_type': order_type, 'price': price},
        {'pair_or_coin': pair_or_coin, 'side': side, 'quantity': qty, 'type': order_type, 'price': price},
        {'pair': pair_or_coin, 'side': side, 'quantity': qty, 'type': order_type, 'price': price},
        {'pair': pair_or_coin, 'side': side, 'quantity': qty, 'order_type': order_type, 'price': price},
        {'symbol': pair_or_coin, 'side': side, 'quantity': qty, 'order_type': order_type, 'price': price},
    ]
    last_exc = None
    for kw in kw_variants:
        try:
            return client.place_order(**kw)
        except TypeError as te:
            last_exc = te
            continue
        except Exception as e:
            # return error payload for visibility
            return {"error": str(e)}

    # positional fallbacks
    positional_variants = [
        (pair_or_coin, side, qty),
        (pair_or_coin, side, qty, order_type),
        (pair_or_coin, side, qty, price),
        (pair_or_coin, side, qty, price, order_type),
    ]
    for args in positional_variants:
        try:
            return client.place_order(*args)
        except TypeError as te:
            last_exc = te
            continue
        except Exception as e:
            return {"error": str(e)}

    # if all attempts failed, raise last TypeError for visibility
    if last_exc:
        raise last_exc
    raise RuntimeError("Unable to call client.place_order with tested signatures")

# --- new helpers: summarize order responses and print runtime status ---
def _summarize_order_resp(resp):
    """Return short string summary for rostoo order/place/query responses."""
    try:
        if resp is None:
            return "no-response"
        if isinstance(resp, dict):
            if resp.get("Success") and resp.get("OrderDetail"):
                od = resp.get("OrderDetail")
                return f"{od.get('Side')} id={od.get('OrderID')} {od.get('Status')} price={od.get('Price')} qty={od.get('Quantity')}"
            # some responses return OrderMatched list
            if resp.get("Success") and resp.get("OrderMatched"):
                matched = resp.get("OrderMatched")
                if len(matched):
                    o = matched[0]
                    return f"{o.get('Side')} id={o.get('OrderID')} {o.get('Status')} price={o.get('Price')} qty={o.get('Quantity')}"
            # generic dict -> show Success/ErrMsg
            if "Success" in resp:
                return f"Success={resp.get('Success')} Err={resp.get('ErrMsg','')}"
            # fallback: stringify limited
            return str({k: resp[k] for k in list(resp)[:6]})
        # other types
        return str(resp)
    except Exception:
        return repr(resp)

def _print_raw_resp(resp, label="raw"):
    """Print the raw API response (safe JSON dump or repr) for debugging."""
    try:
        if resp is None:
            print(f"[API {label}] None")
            return
        if isinstance(resp, (str, int, float)):
            print(f"[API {label}] {resp}")
            return
        # attempt JSON dump, truncate long output
        print(f"[API {label}] {json.dumps(resp, default=str)[:8000]}")
    except Exception:
        print(f"[API {label}] repr:", repr(resp))

def _print_status(port: 'SimplePortfolio', pair: str, last_signal, last_order_resp, verbose: bool=False):
    """Print compact bot status line to terminal."""
    # REMOVED status spam unless verbose explicitly True
    if not verbose:
        return
    print(f"[STATUS {datetime.utcnow()}] pair={pair} cash={port.cash:.2f} "
          f"pos={port.positions:.6f} avgp={port.avg_price if port.avg_price else 'N/A'} "
          f"trades={port.trade_count} last_signal={last_signal} "
          f"last_order={'no-response' if last_order_resp is None else 'ok'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=False)
    parser.add_argument("--symbols", required=False, help="comma-separated symbols, e.g. BTCUSDT,ETHUSDT,TRXUSDT,LTCUSDT,SUSDT")
    parser.add_argument("--interval", required=True)
    parser.add_argument("--source", default="horus")
    parser.add_argument("--limit", type=int, default=168)
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--risk-mult", type=float, default=1.0)
    parser.add_argument("--allocation", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--apikey", default=None)
    parser.add_argument("--api-secret", default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--vol-mult", type=float, default=1.0)
    parser.add_argument("--atr-mult", type=float, default=1.0)
    parser.add_argument("--mom-period", type=int, default=3)
    parser.add_argument("--stop-loss", type=float, default=0.05)
    parser.add_argument("--debug-signals", action="store_true")
    parser.add_argument("--poll-secs", type=int, default=None)
    parser.add_argument("--momentum-epsilon", type=float, default=0.0005)
    parser.add_argument("--entry-epsilon", type=float, default=0.0008)
    parser.add_argument("--exit-epsilon", type=float, default=0.0005)
    parser.add_argument("--cooldown-bars", type=int, default=1)
    parser.add_argument("--atr-mom-mult", type=float, default=0.0)
    parser.add_argument("--min-hold-bars", type=int, default=2)
    parser.add_argument("--take-profit-pct", type=float, default=0.01)
    parser.add_argument("--max-open-symbols", type=int, default=3)
    parser.add_argument("--max-daily-dd-pct", type=float, default=0.05)
    args = parser.parse_args()

    print("START main.py", {"argv": sys.argv, "env_DEPLOY": os.environ.get("DEPLOY")}, flush=True)

    try:
        if args.deploy:
            if args.symbols:
                syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
                run_live_multi(syms, args.interval, args.apikey, args.api_secret,
                               args.capital, 0.0001, args.risk_mult, args.allocation,
                               args.verbose, args.force, do_check=args.check,
                               vol_mult=args.vol_mult, atr_mult=args.atr_mult,  # FIX: use atr_mult not atr_mom_mult
                               mom_period=args.mom_period, stop_loss=args.stop_loss,
                               momentum_epsilon=args.momentum_epsilon,
                               entry_epsilon=args.entry_epsilon,
                               exit_epsilon=args.exit_epsilon,
                               cooldown_bars=args.cooldown_bars,
                               atr_mom_mult=args.atr_mom_mult,
                               min_hold_bars=args.min_hold_bars,
                               take_profit_pct=args.take_profit_pct,
                               max_open_symbols=args.max_open_symbols,
                               max_daily_dd_pct=args.max_daily_dd_pct,
                               debug_signals=args.debug_signals,
                               poll_secs=args.poll_secs)
            else:
                run_live(args.symbol or "BTCUSDT", args.interval, args.apikey, args.api_secret,
                         args.capital, 0.0001, args.risk_mult, args.allocation,
                         args.verbose, args.force, do_check=args.check,
                         vol_mult=args.vol_mult, atr_mult=args.atr_mult,  # FIX here too
                         mom_period=args.mom_period, stop_loss=args.stop_loss,
                         momentum_epsilon=args.momentum_epsilon,
                         entry_epsilon=args.entry_epsilon,
                         exit_epsilon=args.exit_epsilon,
                         cooldown_bars=args.cooldown_bars,
                         atr_mom_mult=args.atr_mom_mult,
                         min_hold_bars=args.min_hold_bars,
                         take_profit_pct=args.take_profit_pct,
                         max_open_symbols=args.max_open_symbols,
                         max_daily_dd_pct=args.max_daily_dd_pct,
                         debug_signals=args.debug_signals,
                         poll_secs=args.poll_secs)
        else:
            print("Calling run_backtest ...", flush=True)
            run_backtest((args.symbol or "BTCUSDT"), args.interval, args.limit,
                         apikey=args.apikey, source=args.source,
                         capital=args.capital, fee=0.0001, risk_mult=args.risk_mult, verbose=args.verbose)
    except Exception as e:
        print("UNHANDLED EXCEPTION in main:", str(e), flush=True)
        traceback.print_exc()
        sys.exit(2)
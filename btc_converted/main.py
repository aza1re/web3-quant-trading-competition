import os
import sys
# ensure repo root on path to import horus and Indicators if needed
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from binance import fetch_klines as fetch_binance_klines

from horus import fetch_klines as fetch_horus_klines
from alpha import HybridAlphaConverted
from rostoo import RoostooClient, BASE_URL

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
def run_live(symbol: str, interval: str, apikey: str, apisecret: str, capital: float, fee: float, risk_mult: float, alloc: float, verbose: bool, force: bool):
    pair = _pair_from_symbol(symbol)
    client = RoostooClient(api_key=apikey, secret_key=apisecret)
    alpha = HybridAlphaConverted(volume_period=5, atr_period=10, momentum_period=3,
                                 volume_multiplier=1.0, atr_multiplier=1.0, stop_loss_pct=0.05)
    # live portfolio tracking (local)
    port = SimplePortfolio(cash=capital, fee=fee, risk_mult=risk_mult)

    interval_secs = 86400 if interval == '1d' else 3600 if interval == '1h' else 900
    print(f"DEPLOY MODE: polling {pair} every {interval_secs}s. Dry-run={not force}")

    try:
        while True:
            price = fetch_roostoo_ticker(pair)
            now = pd.to_datetime('now')
            if price is None:
                if verbose:
                    print(f"[{now}] failed to fetch price for {pair}")
                time.sleep(max(5, interval_secs))
                continue

            bar = {'time': now, 'open': price, 'high': price, 'low': price, 'close': price, 'volume': 0.0}
            sig = alpha.update(bar)

            if verbose:
                print(f"[{now}] price={price:.2f} signal={sig} entry_price={alpha.entry_price}")

            if sig == 'buy':
                # compute quantity: use allocation fraction of portfolio
                qty = None
                if alloc is not None:
                    qty = (port.portfolio_value(price) * alloc * risk_mult) / price
                qty = float(max(0.0, qty or 0.0))
                if qty <= 0:
                    print("Computed zero qty, skipping order.")
                else:
                    print(f"[LIVE] BUY {symbol} qty={qty:.8f} price={price:.2f}")
                    if force:
                        try:
                            resp = client.place_order(symbol, "BUY", qty, order_type='MARKET')
                            print("Order resp:", resp)
                        except Exception as e:
                            print("Order failed:", e)
                    else:
                        print("Dry-run: not sending order. Use --force to execute.")
                    # update local portfolio state optimistically
                    port.buy_allocation(price, alloc)
            elif sig == 'sell':
                # close position
                qty = port.positions
                if qty > 0:
                    print(f"[LIVE] SELL {symbol} qty={qty:.8f} price={price:.2f}")
                    if force:
                        try:
                            resp = client.place_order(symbol, "SELL", qty, order_type='MARKET')
                            print("Order resp:", resp)
                        except Exception as e:
                            print("Order failed:", e)
                    else:
                        print("Dry-run: not sending order. Use --force to execute.")
                    # update local portfolio
                    port.sell_all(price)

            # sleep until next poll
            time.sleep(interval_secs)
    except KeyboardInterrupt:
        print("Stopping live deploy loop (KeyboardInterrupt).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converted BTC backtest using Horus or live Roostoo deploy")
    parser.add_argument('--symbol', type=str, default="BTCUSDT")
    parser.add_argument('--interval', type=str, default='1d', choices=['1d','1h','15m'])
    parser.add_argument('--source', type=str, default='horus', choices=['horus','binance'], help='Data source for backtest')
    parser.add_argument('--limit', type=int, default=None, help="Number of bars to fetch (defaults to 1y depending on interval)")
    parser.add_argument('--capital', type=float, default=100000.0)
    parser.add_argument('--fee', type=float, default=0.0001)
    parser.add_argument('--apikey', type=str, required=False)
    parser.add_argument('--risk-mult', type=float, default=1.0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--deploy', action='store_true', help='Run in live deploy mode using Roostoo')
    parser.add_argument('--api-secret', type=str, default=None, help='Roostoo API secret (required for deploy)')
    parser.add_argument('--allocation', type=float, default=0.5, help='Fraction of portfolio used when opening a trade (live)')
    parser.add_argument('--force', action='store_true', help='Actually execute live orders (default dry-run)')
    args = parser.parse_args()

    if args.deploy:
        # require keys for deploy
        if not args.apikey or not args.api_secret:
            print("Deploy mode requires --apikey and --api-secret")
            sys.exit(1)
        run_live(symbol=args.symbol,
                 interval=args.interval,
                 apikey=args.apikey,
                 apisecret=args.api_secret,
                 capital=args.capital,
                 fee=args.fee,
                 risk_mult=args.risk_mult,
                 alloc=args.allocation,
                 verbose=args.verbose,
                 force=args.force)
    else:
        if args.limit is None:
            if args.interval == '1d':
                args.limit = 365
            elif args.interval == '1h':
                args.limit = 24 * 365
            else:
                args.limit = 4 * 365

        run_backtest(symbol=args.symbol,
                     interval=args.interval,
                     limit=args.limit,
                     apikey=args.apikey,
                     source=args.source,
                     capital=args.capital,
                     fee=args.fee,
                     risk_mult=args.risk_mult,
                     verbose=args.verbose)
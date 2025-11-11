import argparse
from typing import List
import time
import os
import sys

# ensure repo root is on sys.path so imports like "import horus" work when running this script directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from horus import fetch_klines as fetch_horus_klines
from Indicators import CustomBollingerBands, RSIIndicator, HistoricalVolatility, TrendFilter
from AssetStrategy import AssetArbitrageStrategy

# Simple AlgoAPI used by AssetArbitrageStrategy
class AlgoAPI:
    def __init__(self, symbols: List[str], cash=100000.0, fee=0.0001, verbose: bool = False):
        self.cash = float(cash)
        self.fee = float(fee)
        self.symbols = symbols
        self.positions = {s: 0.0 for s in symbols}
        self.avg_price = {s: None for s in symbols}
        self.last_price = {s: None for s in symbols}
        self.trades = 0
        self.verbose = verbose

    def Price(self, symbol):
        return self.last_price.get(symbol, None)

    def HoldingsQty(self, symbol):
        return self.positions.get(symbol, 0.0)

    def HoldingsAvgPrice(self, symbol):
        return self.avg_price.get(symbol, None)

    def PortfolioValue(self):
        mv = sum((self.last_price.get(s) or 0.0) * qty for s, qty in self.positions.items())
        return self.cash + mv

    def PortfolioExposure(self):
        total = self.PortfolioValue()
        if total == 0:
            return 0.0
        invested = sum(abs((self.last_price.get(s) or 0.0) * qty) for s, qty in self.positions.items())
        return invested / total

    def SetHoldings(self, symbol, target_weight):
        price = self.Price(symbol)
        if price is None or price <= 0:
            if self.verbose:
                print(f"[SetHoldings] missing price for {symbol}")
            return
        pv = self.PortfolioValue()
        target_value = pv * target_weight
        current_value = self.positions.get(symbol, 0.0) * price
        delta_value = target_value - current_value
        if abs(delta_value) < 1e-8:
            return
        qty = delta_value / price
        cost = qty * price
        fee = abs(cost) * self.fee
        # execute
        self.cash -= (cost + fee)
        new_qty = self.positions.get(symbol, 0.0) + qty
        # update avg price
        if new_qty == 0:
            self.avg_price[symbol] = None
        else:
            prev_qty = self.positions.get(symbol, 0.0)
            prev_avg = self.avg_price.get(symbol)
            if prev_avg is None or prev_qty == 0:
                self.avg_price[symbol] = price
            else:
                prev_val = prev_avg * prev_qty
                new_val = prev_val + cost
                self.avg_price[symbol] = new_val / new_qty
        self.positions[symbol] = new_qty
        self.trades += 1
        if self.verbose:
            print(f"[TRADE] {symbol} qty_change={qty:.6f} price={price:.2f} fee={fee:.2f} cash={self.cash:.2f} new_qty={new_qty:.6f}")

    def Liquidate(self, symbol):
        price = self.Price(symbol)
        qty = self.positions.get(symbol, 0.0)
        if qty == 0 or price is None:
            return
        proceeds = qty * price
        fee = abs(proceeds) * self.fee
        self.cash += (proceeds - fee)
        self.positions[symbol] = 0.0
        self.avg_price[symbol] = None
        self.trades += 1
        if self.verbose:
            print(f"[LIQUIDATE] {symbol} qty={qty:.6f} price={price:.2f} fee={fee:.2f} cash={self.cash:.2f}")

def fetch_symbol_close_series(symbol: str, interval: str, limit: int, api_key: str):
    df = fetch_horus_klines(symbol=symbol, interval=interval, limit=limit, api_key=api_key)
    if df is None or df.empty:
        return None
    df = df.sort_values('open_time').reset_index(drop=True)
    df = df[['open_time', 'close']].rename(columns={'open_time': 'time', 'close': symbol})
    return df

def visualize_arbitrage(results: dict, merged_equity: pd.DataFrame, total_initial: float):
    """
    results: dict symbol -> DataFrame with columns: time, close, upper, middle, lower, rsi, vol, position, position_value
    merged_equity: DataFrame time + total_equity + drawdown
    """
    symbols = list(results.keys())
    n = len(symbols)
    fig_rows = n + 1
    fig, axes = plt.subplots(fig_rows, 1, figsize=(14, 4 * fig_rows), gridspec_kw={'height_ratios': [2]*n + [2]})
    if n == 1:
        axes = [axes]

    for idx, s in enumerate(symbols):
        ax = axes[idx]
        df = results[s]
        ax.plot(df['time'], df['close'], label=f'{s} Close', color='blue', alpha=0.7)
        # plot Bollinger bands if available
        if 'upper' in df.columns and df['upper'].notna().any():
            ax.plot(df['time'], df['upper'], label='Upper BB', color='gray', linestyle='--', alpha=0.7)
            ax.plot(df['time'], df['middle'], label='Middle BB', color='black', linestyle=':', alpha=0.7)
            ax.plot(df['time'], df['lower'], label='Lower BB', color='gray', linestyle='--', alpha=0.7)

        # mark entry/exit events (position change)
        df['pos_change'] = df['position'].diff().fillna(0)
        buys = df[(df['pos_change'] > 0)]
        sells = df[(df['pos_change'] < 0)]
        if not buys.empty:
            ax.plot(buys['time'], buys['close'], '^', markersize=8, color='green', lw=0, label='Entry')
        if not sells.empty:
            ax.plot(sells['time'], sells['close'], 'v', markersize=8, color='red', lw=0, label='Exit')

        ax.set_title(f'{s} Price, Bollinger Bands & Signals')
        ax.set_ylabel('Price')
        ax.legend(loc='upper left')
        ax.grid(True)

    # equity subplot
    ax_eq = axes[-1]
    ax_eq.plot(merged_equity['time'], merged_equity['total_equity'], label='Portfolio Equity', color='green')
    ax_eq.set_title('Aggregated Portfolio Equity')
    ax_eq.set_ylabel('Portfolio Value ($)')
    ax_eq.grid(True)
    ax_eq.legend(loc='upper left')

    ax_dd = ax_eq.twinx()
    ax_dd.fill_between(merged_equity['time'], merged_equity['drawdown'] * 100, 0, color='red', alpha=0.3, label='Drawdown')
    ax_dd.set_ylabel('Drawdown (%)', color='red')
    ax_dd.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    plt.show()

def run(symbols: List[str], interval: str, limit: int, api_key: str, capital: float = 100000.0, fee: float = 0.0001, verbose: bool = False):
    # fetch per-symbol series and merge
    frames = {}
    for s in symbols:
        if verbose:
            print(f"Fetching {s} from Horus...")
        df = fetch_symbol_close_series(s, interval, limit, api_key)
        if df is None:
            print(f"No data for {s}, aborting.")
            return
        frames[s] = df
        if verbose:
            print(f"  {s}: rows={len(df)} start={df['time'].iloc[0]} end={df['time'].iloc[-1]}")

    # merge asof on time (nearest)
    merged = frames[symbols[0]]
    for s in symbols[1:]:
        merged = pd.merge_asof(merged.sort_values('time'),
                               frames[s].sort_values('time'),
                               on='time', direction='nearest', tolerance=pd.Timedelta('1h'))
    merged = merged.sort_values('time').ffill().dropna().reset_index(drop=True)

    # prepare AlgoAPI and per-symbol indicators + strategy
    internal_symbols = [s.upper().replace('USDT', '').replace('USD', '') + 'USD' for s in symbols]
    api = AlgoAPI(internal_symbols, cash=capital, fee=fee, verbose=verbose)

    state = {}
    # per-symbol history storage
    history = {internal: [] for internal in internal_symbols}

    for i, s in enumerate(symbols):
        internal = internal_symbols[i]
        state[internal] = {
            "indicators": {
                "contrarian_bands": CustomBollingerBands(period=20, deviations=2),
                "rsi": RSIIndicator(period=14),
                "hist_vol": HistoricalVolatility(period=20),
                "trend": TrendFilter(sma_period=50)
            },
            "strategy": AssetArbitrageStrategy(api, internal, risk_multiplier=args.risk_mult, tighter_stops=args.tighter_stops)
        }

    equity = []
    times = []
    for idx, row in merged.iterrows():
        t = row['time']
        times.append(t)
        # update prices on API
        for i, s in enumerate(symbols):
            internal = internal_symbols[i]
            price = float(row[s])
            api.last_price[internal] = price

        # update indicators and run strategy for each asset
        for i, s in enumerate(symbols):
            internal = internal_symbols[i]
            p = api.Price(internal)
            ind = state[internal]['indicators']
            ind['contrarian_bands'].Update(p)
            ind['rsi'].Update(p)
            ind['hist_vol'].Update(p)
            ind['trend'].Update(p)
            # Execute will read indicators by name contrarian_bands, rsi, trend (and volatility)
            state[internal]['strategy'].Execute({
                "contrarian_bands": ind['contrarian_bands'],
                "rsi": ind['rsi'],
                "trend": ind['trend'],
                "volatility": ind['hist_vol']
            })

            # record per-symbol snapshot
            bb = ind['contrarian_bands']
            rsi = ind['rsi']
            vol = ind['hist_vol']
            history[internal].append({
                'time': t,
                'close': p,
                'upper': getattr(bb, 'UpperBand', np.nan),
                'middle': getattr(bb, 'MiddleBand', np.nan),
                'lower': getattr(bb, 'LowerBand', np.nan),
                'rsi': getattr(rsi, 'Current', np.nan),
                'volatility': getattr(vol, 'current_vol', np.nan) if hasattr(vol, 'current_vol') else np.nan,
                'position': api.positions.get(internal, 0.0),
                'position_value': api.positions.get(internal, 0.0) * p,
                'portfolio_value': api.PortfolioValue()
            })

        equity.append(api.PortfolioValue())

    # build per-symbol DataFrames
    results = {}
    for i, s in enumerate(symbols):
        internal = internal_symbols[i]
        df = pd.DataFrame(history[internal])
        # position flag (1 if qty>0)
        df['position'] = (df['position'] != 0).astype(int)
        results[s] = df

    # build merged equity DataFrame for plotting and metrics
    merged_eq = pd.DataFrame({'time': times, 'total_equity': equity})
    merged_eq['peak'] = merged_eq['total_equity'].cummax()
    merged_eq['drawdown'] = (merged_eq['total_equity'] - merged_eq['peak']) / merged_eq['peak']

    # aggregated summary
    start = merged_eq['time'].iloc[0] if not merged_eq.empty else None
    end = merged_eq['time'].iloc[-1] if not merged_eq.empty else None
    final = merged_eq['total_equity'].iloc[-1] if not merged_eq.empty else api.cash
    total_return = (final / capital - 1.0) if capital else 0.0

    # basic annualization
    per = interval.lower()
    periods_per_year = 365 if per == '1d' else 24 * 365 if per == '1h' else 4 * 365 if per == '15m' else 365
    ser = merged_eq['total_equity'].pct_change().dropna()
    mean_r = ser.mean() if not ser.empty else 0.0
    vol = ser.std() if not ser.empty else 0.0
    ann_return = (1 + mean_r) ** periods_per_year - 1 if periods_per_year and mean_r != 0 else total_return
    ann_vol = vol * (periods_per_year ** 0.5) if periods_per_year else vol

    peak = merged_eq['total_equity'].cummax()
    drawdown = (merged_eq['total_equity'] - peak) / peak
    max_dd = drawdown.min() if not drawdown.empty else 0.0

    print("\n--- Arbitrage Backtest (Horus) ---")
    print(f"Symbols: {symbols}")
    print(f"Period: {start} to {end}")
    print(f"Initial Capital: {capital:.2f}")
    print(f"Final Portfolio Value: {final:.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return (approx): {ann_return:.2%}")
    print(f"Annualized Volatility (approx): {ann_vol:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Total Trades: {api.trades}")

    # visualize like root main.py
    try:
        visualize_arbitrage(results, merged_eq, total_initial=capital)
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run arbitrage backtest with Horus (multi-symbol)")
    parser.add_argument('--symbols', type=str, default="BTCUSDT,ETHUSDT,SOLUSDT", help="Comma-separated symbols")
    parser.add_argument('--interval', type=str, default='1h', choices=['1d','1h','15m'])
    parser.add_argument('--limit', type=int, default=None, help="Number of points to fetch. Default: 1 year based on interval")
    parser.add_argument('--capital', type=float, default=100000.0)
    parser.add_argument('--fee', type=float, default=0.0001)
    parser.add_argument('--apikey', type=str, required=True, help="Horus API key")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--risk-mult', type=float, default=1.0, help="Risk multiplier (>1 increases position sizes and exposure)")
    parser.add_argument('--tighter-stops', action='store_true', help="Use tighter stops (more aggressive)")

    args = parser.parse_args()

    # default 1 year mapping
    if args.limit is None:
        if args.interval == '1d':
            args.limit = 365
        elif args.interval == '1h':
            args.limit = 24 * 365  # ~8760
        else:
            args.limit = 4 * 365

    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]

    # pass risk options when creating strategies
    run(symbols, interval=args.interval, limit=args.limit, api_key=args.apikey,
        capital=args.capital, fee=args.fee, verbose=args.verbose)
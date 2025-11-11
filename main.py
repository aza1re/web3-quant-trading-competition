import argparse
import sys
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from binance import fetch_klines as fetch_binance_klines
from horus import fetch_klines as fetch_horus_klines
from rostoo import RoostooClient

def backtest_ma_crossover(df,
                          short_window=10,
                          long_window=50,
                          trading_fee=0.001,
                          initial_capital=10000,
                          trade_amount=10000,
                          verbose: bool = False):
    """
    Backtests a Moving Average Crossover strategy with ATR trailing stop.
    Returns DataFrame with portfolio values and signals (same layout as before).
    """
    if df is None or 'close' not in df.columns:
        if verbose:
            print("DataFrame is invalid or missing 'close'.")
        return None

    data = df.copy().reset_index(drop=True)
    data['short_ma'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['long_ma'] = data['close'].ewm(span=long_window, adjust=False).mean()

    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    data['atr'] = tr.rolling(window=14).mean()

    data['signal'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
    data['crossover_position'] = data['signal'].shift(1).fillna(0)

    data = data.dropna().reset_index(drop=True)
    if data.empty:
        if verbose:
            print("Not enough data after processing.")
        return None

    position_qty = 0.0
    entry_price = 0.0
    stop_loss = 0.0
    capital = float(initial_capital)
    portfolio_values = []
    trades = 0
    position_binary_list = []

    for i in range(len(data)):
        current_price = float(data['close'].iloc[i])
        atr = float(data['atr'].iloc[i]) if not pd.isna(data['atr'].iloc[i]) else 0.0

        if position_qty == 0:
            position_binary_list.append(0)
            if data['crossover_position'].iloc[i] == 1:
                if capital >= trade_amount and current_price > 0:
                    position_qty = trade_amount / current_price
                    entry_price = current_price
                    stop_loss = current_price - 1.5 * atr if atr > 0 else 0.0
                    trade_cost = position_qty * current_price
                    trade_fee_cost = trade_cost * trading_fee
                    capital -= (trade_cost + trade_fee_cost)
                    trades += 1
                    if verbose:
                        print(f"[BUY] idx={i} price={current_price:.2f} qty={position_qty:.6f} fee={trade_fee_cost:.2f} cash={capital:.2f}")
        else:
            position_binary_list.append(1)
            new_stop_loss = current_price - 1.5 * atr if atr > 0 else stop_loss
            stop_loss = max(stop_loss, new_stop_loss)
            if stop_loss and current_price <= stop_loss:
                sell_proceeds = position_qty * current_price
                trade_fee_cost = sell_proceeds * trading_fee
                capital += (sell_proceeds - trade_fee_cost)
                if verbose:
                    print(f"[STOP] idx={i} price={current_price:.2f} qty={position_qty:.6f} fee={trade_fee_cost:.2f} cash={capital:.2f}")
                position_qty = 0.0
                trades += 1
            elif data['crossover_position'].iloc[i] == 0:
                sell_proceeds = position_qty * current_price
                trade_fee_cost = sell_proceeds * trading_fee
                capital += (sell_proceeds - trade_fee_cost)
                if verbose:
                    print(f"[SELL] idx={i} price={current_price:.2f} qty={position_qty:.6f} fee={trade_fee_cost:.2f} cash={capital:.2f}")
                position_qty = 0.0
                trades += 1

        current_portfolio = capital + (position_qty * current_price)
        portfolio_values.append(current_portfolio)

    if position_qty > 0:
        sell_proceeds = position_qty * current_price
        trade_fee_cost = sell_proceeds * trading_fee
        capital += (sell_proceeds - trade_fee_cost)
        trades += 1
        portfolio_values[-1] = capital
        if verbose:
            print(f"[FINAL CLOSE] price={current_price:.2f} cash={capital:.2f}")

    data['portfolio_value'] = portfolio_values
    data['position'] = position_binary_list
    data['strategy_returns'] = data['portfolio_value'].pct_change().fillna(0)
    data['peak_value'] = data['portfolio_value'].cummax()
    data['drawdown'] = (data['portfolio_value'] - data['peak_value']) / data['peak_value']

    if verbose:
        print(f"Trades: {trades}, Start: {data['open_time'].iloc[0]}, End: {data['open_time'].iloc[-1]}, Final PV: {data['portfolio_value'].iloc[-1]:.2f}")

    return data

def visualize_multi_backtest(results: dict, total_initial: float):
    """
    Keep original graph style per symbol and an aggregated equity subplot.
    results: dict symbol -> dataframe returned by backtest_ma_crossover
    """
    symbols = list(results.keys())
    n = len(symbols)

    # build merged equity timeseries
    eq_frames = []
    for s in symbols:
        res = results[s].copy()
        res = res.rename(columns={'portfolio_value': f'pv_{s}'})
        eq_frames.append(res[['open_time', f'pv_{s}']])

    merged = eq_frames[0]
    for f in eq_frames[1:]:
        merged = pd.merge_asof(merged.sort_values('open_time'),
                               f.sort_values('open_time'),
                               on='open_time', direction='nearest', tolerance=pd.Timedelta('1h'))
    merged = merged.sort_values('open_time').ffill().reset_index(drop=True)
    pv_cols = [c for c in merged.columns if c.startswith('pv_')]
    merged['total_equity'] = merged[pv_cols].sum(axis=1)

    merged['peak'] = merged['total_equity'].cummax()
    merged['drawdown'] = (merged['total_equity'] - merged['peak']) / merged['peak']

    fig_rows = n + 1
    fig, axes = plt.subplots(fig_rows, 1, figsize=(14, 4 * fig_rows), gridspec_kw={'height_ratios': [2]*n + [2]})
    if n == 1:
        axes = [axes]

    for idx, s in enumerate(symbols):
        ax = axes[idx]
        df = results[s]
        ax.plot(df['open_time'], df['close'], label=f'{s} Close', color='blue', alpha=0.7)
        ax.plot(df['open_time'], df['short_ma'], label='Short MA', color='orange', linestyle='--')
        ax.plot(df['open_time'], df['long_ma'], label='Long MA', color='purple', linestyle='--')

        buy_signals = df[(df['position'] == 1) & (df['position'].diff() > 0)]
        sell_signals = df[(df['position'] == 0) & (df['position'].diff() < 0)]

        if not buy_signals.empty:
            ax.plot(buy_signals['open_time'], buy_signals['close'], '^', markersize=8, color='green', lw=0, label='Buy')
        if not sell_signals.empty:
            ax.plot(sell_signals['open_time'], sell_signals['close'], 'v', markersize=8, color='red', lw=0, label='Sell')

        ax.set_title(f'{s} Price & Signals')
        ax.set_ylabel('Price')
        ax.legend(loc='upper left')
        ax.grid(True)

    ax_eq = axes[-1]
    ax_eq.plot(merged['open_time'], merged['total_equity'], label='Portfolio Equity', color='green')
    ax_eq.set_title('Aggregated Portfolio Equity')
    ax_eq.set_ylabel('Portfolio Value ($)')
    ax_eq.grid(True)
    ax_eq.legend(loc='upper left')

    ax_dd = ax_eq.twinx()
    ax_dd.fill_between(merged['open_time'], merged['drawdown'] * 100, 0, color='red', alpha=0.3, label='Drawdown')
    ax_dd.set_ylabel('Drawdown (%)', color='red')
    ax_dd.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    plt.show()

# Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-symbol backtest, keep original graphs")
    parser.add_argument('--source', choices=['binance', 'horus'], default='binance',
                        help="Data source for backtest")
    parser.add_argument('--apikey', type=str, default=None, help="API key for Horus")
    parser.add_argument('--symbols', type=str, default="BTCUSDT,ETHUSDT,SOLUSDT",
                        help="Comma-separated symbols")
    parser.add_argument('--limit', type=int, default=365, help="Kline fetch limit (points). For 1y daily use 365")
    parser.add_argument('--interval', type=str, default='1d', help="Interval: '1d' or '1h' or '15m'")
    parser.add_argument('--short', type=int, default=5, help="Short EMA window")
    parser.add_argument('--long', type=int, default=10, help="Long EMA window")
    parser.add_argument('--capital', type=float, default=50000, help="Total capital (split equally)")
    parser.add_argument('--trade-pct', type=float, default=0.2, help="Fraction of per-symbol capital used per trade (0-1)")
    parser.add_argument('--fee', type=float, default=0.0001, help="Trading fee fraction")
    parser.add_argument('--verbose', action='store_true', help="Verbose logging")
    parser.add_argument('--risk-mult', type=float, default=1.0, help="Risk multiplier (>1 increases per-trade size)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    if not symbols:
        print("No symbols provided.")
        sys.exit(1)

    per_symbol_capital = args.capital / len(symbols)
    results = {}

    for s in symbols:
        print(f"Fetching {s} from {args.source}...")
        if args.source == 'binance':
            df = fetch_binance_klines(symbol=s, interval=args.interval, limit=args.limit)
        else:
            if not args.apikey:
                print("Horus source selected but --apikey not provided.")
                sys.exit(1)
            df = fetch_horus_klines(symbol=s, interval=args.interval, limit=args.limit, api_key=args.apikey)

        if df is None or df.empty:
            print(f"No data for {s}, skipping.")
            continue

        trade_amount = per_symbol_capital * args.trade_pct
        # apply risk multiplier (cap to per-symbol capital)
        trade_amount = min(per_symbol_capital, trade_amount * getattr(args, "risk_mult", 1.0))
        if args.verbose:
             print(f"{s}: per-symbol capital={per_symbol_capital:.2f}, trade_amount={trade_amount:.2f}")

        res = backtest_ma_crossover(df,
                                    short_window=args.short,
                                    long_window=args.long,
                                    trading_fee=args.fee,
                                    initial_capital=per_symbol_capital,
                                    trade_amount=trade_amount,
                                    verbose=args.verbose)
        if res is None:
            print(f"Backtest failed/insufficient data for {s}")
            continue

        results[s] = res

    if not results:
        print("No results to visualize.")
        sys.exit(0)

    # --- aggregated summary (insert here) ---
    # Build merged equity timeseries (same logic as visualize_multi_backtest)
    eq_frames = []
    for s, df in results.items():
        tmp = df[['open_time', 'portfolio_value']].rename(columns={'portfolio_value': f'pv_{s}'})
        eq_frames.append(tmp)

    merged_eq = eq_frames[0]
    for f in eq_frames[1:]:
        merged_eq = pd.merge_asof(merged_eq.sort_values('open_time'),
                                  f.sort_values('open_time'),
                                  on='open_time', direction='nearest', tolerance=pd.Timedelta('1h'))
    merged_eq = merged_eq.sort_values('open_time').ffill().reset_index(drop=True)
    pv_cols = [c for c in merged_eq.columns if c.startswith('pv_')]
    merged_eq['total_equity'] = merged_eq[pv_cols].sum(axis=1)

    # basic metrics
    start = merged_eq['open_time'].iloc[0]
    end = merged_eq['open_time'].iloc[-1]
    total_return = merged_eq['total_equity'].iloc[-1] / merged_eq['total_equity'].iloc[0] - 1
    # determine periods per year from interval
    per = args.interval.lower()
    periods_per_year = 365 if per == '1d' else 24 * 365 if per == '1h' else 4 * 365 if per == '15m' else 365
    rets = merged_eq['total_equity'].pct_change().dropna()
    mean_r = rets.mean() if not rets.empty else 0.0
    vol = rets.std() if not rets.empty else 0.0
    ann_return = (1 + mean_r) ** periods_per_year - 1 if periods_per_year and mean_r != 0 else total_return
    ann_vol = vol * (periods_per_year ** 0.5)

    peak = merged_eq['total_equity'].cummax()
    drawdown = (merged_eq['total_equity'] - peak) / peak
    max_dd = drawdown.min() if not drawdown.empty else 0.0

    # risk metrics
    sharpe = (mean_r / vol) * (periods_per_year ** 0.5) if vol > 0 else float('nan')
    downside = rets[rets < 0]
    dstd = downside.std() if not downside.empty else 0.0
    sortino = (mean_r / dstd) * (periods_per_year ** 0.5) if dstd > 0 else float('nan')
    calmar = ann_return / abs(max_dd) if max_dd < 0 else float('nan')

    # trades: sum of position-change events across symbols
    total_trades = 0
    for s, df in results.items():
        total_trades += int(df['position'].diff().abs().fillna(0).gt(0).sum())

    print("\nBacktest Complete:")
    print(f"Period: {start} to {end}")
    print(f"Total Strategy Return: {total_return:.2%}")
    print(f"Total Trades: {total_trades}")
    # average trades per day
    try:
        days = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 86400.0
        days = max(days, 1.0)
        avg_trades_per_day = total_trades / days
    except Exception:
        avg_trades_per_day = float('nan')
    print(f"Avg Trades / Day: {avg_trades_per_day:.2f}")
    print(f"Annualized Return (approx): {ann_return:.2%}")
    print(f"Annualized Volatility: {ann_vol:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}, Sortino: {sortino:.2f}, Calmar: {calmar:.2f}")
    # --- end aggregated summary ---

    visualize_multi_backtest(results, total_initial=args.capital)
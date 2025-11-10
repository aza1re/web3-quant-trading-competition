import argparse
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from binance import fetch_klines

# Simple per-symbol loop backtest that uses ATR-based trailing stop and risk-based position sizing.
def backtest_symbol(df: pd.DataFrame,
                    allocated_capital: float,
                    risk_per_trade_pct: float = 0.01,
                    short_window: int = 5,
                    long_window: int = 10,
                    trading_fee: float = 0.0001,
                    stop_multiplier: float = 1.5):
    if df is None or 'close' not in df.columns:
        return None, {}

    data = df.copy().reset_index(drop=True)
    data['short_ma'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['long_ma'] = data['close'].ewm(span=long_window, adjust=False).mean()

    # ATR(14)
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift(1))
    low_close = np.abs(data['low'] - data['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    data['atr'] = tr.rolling(14).mean()

    data['signal'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
    data['crossover_position'] = data['signal'].shift(1).fillna(0)
    data = data.dropna().reset_index(drop=True)
    if data.empty:
        return None, {}

    cash = allocated_capital
    position_qty = 0.0
    stop_loss = 0.0
    portfolio_vals = []
    position_flag = []
    trades = 0

    # absolute dollars risk per trade
    risk_amount = allocated_capital * risk_per_trade_pct

    for i, row in data.iterrows():
        price = row['close']
        atr = row['atr'] if not np.isnan(row['atr']) else 0.0

        # no position -> consider buy
        if position_qty == 0:
            position_flag.append(0)
            if row['crossover_position'] == 1 and price > 0 and atr > 0:
                stop_distance = stop_multiplier * atr
                # compute qty by risk: qty * stop_distance ~= risk_amount
                qty = risk_amount / stop_distance if stop_distance > 0 else 0
                # Ensure we don't exceed cash (account for fee)
                max_qty_by_cash = cash / (price * (1 + trading_fee))
                qty = min(qty, max_qty_by_cash)
                if qty * price > 0 and qty > 0:
                    position_qty = qty
                    entry_price = price
                    stop_loss = price - stop_distance
                    trade_cost = qty * price
                    fee = trade_cost * trading_fee
                    cash -= (trade_cost + fee)
                    trades += 1
        else:
            position_flag.append(1)
            # trailing stop update
            new_stop = price - stop_multiplier * atr if atr > 0 else stop_loss
            stop_loss = max(stop_loss, new_stop)

            # stop hit?
            if price <= stop_loss:
                proceeds = position_qty * price
                fee = proceeds * trading_fee
                cash += (proceeds - fee)
                position_qty = 0.0
                trades += 1
            # crossover sell
            elif row['crossover_position'] == 0:
                proceeds = position_qty * price
                fee = proceeds * trading_fee
                cash += (proceeds - fee)
                position_qty = 0.0
                trades += 1

        portfolio_vals.append(cash + position_qty * price)

    # close any remaining
    if position_qty > 0:
        price = data['close'].iloc[-1]
        proceeds = position_qty * price
        fee = proceeds * trading_fee
        cash += (proceeds - fee)
        trades += 1
        portfolio_vals[-1] = cash

    data['portfolio_value'] = portfolio_vals
    data['position'] = position_flag
    # returns based on portfolio value
    data['strategy_returns'] = data['portfolio_value'].pct_change().fillna(0)

    # compute peak and drawdown for this symbol (needed by run_portfolio)
    data['peak'] = data['portfolio_value'].cummax()
    # drawdown is negative when below peak
    data['drawdown'] = (data['portfolio_value'] - data['peak']) / data['peak']

    metrics = {}
    metrics['final_value'] = data['portfolio_value'].iloc[-1]
    metrics['total_return'] = (metrics['final_value'] / allocated_capital) - 1
    metrics['trades'] = trades
    metrics['mean_return_hourly'] = data['strategy_returns'].mean()
    metrics['vol_hourly'] = data['strategy_returns'].std()
    # max drawdown (negative value)
    metrics['max_drawdown'] = data['drawdown'].min()

    return data, metrics

def run_portfolio(symbols: List[str],
                  initial_capital: float = 50000,
                  risk_per_trade_pct: float = 0.01,
                  limit: int = 500,
                  interval: str = '1h',
                  short_window: int = 5,
                  long_window: int = 10,
                  stop_multiplier: float = 1.5,
                  trading_fee: float = 0.0001):
    n = len(symbols)
    alloc = initial_capital / n
    dfs = {}
    metrics = {}

    for s in symbols:
        print(f"Fetching {s}...")
        df = fetch_klines(symbol=s, interval=interval, limit=limit)
        if df is None:
            print(f"Failed to fetch {s}, skipping.")
            continue
        df, m = backtest_symbol(df,
                                allocated_capital=alloc,
                                risk_per_trade_pct=risk_per_trade_pct,
                                short_window=short_window,
                                long_window=long_window,
                                trading_fee=trading_fee,
                                stop_multiplier=stop_multiplier)
        if df is None:
            print(f"Not enough data for {s}, skipping.")
            continue
        dfs[s] = df[['open_time', 'portfolio_value', 'position', 'drawdown']].rename(
            columns={'portfolio_value': f'pv_{s}', 'position': f'pos_{s}'}
        )
        metrics[s] = m

    if not dfs:
        print("No symbols available to backtest.")
        return

    # Merge on open_time - assume similar timestamps; perform outer merge on time and forward-fill
    merged = None
    for s, df in dfs.items():
        if merged is None:
            merged = df
        else:
            merged = pd.merge_asof(merged.sort_values('open_time'),
                                   df.sort_values('open_time'),
                                   on='open_time', direction='nearest', tolerance=pd.Timedelta('1h'))
    merged = merged.sort_values('open_time')
    # forward-fill aligned timestamps, then replace remaining NaNs with 0
    merged = merged.sort_values('open_time').ffill().fillna(0)

    # Sum per-symbol portfolio value to aggregate full portfolio equity
    pv_cols = [c for c in merged.columns if c.startswith('pv_')]
    merged['total_equity'] = merged[pv_cols].sum(axis=1)

    # compute aggregated metrics
    merged['returns'] = merged['total_equity'].pct_change().fillna(0)
    hours_in_year = 365 * 24
    mean_r = merged['returns'].mean()
    vol_h = merged['returns'].std()
    ann_return = (1 + mean_r) ** hours_in_year - 1
    ann_vol = vol_h * np.sqrt(hours_in_year)
    peak = merged['total_equity'].cummax()
    drawdown = (merged['total_equity'] - peak) / peak
    max_dd = drawdown.min()

    print("\n--- Portfolio Summary ---")
    print(f"Symbols: {symbols}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Equity: ${merged['total_equity'].iloc[-1]:,.2f}")
    print(f"Total Return: {(merged['total_equity'].iloc[-1] / initial_capital - 1):.2%}")
    print(f"Annualized Return (approx): {ann_return:.2%}")
    print(f"Annualized Volatility (approx): {ann_vol:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")

    # print per-symbol quick metrics
    for s, m in metrics.items():
        print(f"{s}: Return {(m['total_return']):.2%}, Trades {m['trades']}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(merged['open_time'], merged['total_equity'], label='Portfolio Equity')
    for s in symbols:
        col = f'pv_{s}'
        if col in merged:
            plt.plot(merged['open_time'], merged[col], alpha=0.4, label=s)
    plt.legend()
    plt.title('Multi-Symbol Portfolio Equity')
    plt.xlabel('Time')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, default="BTCUSDT,ETHUSDT,BNBUSDT",
                        help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument('--capital', type=float, default=50000)
    parser.add_argument('--risk', type=float, default=0.01, help="Risk per trade as fraction of allocated capital")
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--interval', type=str, default='1h')
    parser.add_argument('--short', type=int, default=5, help="Short EMA window")
    parser.add_argument('--long', type=int, default=10, help="Long EMA window")
    parser.add_argument('--stop', type=float, default=1.5, help="ATR stop multiplier")
    parser.add_argument('--fee', type=float, default=0.0001, help="Trading fee fraction")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    run_portfolio(symbols,
                  initial_capital=args.capital,
                  risk_per_trade_pct=args.risk,
                  limit=args.limit,
                  interval=args.interval,
                  short_window=args.short,
                  long_window=args.long,
                  stop_multiplier=args.stop,
                  trading_fee=args.fee)
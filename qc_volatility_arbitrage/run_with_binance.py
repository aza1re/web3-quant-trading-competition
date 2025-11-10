import argparse
import os
import importlib.util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Load the root binance.py by path to avoid local module name collision
root_binance_path = os.path.join(os.getcwd(), "binance.py")
spec = importlib.util.spec_from_file_location("root_binance", root_binance_path)
root_binance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(root_binance)
fetch_klines = root_binance.fetch_klines

from Indicators import CustomBollingerBands, RSIIndicator, HistoricalVolatility, TrendFilter
from AssetStrategy import AssetArbitrageStrategy

# Lightweight AlgoAPI (same semantics as qc_volatility_arbitrage/main.py)
class AlgoAPI:
    def __init__(self, symbols: List[str], cash=100000.0, fee=0.0001):
        self.cash = cash
        self.fee = fee
        self.symbols = symbols
        self.positions = {s: 0.0 for s in symbols}
        self.avg_price = {s: None for s in symbols}
        self.last_price = {s: None for s in symbols}

    def Price(self, symbol): return self.last_price.get(symbol)
    def HoldingsQty(self, symbol): return self.positions.get(symbol, 0.0)
    def HoldingsAvgPrice(self, symbol): return self.avg_price.get(symbol)
    def PortfolioValue(self):
        mv = sum((self.last_price[s] or 0.0) * qty for s, qty in self.positions.items())
        return self.cash + mv
    def PortfolioExposure(self):
        total = self.PortfolioValue()
        if total == 0: return 0.0
        invested = sum(abs((self.last_price[s] or 0.0) * qty) for s, qty in self.positions.items())
        return invested / total

    def SetHoldings(self, symbol, target_weight):
        pv = self.PortfolioValue()
        target_value = pv * target_weight
        price = self.Price(symbol)
        if price is None or price <= 0: return
        current_value = self.positions[symbol] * price
        delta_value = target_value - current_value
        if abs(delta_value) < 1e-8: return
        qty = delta_value / price
        cost = qty * price
        fee = abs(cost) * self.fee
        self.cash -= (cost + fee)
        new_qty = self.positions[symbol] + qty
        if new_qty == 0:
            self.avg_price[symbol] = None
        else:
            prev_val = self.positions[symbol] * (self.avg_price[symbol] or price)
            new_val = prev_val + cost
            self.avg_price[symbol] = new_val / new_qty
        self.positions[symbol] = new_qty

    def Liquidate(self, symbol):
        price = self.Price(symbol)
        qty = self.positions.get(symbol, 0.0)
        if qty == 0 or price is None: return
        proceeds = qty * price
        fee = abs(proceeds) * self.fee
        self.cash += (proceeds - fee)
        self.positions[symbol] = 0.0
        self.avg_price[symbol] = None

def fetch_daily_df(binance_symbol: str, limit: int):
    df = fetch_klines(symbol=binance_symbol, interval='1d', limit=limit)
    if df is None or df.empty:
        return None
    df = df[['open_time', 'close']].copy()
    df.rename(columns={'open_time': 'date', 'close': binance_symbol}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df[binance_symbol] = pd.to_numeric(df[binance_symbol], errors='coerce')
    df = df.resample('1D').last().ffill().reset_index()
    return df

def run(symbols_binance: List[str], capital: float = 100000.0, limit: int = 1000, fee: float = 0.0001):
    mapping = {f"{s.replace('USDT','USD')}": s for s in symbols_binance}
    merged = None
    for bsym in symbols_binance:
        print(f"Fetching {bsym}...")
        df = fetch_daily_df(bsym, limit)
        if df is None:
            print(f"Failed to fetch {bsym}")
            return
        col = bsym
        if merged is None:
            merged = df[['date', col]].copy()
        else:
            merged = merged.merge(df[['date', col]], on='date', how='outer')
    merged = merged.sort_values('date').ffill().dropna().reset_index(drop=True)

    internal_symbols = [k for k in mapping.keys()]

    api = AlgoAPI(internal_symbols, cash=capital, fee=fee)
    state = {}
    for s in internal_symbols:
        state[s] = {
            "indicators": {
                "contrarian_bands": CustomBollingerBands(period=20, deviations=2),
                "rsi": RSIIndicator(period=14),
                "hist_vol": HistoricalVolatility(period=20),
                "trend": TrendFilter(sma_period=50)
            },
            "strategy": AssetArbitrageStrategy(api, s)
        }

    equity = []
    dates = []
    for _, row in merged.iterrows():
        date = row['date']
        for internal, bsym in mapping.items():
            price = float(row.get(bsym, np.nan))
            api.last_price[internal] = price

        for s in internal_symbols:
            ind = state[s]['indicators']
            p = api.Price(s)
            ind['contrarian_bands'].Update(p)
            ind['rsi'].Update(p)
            ind['hist_vol'].Update(p)
            ind['trend'].Update(p)
            state[s]['strategy'].Execute(ind)

        equity.append(api.PortfolioValue())
        dates.append(date)

    print("--- Backtest (Binance daily) ---")
    print(f"Start: {dates[0]}, End: {dates[-1]}")
    print(f"Initial Cash: {capital:.2f}")
    print(f"Final Portfolio Value: {equity[-1]:.2f}")
    print(f"Total Return: {(equity[-1]/capital - 1):.2%}")

    try:
        plt.figure(figsize=(10,5))
        plt.plot(dates, equity, label='Equity')
        plt.title('Equity Curve — Binance daily')
        plt.xlabel('Date'); plt.ylabel('Portfolio Value')
        plt.grid(True); plt.legend(); plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--capital', type=float, default=100000.0)
    parser.add_argument('--fee', type=float, default=0.0001)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    run(symbols, capital=args.capital, limit=args.limit, fee=args.fee)
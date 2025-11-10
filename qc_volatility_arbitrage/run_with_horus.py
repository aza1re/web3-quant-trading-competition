import argparse
import time
import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from datetime import datetime, timezone

from Indicators import CustomBollingerBands, RSIIndicator, HistoricalVolatility, TrendFilter
from AssetStrategy import AssetArbitrageStrategy

HORUS_BASE = "https://api.horus.example"  # replace if different

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

def fetch_horus_price_series(asset: str, interval: str, limit: int, api_key: str):
    # asset: e.g. "BTC", "ETH", "SOL"
    interval_seconds = 86400 if interval == '1d' else 3600 if interval == '1h' else 900
    end_ts = int(time.time())
    start_ts = end_ts - limit * interval_seconds
    url = f"https://api.horus.com/market/price"  # use Horus host; adjust if needed
    headers = {"X-API-Key": api_key}
    params = {
        "asset": asset,
        "interval": interval,
        "start": start_ts,
        "end": end_ts,
        "format": "json"
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(None)
    df = df[['date', 'price']].rename(columns={'price': asset})
    # ensure regular cadence and forward fill
    freq = '1D' if interval == '1d' else '1H'
    df = df.set_index('date').resample(freq).last().ffill().reset_index()
    return df

def run(symbols_horus: List[str], capital: float = 100000.0, limit: int = 500, fee: float = 0.0001, api_key: str = None):
    if api_key is None:
        raise SystemExit("Provide --apikey")
    # Normalize provided tickers: accept "BTCUSDT" or "BTC"
    assets = []
    for s in symbols_horus:
        s = s.upper()
        if s.endswith('USDT') or s.endswith('USD'):
            assets.append(s.replace('USDT','').replace('USD',''))
        else:
            assets.append(s)
    # fetch and merge
    merged = None
    for asset in assets:
        print(f"Fetching {asset} from Horus...")
        df = fetch_horus_price_series(asset, interval='1d', limit=limit, api_key=api_key)
        if df is None:
            print(f"No data for {asset}")
            return
        col = asset
        if merged is None:
            merged = df.copy()
        else:
            merged = merged.merge(df[['date', col]], on='date', how='outer')
    merged = merged.sort_values('date').ffill().dropna().reset_index(drop=True)

    # internal symbol names e.g. BTCUSD
    internal_symbols = [f"{a}USD" for a in assets]
    api = AlgoAPI(internal_symbols, cash=capital, fee=fee)

    # build indicator + strategy per internal symbol
    state = {}
    for s, a in zip(internal_symbols, assets):
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
        for s, a in zip(internal_symbols, assets):
            price = float(row.get(a, np.nan))
            api.last_price[s] = price
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

    print("--- Backtest (Horus daily) ---")
    print(f"Start: {dates[0]}, End: {dates[-1]}")
    print(f"Initial Cash: {capital:.2f}")
    print(f"Final Portfolio Value: {equity[-1]:.2f}")
    print(f"Total Return: {(equity[-1]/capital - 1):.2%}")

    try:
        plt.figure(figsize=(10,5))
        plt.plot(dates, equity, label='Equity')
        plt.title('Equity Curve — Horus daily')
        plt.xlabel('Date'); plt.ylabel('Portfolio Value')
        plt.grid(True); plt.legend(); plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--capital', type=float, default=100000.0)
    parser.add_argument('--fee', type=float, default=0.0001)
    parser.add_argument('--apikey', type=str, default=None, help="Horus API key")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    run(symbols, capital=args.capital, limit=args.limit, fee=args.fee, api_key=args.apikey)
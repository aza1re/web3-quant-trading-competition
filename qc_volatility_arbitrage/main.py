import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from Indicators import CustomBollingerBands, RSIIndicator, HistoricalVolatility, TrendFilter
from AssetStrategy import AssetArbitrageStrategy
import os

# Simple backtester API used by AssetArbitrageStrategy
class AlgoAPI:
    def __init__(self, symbols, cash=100000.0, fee=0.0001):
        self.cash = cash
        self.fee = fee
        self.symbols = symbols
        self.positions = {s: 0.0 for s in symbols}     # quantity
        self.avg_price = {s: None for s in symbols}
        self.last_price = {s: None for s in symbols}
        self.history_equity = []

    def Price(self, symbol):
        return self.last_price.get(symbol, None)

    def HoldingsQty(self, symbol):
        return self.positions.get(symbol, 0.0)

    def HoldingsAvgPrice(self, symbol):
        return self.avg_price.get(symbol, None)

    def PortfolioValue(self):
        mv = sum((self.last_price[s] or 0.0) * qty for s, qty in self.positions.items())
        return self.cash + mv

    def PortfolioExposure(self):
        total = self.PortfolioValue()
        if total == 0:
            return 0.0
        invested = sum(abs((self.last_price[s] or 0.0) * qty) for s, qty in self.positions.items())
        return invested / total

    def SetHoldings(self, symbol, target_weight):
        # buy/sell to reach target_weight of current portfolio value
        pv = self.PortfolioValue()
        target_value = pv * target_weight
        price = self.Price(symbol)
        if price is None or price <= 0:
            return
        current_value = (self.positions[symbol] * price)
        delta_value = target_value - current_value
        if abs(delta_value) < 1e-6:
            return
        qty = delta_value / price
        # execute
        cost = qty * price
        fee = abs(cost) * self.fee
        # if buying, decrease cash; if selling, increase cash
        self.cash -= (cost + fee)
        # update position average price
        new_qty = self.positions[symbol] + qty
        if new_qty == 0:
            self.avg_price[symbol] = None
        else:
            # compute new avg price for net position
            prev_val = (self.positions[symbol] * (self.avg_price[symbol] or price))
            new_val = prev_val + cost
            self.avg_price[symbol] = new_val / new_qty
        self.positions[symbol] = new_qty

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

def load_csv_symbol(path):
    df = pd.read_csv(path)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df[['dt', 'price']].rename(columns={'dt': 'date', 'price': 'close'})
    df = df.set_index('date').resample('1D').last().ffill()
    df = df.reset_index()
    return df

def main():
    # symbols and CSVs in workspace
    mapping = {
        'BTCUSD': 'BTC.csv',
        'ETHUSD': 'ETH.csv',
        'SOLUSD': 'SOL.csv'
    }
    # load dataframes, align by date
    dfs = {}
    for sym, csv in mapping.items():
        p = os.path.join(os.getcwd(), csv)
        if not os.path.exists(p):
            print(f"Missing CSV: {csv}, skipping {sym}")
            return
        dfs[sym] = load_csv_symbol(p)

    # merge on date
    merged = dfs['BTCUSD'][['date']].copy()
    for sym, df in dfs.items():
        merged = merged.merge(df[['date', 'close']].rename(columns={'close': sym}), on='date', how='outer')
    merged = merged.sort_values('date').ffill().dropna().reset_index(drop=True)

    symbols = list(mapping.keys())
    api = AlgoAPI(symbols, cash=100000.0, fee=0.0001)

    # initialize indicator & strategy objects per symbol
    symbol_state = {}
    for s in symbols:
        symbol_state[s] = {
            "indicators": {
                "contrarian_bands": CustomBollingerBands(period=20, deviations=2),
                "rsi": RSIIndicator(period=14),
                "hist_vol": HistoricalVolatility(period=20),
                "trend": TrendFilter(sma_period=50)
            },
            "strategy": AssetArbitrageStrategy(api, s)
        }

    equity_curve = []
    dates = []
    for idx, row in merged.iterrows():
        date = row['date']
        # update prices first
        for s in symbols:
            price = float(row[s])
            api.last_price[s] = price

        # update indicators and run strategies
        for s in symbols:
            price = api.Price(s)
            ind = symbol_state[s]["indicators"]
            ind["contrarian_bands"].Update(price)
            ind["rsi"].Update(price)
            ind["hist_vol"].Update(price)
            ind["trend"].Update(price)
            symbol_state[s]["strategy"].Execute(ind)

        ev = api.PortfolioValue()
        equity_curve.append(ev)
        dates.append(date)

    # final summary
    print("--- Backtest summary ---")
    print(f"Start: {dates[0]}, End: {dates[-1]}")
    print(f"Initial Cash: 100000.00")
    print(f"Final Portfolio Value: {equity_curve[-1]:.2f}")
    total_return = (equity_curve[-1] / 100000.0) - 1
    print(f"Total Return: {total_return:.2%}")

    # simple plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        plt.plot(dates, equity_curve, label='Equity')
        plt.title('Equity Curve — Volatility Arbitrage (daily, simplified)')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.legend()
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
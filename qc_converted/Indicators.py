import numpy as np
import collections
from typing import Deque

class CustomBollingerBands:
    def __init__(self, period=20, deviations=2):
        self.period = period
        self.deviations = deviations
        self.window: Deque[float] = collections.deque(maxlen=period)
        self.IsReady = False
        self.UpperBand = None
        self.MiddleBand = None
        self.LowerBand = None

    def Update(self, price: float):
        if price is None:
            return
        self.window.append(price)
        if len(self.window) == self.period:
            arr = np.array(self.window)
            mean = arr.mean()
            std = arr.std(ddof=0)
            self.MiddleBand = mean
            self.UpperBand = mean + self.deviations * std
            self.LowerBand = mean - self.deviations * std
            self.IsReady = True

class RSIIndicator:
    def __init__(self, period=14):
        self.period = period
        self.gains = collections.deque(maxlen=period)
        self.losses = collections.deque(maxlen=period)
        self.prev = None
        self.IsReady = False
        self.Current = None

    def Update(self, price: float):
        if price is None:
            return
        if self.prev is None:
            self.prev = price
            return
        change = price - self.prev
        self.gains.append(max(change, 0.0))
        self.losses.append(max(-change, 0.0))
        self.prev = price
        if len(self.gains) == self.period:
            avg_gain = float(np.mean(self.gains))
            avg_loss = float(np.mean(self.losses))
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            self.Current = rsi
            self.IsReady = True

class HistoricalVolatility:
    def __init__(self, period=20):
        self.period = period
        self.prices = collections.deque(maxlen=period+1)
        self.IsReady = False
        self.current_vol = None

    def Update(self, price: float):
        if price is None:
            return
        self.prices.append(price)
        if len(self.prices) == self.prices.maxlen:
            p = np.array(self.prices)
            returns = np.diff(np.log(p))
            self.current_vol = float(np.std(returns, ddof=0)) * np.sqrt(252)  # annualized approx
            self.IsReady = True

    def GetVolatility(self):
        return self.current_vol

class TrendFilter:
    def __init__(self, sma_period=50):
        self.sma_period = sma_period
        self.window = collections.deque(maxlen=sma_period)
        self.IsReady = False
        self.Current = None

    def Update(self, price: float):
        if price is None:
            return
        self.window.append(price)
        if len(self.window) == self.sma_period:
            self.Current = float(np.mean(self.window))
            self.IsReady = True

    def IsUptrend(self, price: float):
        return self.IsReady and price > self.Current

    def IsDowntrend(self, price: float):
        return self.IsReady and price < self.Current
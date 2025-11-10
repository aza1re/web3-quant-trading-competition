from typing import Dict

class AssetArbitrageStrategy:
    def __init__(self, algo_api, symbol: str):
        self.algo = algo_api
        self.symbol = symbol
        self.long_trade_size = 0.05   # target weight for longs
        self.short_trade_size = 0.03  # target weight for shorts
        self.long_stop_loss = 0.05
        self.short_stop_loss = 0.03
        self.max_portfolio_exposure = 0.80

    def Execute(self, indicators: Dict):
        cb = indicators.get("contrarian_bands")
        rsi = indicators.get("rsi")
        trend = indicators.get("trend")

        # need signals ready
        if not (cb and rsi and trend):
            return
        if not (cb.IsReady and rsi.IsReady and trend.IsReady):
            return

        price = self.algo.Price(self.symbol)
        if price is None or price <= 0:
            return

        holdings = self.algo.HoldingsQty(self.symbol)
        avg_price = self.algo.HoldingsAvgPrice(self.symbol)

        # exposure check (fraction of total capital invested abs)
        exposure = self.algo.PortfolioExposure()
        if exposure > self.max_portfolio_exposure:
            return

        # LONG ENTRY
        if holdings == 0 and price < cb.LowerBand and rsi.Current < 30 and trend.IsUptrend(price):
            self.algo.SetHoldings(self.symbol, self.long_trade_size)
            return

        # SHORT ENTRY
        if holdings == 0 and price > cb.UpperBand and rsi.Current > 70 and trend.IsDowntrend(price):
            self.algo.SetHoldings(self.symbol, -self.short_trade_size)
            return

        # Stop losses
        if holdings > 0 and avg_price is not None and price < avg_price * (1 - self.long_stop_loss):
            self.algo.Liquidate(self.symbol)
            return
        if holdings < 0 and avg_price is not None and price > avg_price * (1 + self.short_stop_loss):
            self.algo.Liquidate(self.symbol)
            return

        # Exit on mean reversion
        if holdings > 0 and price >= cb.MiddleBand:
            self.algo.Liquidate(self.symbol)
            return
        if holdings < 0 and price <= cb.MiddleBand:
            self.algo.Liquidate(self.symbol)
            return
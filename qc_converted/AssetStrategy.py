from typing import Dict

class AssetArbitrageStrategy:
    def __init__(self, algo_api, symbol: str, risk_multiplier: float = 1.0, tighter_stops: bool = False):
        self.algo = algo_api
        self.symbol = symbol
        # base sizes
        self.base_long_trade_size = 0.05   # base target weight for longs
        self.base_short_trade_size = 0.03  # base target weight for shorts
        self.base_long_stop_loss = 0.05
        self.base_short_stop_loss = 0.03
        self.base_max_portfolio_exposure = 0.80

        # apply risk multiplier
        self.risk_multiplier = float(risk_multiplier) if risk_multiplier and risk_multiplier > 0 else 1.0
        self.long_trade_size = min(1.0, self.base_long_trade_size * self.risk_multiplier)
        self.short_trade_size = min(1.0, self.base_short_trade_size * self.risk_multiplier)
        # allow max exposure to grow but cap at 1.5
        self.max_portfolio_exposure = min(1.5, self.base_max_portfolio_exposure * self.risk_multiplier)

        # tighter stops (optional) — reduce stop distance (more aggressive)
        if tighter_stops:
            self.long_stop_loss = max(0.001, self.base_long_stop_loss / max(1.0, self.risk_multiplier))
            self.short_stop_loss = max(0.001, self.base_short_stop_loss / max(1.0, self.risk_multiplier))
        else:
            # if risk multiplier large, you may want wider stops to avoid noise: scale stops up slightly
            self.long_stop_loss = min(0.5, self.base_long_stop_loss * (1.0 if self.risk_multiplier <= 1 else self.risk_multiplier ** 0.25))
            self.short_stop_loss = min(0.5, self.base_short_stop_loss * (1.0 if self.risk_multiplier <= 1 else self.risk_multiplier ** 0.25))

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
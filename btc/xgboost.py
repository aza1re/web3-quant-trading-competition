from AlgorithmImports import *

class HybridAlpha(AlphaModel):
    def __init__(self, algorithm, binance_symbol, coinbase_symbol, stop_loss_percentage=0.03):
        self.algorithm = algorithm
        self.binance_symbol = binance_symbol
        self.coinbase_symbol = coinbase_symbol
        self.stop_loss_percentage = stop_loss_percentage
        self.last_signal_time = datetime.min
        self.entry_price = None
        self.arbitrage_active = False

        self.volume_period = 10
        self.atr_period = 14
        self.momentum_period = 5
        self.atr_sma_period = 10
        self.window_size = max(self.volume_period, self.atr_period, self.momentum_period)

        self.volume_window = RollingWindow(self.volume_period)
        self.atr_window = RollingWindow(self.atr_period)
        self.atr_window_sma = RollingWindow(self.atr_sma_period)
        self.window = RollingWindow(self.window_size)

        self.volume_multiplier = 1.2
        self.atr_multiplier = 1.1
        self.spread_threshold = 0.015

        # Consolidators for both symbols
        self.consolidators = {}
        for symbol in [self.binance_symbol, self.coinbase_symbol]:
            consolidator = TradeBarConsolidator(timedelta(days=1))
            consolidator.data_consolidated += (
                lambda sender, bar, sym=symbol: self.on_consolidated(sym, bar)
            )
            self.consolidators[symbol] = consolidator
            algorithm.subscription_manager.add_consolidator(symbol, consolidator)

        # Warmup with daily historical data
        for symbol in [self.binance_symbol, self.coinbase_symbol]:
            history = algorithm.history[TradeBar](symbol, self.window_size, Resolution.DAILY)
            bar_found = False
            for bar in history:
                self.consolidators[symbol].update(bar)
                bar_found = True
            if not bar_found:
                algorithm.log(f"Warning: No historical data for {symbol}")

    def on_consolidated(self, symbol, bar):
        if symbol == self.binance_symbol:
            self.window.add(bar)
            self.algorithm.log(f"Updated window for {symbol}, count: {self.window.count}")

            if self.window.count >= self.volume_period:
                volumes = [self.window[i].volume for i in range(self.volume_period)]
                avg_volume = sum(volumes) / self.volume_period if volumes else 0
                self.volume_window.add(avg_volume)
                self.algorithm.log(f"Volume SMA for {symbol}: {avg_volume:.0f}")

            if self.window.count >= 2:
                prev_close = self.window[1].close
                tr = max(
                    bar.high - bar.low,
                    abs(bar.high - prev_close),
                    abs(prev_close - bar.low),
                )
                self.atr_window.add(tr)

                if self.atr_window.count >= self.atr_period:
                    atr = sum(self.atr_window) / self.atr_period
                    self.atr_window_sma.add(atr)
                    self.algorithm.log(f"ATR SMA for {symbol}: {atr:.2f}")

    def update(self, algorithm, data):
        insights = []
        if not (data.contains_key(self.binance_symbol) and data.contains_key(self.coinbase_symbol)):
            algorithm.log(f"Missing data for {self.binance_symbol} or {self.coinbase_symbol}")
            return insights

        if self.window.count < self.window_size:
            algorithm.log(
                f"Insufficient window data for {self.binance_symbol}: "
                f"{self.window.count}/{self.window_size}"
            )
            return insights

        if (algorithm.time - self.last_signal_time).total_seconds() / 60 < 15:
            return insights
        self.last_signal_time = algorithm.time

        binance_price = algorithm.securities[self.binance_symbol].price
        coinbase_price = algorithm.securities[self.coinbase_symbol].price
        if binance_price == 0 or coinbase_price == 0:
            algorithm.log(f"Zero price detected for {self.binance_symbol} or {self.coinbase_symbol}")
            return insights

        bar = self.window[0]

        if not self.arbitrage_active:
            if self.entry_price is not None and algorithm.portfolio[self.binance_symbol].quantity > 0:
                stop_loss_price = self.entry_price * (1 - self.stop_loss_percentage)
                if binance_price <= stop_loss_price:
                    insights.append(
                        Insight(self.binance_symbol, timedelta(hours=1), InsightType.PRICE, InsightDirection.FLAT)
                    )
                    algorithm.log(f"Directional stop-loss for {self.binance_symbol} at {stop_loss_price:.2f}")
                    self.entry_price = None
                    return insights

            current_volume = bar.volume
            avg_volume = self.volume_window[0] if self.volume_window.count > 0 else 0
            volume_surge = current_volume > avg_volume * self.volume_multiplier if avg_volume > 0 else False

            current_atr = (
                sum(self.atr_window) / self.atr_period
                if self.atr_window.count >= self.atr_period
                else 0
            )
            avg_atr = (
                sum(self.atr_window_sma) / self.atr_sma_period
                if self.atr_window_sma.count >= self.atr_sma_period
                else 0
            )
            high_volatility = current_atr > avg_atr * self.atr_multiplier if avg_atr > 0 else False

            if self.window.count >= self.momentum_period + 1:
                past_price = self.window[self.momentum_period].close
                momentum = (binance_price - past_price) / past_price if past_price != 0 else 0
            else:
                momentum = 0
            positive_momentum = momentum > 0

            algorithm.log(
                f"{self.binance_symbol}: Volume={current_volume:.0f}, AvgVol={avg_volume:.0f}, "
                f"VolumeSurge={volume_surge}, ATR={current_atr:.2f}, AvgATR={avg_atr:.2f}, "
                f"HighVol={high_volatility}, Momentum={momentum:.2%}, PositiveMom={positive_momentum}"
            )

            if volume_surge and high_volatility and positive_momentum:
                self.entry_price = binance_price
                algorithm.log(f"Directional buy for {self.binance_symbol}: Volume surge, high volatility, positive momentum")
                insights.append(
                    Insight(self.binance_symbol, timedelta(days=30), InsightType.PRICE, InsightDirection.UP)
                )

            if algorithm.portfolio[self.binance_symbol].invested and not (volume_surge or positive_momentum):
                insights.append(
                    Insight(self.binance_symbol, timedelta(hours=1), InsightType.PRICE, InsightDirection.FLAT)
                )
                algorithm.log(f"Directional sell for {self.binance_symbol}: Conditions weakened")
                self.entry_price = None

        spread = (
            abs(binance_price - coinbase_price) / min(binance_price, coinbase_price)
            if min(binance_price, coinbase_price) > 0
            else 0
        )
        algorithm.log(f"Arbitrage {self.binance_symbol}/{self.coinbase_symbol}: Spread={spread:.2%}, Threshold={self.spread_threshold:.2%}")

        if spread > self.spread_threshold and not (algorithm.portfolio[self.binance_symbol].invested or algorithm.portfolio[self.coinbase_symbol].invested):
            self.arbitrage_active = True
            if binance_price < coinbase_price:
                insights.append(
                    Insight(self.binance_symbol, timedelta(hours=24), InsightType.PRICE, InsightDirection.UP)
                )
                insights.append(
                    Insight(self.coinbase_symbol, timedelta(hours=24), InsightType.PRICE, InsightDirection.DOWN)
                )
                algorithm.log(
                    f"Arbitrage: Buy {self.binance_symbol} ({binance_price:.2f}), Sell {self.coinbase_symbol} ({coinbase_price:.2f}), Spread {spread:.2%}"
                )
            else:
                insights.append(
                    Insight(self.binance_symbol, timedelta(hours=24), InsightType.PRICE, InsightDirection.DOWN)
                )
                insights.append(
                    Insight(self.coinbase_symbol, timedelta(hours=24), InsightType.PRICE, InsightDirection.UP)
                )
                algorithm.log(
                    f"Arbitrage: Buy {self.coinbase_symbol} ({coinbase_price:.2f}), Sell {self.binance_symbol} ({binance_price:.2f}), Spread {spread:.2%}"
                )
        elif self.arbitrage_active and spread < self.spread_threshold / 2:
            insights.append(
                Insight(self.binance_symbol, timedelta(hours=1), InsightType.PRICE, InsightDirection.FLAT)
            )
            insights.append(
                Insight(self.coinbase_symbol, timedelta(hours=1), InsightType.PRICE, InsightDirection.FLAT)
            )
            algorithm.log(
                f"Arbitrage exit for {self.binance_symbol}/{self.coinbase_symbol}: Spread narrowed to {spread:.2%}"
            )
            self.arbitrage_active = False

        return insights
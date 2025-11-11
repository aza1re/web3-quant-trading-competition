from AlgorithmImports import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class HybridAlpha(AlphaModel):
    def __init__(self, algorithm, binance_symbol, coinbase_symbol, stop_loss_percentage=0.03):
        self.algorithm = algorithm
        self.binance = binance_symbol
        self.coinbase = coinbase_symbol
        self.hour = -4
        self.stop_loss_percentage = stop_loss_percentage
        self.entry_price = None

        # Directional parameters
        self.volume_period = 5
        self.volume_window = RollingWindow[float](self.volume_period)
        self.atr_period = 10
        self.atr_window = RollingWindow[float](self.atr_period)
        self.momentum_period = 3
        self.window = RollingWindow[TradeBar](max(self.volume_period, self.atr_period, self.momentum_period) + 20)

        # Arbitrage parameters
        self.spread_threshold = 0.005
        self.arbitrage_active = False

        # ML components
        self.ml_model = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42, min_samples_split=10)
        self.ml_trained = False
        self.ml_lookback = 5  # Use n days of history for training
        self.train_once = True  # Train once after warmup

        # Consolidator
        self.consolidator = TradeBarConsolidator(timedelta(days=1))
        self.consolidator.data_consolidated += self.on_consolidated
        algorithm.subscription_manager.add_consolidator(self.binance, self.consolidator)

        # Warmup - need extra data for labeling
        warmup_days = max(self.volume_period, self.atr_period, self.ml_lookback) + 10
        history = algorithm.history[TradeBar](self.binance, warmup_days * 24 * 60, Resolution.MINUTE)
        for bar in history:
            self.consolidator.update(bar)

    def on_consolidated(self, sender, bar):
        self.window.add(bar)

        if self.window.count >= self.volume_period:
            volumes = [self.window[i].volume for i in range(self.volume_period)]
            avg_volume = sum(volumes) / self.volume_period
            self.volume_window.add(avg_volume)

        if self.window.count >= 2:
            prev_close = self.window[1].close
            tr = max(bar.high - bar.low, abs(bar.high - prev_close), abs(prev_close - bar.low))
            self.atr_window.add(tr)

    def extract_features(self, index=0):
        """Extract features at specific window index for backtesting"""
        if self.window.count < max(self.volume_period, self.atr_period) + index:
            return None

        bar = self.window[index]
        
        # Volume ratio
        if self.volume_window.count > index:
            avg_volume = self.volume_window[index]
            volume_ratio = bar.volume / avg_volume if avg_volume > 0 else 1
        else:
            return None

        # ATR
        if self.atr_window.count >= self.atr_period:
            current_atr = sum([self.atr_window[i] for i in range(index, min(index + self.atr_period, self.atr_window.count))]) / self.atr_period
            avg_atr = sum(self.atr_window) / self.atr_window.count
            atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1
        else:
            return None

        # Momentum
        if self.window.count >= self.momentum_period + index + 1:
            past_price = self.window[self.momentum_period + index].close
            momentum = (bar.close - past_price) / past_price if past_price != 0 else 0
        else:
            return None

        # Price changes
        price_change_1d = (bar.close - bar.open) / bar.open if bar.open > 0 else 0
        if self.window.count >= 3 + index:
            price_change_3d = (bar.close - self.window[2 + index].close) / self.window[2 + index].close if self.window[2 + index].close > 0 else 0
        else:
            price_change_3d = 0

        # Simple RSI
        rsi = 50
        if self.window.count >= 14 + index:
            gains = sum([max(self.window[i].close - self.window[i+1].close, 0) for i in range(index, min(index + 14, self.window.count - 1))])
            losses = sum([max(self.window[i+1].close - self.window[i].close, 0) for i in range(index, min(index + 14, self.window.count - 1))])
            if losses > 0:
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))

        return [volume_ratio, atr_ratio, momentum, price_change_1d, price_change_3d, rsi / 100]

    def train_model(self):
        """Train model once using historical data in backtest"""
        if self.window.count < self.ml_lookback + 5:
            return False

        X_train = []
        y_train = []

        # Build training dataset from historical window
        for i in range(5, min(self.ml_lookback, self.window.count - 1)):
            features = self.extract_features(i)
            if features is None:
                continue
            
            # Label: 1 if price increased >2% in next 5 days, 0 otherwise
            current_price = self.window[i].close
            future_price = self.window[max(0, i - 5)].close  # Looking forward (reversed index)
            label = 1 if future_price > current_price * 1.02 else 0
            
            X_train.append(features)
            y_train.append(label)

        if len(X_train) >= 20:  # Need minimum samples
            X = np.array(X_train)
            y = np.array(y_train)
            self.ml_model.fit(X, y)
            self.ml_trained = True
            self.algorithm.log(f"ML model trained with {len(X)} historical samples")
            return True
        
        return False

    def update(self, algorithm, data):
        if not (data.contains_key(self.binance) and data.contains_key(self.coinbase)) or self.window.count < max(self.volume_period, self.atr_period):
            return []

        if self.hour == algorithm.time.hour:
            return []
        self.hour = algorithm.time.hour

        # Train model once after warmup in backtest
        if self.train_once and not self.ml_trained:
            if self.train_model():
                self.train_once = False

        binance_price = algorithm.securities[self.binance].price
        coinbase_price = algorithm.securities[self.coinbase].price
        bar = self.window[0]
        insights = []

        # Directional strategy with ML
        if not self.arbitrage_active:
            # Stop-loss
            if self.entry_price is not None and algorithm.portfolio[self.binance].quantity > 0:
                stop_loss_price = self.entry_price * (1 - self.stop_loss_percentage)
                if binance_price <= stop_loss_price:
                    algorithm.insights.cancel([self.binance])
                    algorithm.log(f"Stop-loss at {stop_loss_price:.2f}")
                    self.entry_price = None
                    return []

            # Traditional signals
            current_volume = bar.volume
            avg_volume = self.volume_window[0] if self.volume_window.count > 0 else 1
            volume_surge = current_volume > avg_volume * 1.2

            current_atr = sum(self.atr_window) / self.atr_period if self.atr_window.count >= self.atr_period else 0
            high_volatility = current_atr > 0  # Simplified

            if self.window.count >= self.momentum_period + 1:
                past_price = self.window[self.momentum_period].close
                momentum = (binance_price - past_price) / past_price if past_price != 0 else 0
            else:
                momentum = 0
            positive_momentum = momentum > 0

            # ML prediction
            ml_signal = False
            if self.ml_trained:
                features = self.extract_features(0)
                if features is not None:
                    try:
                        prediction = self.ml_model.predict([features])[0]
                        proba = self.ml_model.predict_proba([features])[0]
                        confidence = proba[1] if len(proba) > 1 else 0.5
                        ml_signal = prediction == 1 and confidence > 0.65
                    except:
                        ml_signal = False

            # Combined: Traditional AND ML must agree (stronger signal)
            traditional_signal = volume_surge and positive_momentum
            
            if traditional_signal and ml_signal:
                self.entry_price = binance_price
                algorithm.log(f"ML+Traditional BUY at {binance_price:.2f}")
                insights.append(Insight(self.binance, timedelta(days=30), InsightType.PRICE, InsightDirection.UP))
            elif not self.ml_trained and traditional_signal:
                # Fallback to traditional only if ML not ready
                self.entry_price = binance_price
                algorithm.log(f"Traditional BUY at {binance_price:.2f}")
                insights.append(Insight(self.binance, timedelta(days=30), InsightType.PRICE, InsightDirection.UP))

            # Exit
            if algorithm.portfolio[self.binance].invested and not (positive_momentum or ml_signal):
                algorithm.insights.cancel([self.binance])
                algorithm.log("EXIT signal")
                self.entry_price = None

        # Arbitrage (unchanged)
        spread = abs(binance_price - coinbase_price) / min(binance_price, coinbase_price)
        if spread > self.spread_threshold and not algorithm.portfolio[self.binance].invested:
            self.arbitrage_active = True
            if binance_price < coinbase_price:
                insights.append(Insight(self.binance, timedelta(hours=24), InsightType.PRICE, InsightDirection.UP))
                insights.append(Insight(self.coinbase, timedelta(hours=24), InsightType.PRICE, InsightDirection.DOWN))
                algorithm.log(f"ARB: Buy Binance, Sell Coinbase {spread:.2%}")
            else:
                insights.append(Insight(self.binance, timedelta(hours=24), InsightType.PRICE, InsightDirection.DOWN))
                insights.append(Insight(self.coinbase, timedelta(hours=24), InsightType.PRICE, InsightDirection.UP))
                algorithm.log(f"ARB: Buy Coinbase, Sell Binance {spread:.2%}")
        elif self.arbitrage_active and spread < self.spread_threshold / 2:
            algorithm.insights.cancel([self.binance, self.coinbase])
            algorithm.log(f"ARB exit {spread:.2%}")
            self.arbitrage_active = False

        return insights
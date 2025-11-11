from collections import deque
import numpy as np

class HybridAlphaConverted:
    """
    Converted QC HybridAlpha -> simple signal generator for offline backtest.
    Uses rolling windows for volume, ATR and momentum checks.
    Methods:
      - update(bar): feed TradeBar-like dict {'time','open','high','low','close','volume'}
      - signal(): returns 'buy', 'sell', or None based on most recent bar
    """

    def __init__(self,
                 volume_period=5,
                 atr_period=10,
                 momentum_period=3,
                 volume_multiplier=1.2,
                 atr_multiplier=1.1,
                 spread_threshold=0.005,
                 stop_loss_pct=0.03):
        self.volume_period = volume_period
        self.atr_period = atr_period
        self.momentum_period = momentum_period
        self.volume_multiplier = volume_multiplier
        self.atr_multiplier = atr_multiplier
        self.spread_threshold = spread_threshold
        self.stop_loss_pct = stop_loss_pct

        self.window_size = max(self.volume_period, self.atr_period, self.momentum_period) + 5
        self.window = deque(maxlen=self.window_size)  # store bars as dicts
        self.volume_window = deque(maxlen=self.volume_period)
        self.atr_window = deque(maxlen=self.atr_period)
        self.entry_price = None
        self.arbitrage_active = False  # kept for parity (not used without second venue)

    def _compute_tr(self, cur, prev):
        if prev is None:
            return cur['high'] - cur['low']
        return max(cur['high'] - cur['low'],
                   abs(cur['high'] - prev['close']),
                   abs(prev['close'] - cur['low']))

    def update(self, bar: dict):
        """
        Feed a bar (dict must contain 'open','high','low','close','volume','time').
        Returns one of: 'buy', 'sell', None
        """
        prev = self.window[-1] if len(self.window) > 0 else None
        self.window.append(bar)

        # update volume window (use last N bar volumes average)
        if len(self.window) >= 1:
            # compute avg over last volume_period bars if available
            if len(self.volume_window) == 0 and len(self.window) >= self.volume_period:
                vols = [b['volume'] for b in list(self.window)[-self.volume_period:]]
                self.volume_window.append(sum(vols) / len(vols))
            elif len(self.window) >= 1:
                # update running avg using last bar volume
                if len(self.volume_window) < self.volume_period:
                    self.volume_window.append(bar['volume'])
                else:
                    # maintain simple rolling average (append last volume then take mean)
                    tmp = list(self.volume_window)
                    tmp.append(bar['volume'])
                    if len(tmp) > self.volume_period:
                        tmp = tmp[-self.volume_period:]
                    self.volume_window.clear()
                    for v in tmp:
                        self.volume_window.append(v)

        # update ATR window
        tr = self._compute_tr(bar, prev)
        self.atr_window.append(tr)

        # Only signal when we have enough history
        if len(self.window) < max(self.volume_period, self.atr_period, self.momentum_period) + 1:
            return None

        # compute metrics
        avg_vol = np.mean(list(self.volume_window)) if len(self.volume_window) else 0
        # If avg_vol == 0 (Horus returns zero volumes), fall back to momentum-only signal
        if avg_vol <= 0:
            volume_surge = True  # ignore volume requirement when data absent
        else:
            volume_surge = (bar['volume'] > avg_vol * self.volume_multiplier)

        current_atr = np.mean(list(self.atr_window)) if len(self.atr_window) else 0
        # crude avg ATR as rolling mean of atr_window itself (short-term)
        avg_atr = current_atr
        high_volatility = current_atr > avg_atr * self.atr_multiplier if avg_atr > 0 else False

        # momentum: compare to close N bars ago
        if len(self.window) > self.momentum_period:
            past_price = list(self.window)[-1 - self.momentum_period]['close']
            momentum = (bar['close'] - past_price) / past_price if past_price != 0 else 0
        else:
            momentum = 0
        positive_momentum = momentum > 0

        # Simple directional rule (mirrors QC logic)
        if self.entry_price is None:
            if volume_surge and positive_momentum:
                self.entry_price = bar['close']
                return 'buy'
            # no explicit short path here (original QC mixed directional/arb)
        else:
            # stop-loss
            if bar['close'] <= self.entry_price * (1 - self.stop_loss_pct):
                self.entry_price = None
                return 'sell'
            # exit on momentum fade
            if not positive_momentum:
                self.entry_price = None
                return 'sell'

        return None
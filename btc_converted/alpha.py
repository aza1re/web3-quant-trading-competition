from collections import deque
from typing import Optional
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
                 stop_loss_pct=0.03,
                 momentum_epsilon=0.0005,
                 entry_epsilon=0.0008,
                 exit_epsilon=0.0005,
                 cooldown_bars=2,
                 atr_mom_mult=0.2,
                 min_hold_bars=2,
                 take_profit_pct=0.01,
                 trailing_stop_pct=0.02,
                 min_cash_usd=10.0,
                 min_cash_pct_to_buy=0.02,
                 tp_immediate: bool = False   # NEW: take profit immediately when threshold hit
                 ):
        self.volume_period = volume_period
        self.atr_period = atr_period
        self.momentum_period = momentum_period
        self.volume_multiplier = volume_multiplier
        self.atr_multiplier = atr_multiplier
        self.stop_loss_pct = stop_loss_pct
        self.momentum_epsilon = momentum_epsilon
        self.entry_epsilon = entry_epsilon
        self.exit_epsilon = exit_epsilon
        self.cooldown_bars = cooldown_bars
        self.atr_mom_mult = atr_mom_mult
        self.min_hold_bars = min_hold_bars
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.min_cash_usd = float(min_cash_usd)
        self.min_cash_pct_to_buy = float(min_cash_pct_to_buy)
        self.tp_immediate = bool(tp_immediate)  # NEW

        self.window = deque(maxlen=max(volume_period, atr_period, momentum_period) + 5)
        self.volume_window = deque(maxlen=volume_period)
        self.atr_window = deque(maxlen=atr_period)
        self.entry_price = None
        self.entry_bar_index = None
        self.entry_peak_price = None  # NEW: peak close since entry

        # account context (optional; set by main during live)
        self.usd_free = None
        self.usd_total = None
        self.pos_qty = 0.0
        self.pos_avg_price = None

        # diagnostics
        self.last_avg_vol = 0
        self.last_current_atr = 0
        self.last_momentum = 0
        self.last_volume_surge = False
        self.last_positive_momentum = False
        self.last_window_len = 0

        self._bar_index = 0
        self._last_trade_bar = -10

    def set_account_context(self, usd_free: Optional[float] = None, usd_total: Optional[float] = None,
                            pos_qty: Optional[float] = None, pos_avg_price: Optional[float] = None):
        """Optional live context: lets alpha apply cash gates and PnL-aware exits."""
        try:
            if usd_free is not None: self.usd_free = float(usd_free)
            if usd_total is not None: self.usd_total = float(usd_total)
            if pos_qty is not None: self.pos_qty = float(pos_qty)
            if pos_avg_price is not None: self.pos_avg_price = float(pos_avg_price)
        except Exception:
            pass

    def update(self, bar: dict):
        self._bar_index += 1
        self.window.append(bar)
        self.volume_window.append(bar.get('volume', 0) or 0)

        if len(self.window) >= 2:
            prev = list(self.window)[-2]
            tr = max(bar['high'] - bar['low'],
                     abs(bar['high'] - prev['close']),
                     abs(bar['low'] - prev['close']))
        else:
            tr = bar['high'] - bar['low']
        self.atr_window.append(tr)
        current_atr = (sum(self.atr_window) / len(self.atr_window)) if self.atr_window else 0

        avg_vol = sum(self.volume_window) / len(self.volume_window) if self.volume_window else 0
        cur_vol = float(bar.get('volume', 0) or 0)
        volume_surge = True if (avg_vol <= 0 or cur_vol <= 0) else (cur_vol > avg_vol * self.volume_multiplier)

        if len(self.window) > self.momentum_period:
            past_close = list(self.window)[-1 - self.momentum_period]['close']
            momentum = (bar['close'] - past_close) / past_close if past_close else 0.0
        else:
            momentum = 0.0

        px = max(bar['close'], 1e-9)
        atr_scaled = (current_atr / px) * self.atr_mom_mult if self.atr_mom_mult > 0 else 0.0
        entry_thr = max(self.entry_epsilon, self.momentum_epsilon) + atr_scaled
        exit_thr = max(self.exit_epsilon, self.momentum_epsilon) + atr_scaled

        positive_momentum = momentum >= -self.momentum_epsilon

        self.last_avg_vol = avg_vol
        self.last_current_atr = current_atr
        self.last_momentum = momentum
        self.last_volume_surge = volume_surge
        self.last_positive_momentum = positive_momentum
        self.last_window_len = len(self.window)

        can_trade = (self._bar_index - self._last_trade_bar) >= self.cooldown_bars

        # Track peak after entry
        if self.entry_price is not None:
            self.entry_peak_price = max(self.entry_peak_price or self.entry_price, bar['close'])

        # ENTRY (with cash gating if account context is available)
        if self.entry_price is None:
            # cash gates
            if self.usd_free is not None and self.usd_total is not None:
                cash_ok = (self.usd_free >= self.min_cash_usd) and ((self.usd_free / max(self.usd_total, 1e-9)) >= self.min_cash_pct_to_buy)
            else:
                cash_ok = True
            if can_trade and cash_ok and volume_surge and (momentum >= entry_thr):
                self.entry_price = bar['close']
                self.entry_bar_index = self._bar_index
                self.entry_peak_price = self.entry_price
                self._last_trade_bar = self._bar_index
                return 'buy'
        else:
            held_bars = self._bar_index - (self.entry_bar_index or self._bar_index)

            # Compute PnL using live avg price if provided; else entry price
            basis = None
            if self.pos_qty and self.pos_avg_price:
                basis = self.pos_avg_price
            elif self.entry_price:
                basis = self.entry_price
            pnl = (bar['close'] / basis - 1.0) if basis else 0.0

            # STOP-LOSS
            if bar['close'] <= (basis or self.entry_price) * (1 - self.stop_loss_pct):
                self.entry_price = None
                self.entry_bar_index = None
                self.entry_peak_price = None
                self._last_trade_bar = self._bar_index
                return 'sell'

            # Trailing stop: lock profit from peak
            if self.entry_peak_price and bar['close'] < self.entry_peak_price and self.trailing_stop_pct > 0:
                drop = 1.0 - (bar['close'] / self.entry_peak_price)
                if drop >= self.trailing_stop_pct and bar['close'] > (basis or self.entry_price):
                    self.entry_price = None
                    self.entry_bar_index = None
                    self.entry_peak_price = None
                    self._last_trade_bar = self._bar_index
                    return 'sell'

            # TAKE-PROFIT using account position basis if available
            if pnl >= self.take_profit_pct and (self.tp_immediate or momentum <= 0):
                self.entry_price = None
                self.entry_bar_index = None
                self.entry_peak_price = None
                self._last_trade_bar = self._bar_index
                return 'sell'

            # Normal exit only after min_hold_bars
            if held_bars >= self.min_hold_bars and can_trade and (momentum <= -exit_thr):
                self.entry_price = None
                self.entry_bar_index = None
                self.entry_peak_price = None
                self._last_trade_bar = self._bar_index
                return 'sell'

        return None
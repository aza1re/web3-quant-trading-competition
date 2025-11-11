import numpy as np
from collections import deque

# prefer xgboost if available, otherwise use sklearn RandomForest
try:
    from xgboost import XGBClassifier
    _MODEL = "xgboost"
except Exception:
    from sklearn.ensemble import RandomForestClassifier as XGBClassifier
    _MODEL = "sklearn_rf"


class ConvertedXGBoost:
    """
    Small offline replacement for QC xgboost alpha:
    - feeds bars via update(bar)
    - accumulates features in rolling windows
    - trains a classifier once enough samples collected
    - predict() returns 'buy' / 'sell' / None using probability threshold
    """

    def __init__(self,
                 volume_period=10,
                 atr_period=14,
                 momentum_period=5,
                 lookback_for_labels=5,
                 min_train_samples=50,
                 prob_thresh=0.6,
                 random_state=42):
        self.volume_period = volume_period
        self.atr_period = atr_period
        self.momentum_period = momentum_period
        self.lookback_for_labels = lookback_for_labels
        self.min_train_samples = min_train_samples
        self.prob_thresh = prob_thresh

        self.window_size = max(volume_period, atr_period, momentum_period) + 10
        self.window = deque(maxlen=self.window_size)  # store bars
        self.volume_window = deque(maxlen=volume_period)
        self.atr_window = deque(maxlen=atr_period)

        self.X = []
        self.y = []

        self.model = None
        self.random_state = random_state

    def _tr(self, cur, prev):
        if prev is None:
            return cur['high'] - cur['low']
        return max(cur['high'] - cur['low'],
                   abs(cur['high'] - prev['close']),
                   abs(prev['close'] - cur['low']))

    def _extract_features_from_index(self, idx_from_end=0):
        # idx_from_end=0 means most recent bar at window[-1]
        if len(self.window) < max(self.volume_period, self.atr_period, self.momentum_period) + idx_from_end + 1:
            return None
        i = -1 - idx_from_end
        bar = list(self.window)[i]
        # volume ratio
        if len(self.volume_window) >= self.volume_period:
            avg_vol = np.mean(list(self.volume_window)[-self.volume_period:])
            vol_ratio = bar['volume'] / avg_vol if avg_vol > 0 else 1.0
        else:
            vol_ratio = 1.0
        # ATR ratio
        if len(self.atr_window) >= self.atr_period:
            cur_atr = np.mean(list(self.atr_window)[-self.atr_period:])
            avg_atr = np.mean(list(self.atr_window)) if len(self.atr_window) else cur_atr
            atr_ratio = cur_atr / avg_atr if avg_atr > 0 else 1.0
        else:
            atr_ratio = 1.0
        # momentum
        if len(self.window) > self.momentum_period + idx_from_end:
            past = list(self.window)[i - self.momentum_period]
            momentum = (bar['close'] - past['close']) / past['close'] if past['close'] != 0 else 0.0
        else:
            momentum = 0.0
        # simple returns
        price_change_1 = (bar['close'] - bar['open']) / bar['open'] if bar['open'] != 0 else 0.0
        price_change_3 = 0.0
        if len(self.window) > 3 + idx_from_end:
            past3 = list(self.window)[i - 3]
            price_change_3 = (bar['close'] - past3['close']) / past3['close'] if past3['close'] != 0 else 0.0

        return [vol_ratio, atr_ratio, momentum, price_change_1, price_change_3]

    def update(self, bar: dict):
        """
        bar: {'time','open','high','low','close','volume'}
        - call this for each bar
        - returns a dict {'train':bool, 'pred': 'buy'/'sell'/None, 'prob':float}
        """
        prev = self.window[-1] if len(self.window) else None
        self.window.append(bar)

        # update rolling windows
        if len(self.volume_window) < self.volume_period:
            self.volume_window.append(bar['volume'])
        else:
            tmp = list(self.volume_window)[1:] + [bar['volume']]
            self.volume_window.clear()
            for v in tmp:
                self.volume_window.append(v)

        tr = self._tr(bar, prev)
        if len(self.atr_window) < self.atr_period:
            self.atr_window.append(tr)
        else:
            tmp = list(self.atr_window)[1:] + [tr]
            self.atr_window.clear()
            for v in tmp:
                self.atr_window.append(v)

        # prepare feature for the observation that will be labeled LATER (lookahead)
        feat = self._extract_features_from_index(0)
        if feat is not None:
            self.X.append(feat)

        # label generation: when we have lookback bars available, label earlier sample
        # label = 1 (buy) if price up by >1.5% over lookback horizon, else 0
        if len(self.window) > self.lookback_for_labels + 1:
            # label the sample that occurred lookback bars ago
            recent = list(self.window)
            past_idx = -1 - self.lookback_for_labels
            future_idx = -1
            past_price = recent[past_idx]['close']
            future_price = recent[future_idx]['close']
            label = 1 if future_price > past_price * 1.015 else 0
            # ensure X has matching index
            if len(self.X) >= self.lookback_for_labels + 1:
                label_index = len(self.X) - 1 - self.lookback_for_labels
                if 0 <= label_index < len(self.X):
                    self.y.append(label)

        train_flag = False
        pred = None
        prob = 0.0

        # train if enough labeled samples and model not trained yet
        if len(self.y) >= self.min_train_samples and self.model is None:
            try:
                self.model = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss', random_state=self.random_state)
            except TypeError:
                # sklearn RF fallback class alias
                self.model = XGBClassifier(n_estimators=50, random_state=self.random_state)
            X_arr = np.array(self.X[-len(self.y):])
            y_arr = np.array(self.y)
            try:
                self.model.fit(X_arr, y_arr)
                train_flag = True
            except Exception:
                self.model = None

        # predict on most recent features if model ready
        if self.model is not None and feat is not None:
            probs = None
            try:
                probs = self.model.predict_proba([feat])[0]
                prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            except Exception:
                try:
                    pred_label = int(self.model.predict([feat])[0])
                    prob = 1.0 if pred_label == 1 else 0.0
                except Exception:
                    prob = 0.0

            if prob >= self.prob_thresh:
                pred = 'buy'
            else:
                pred = None

        return {'train': train_flag, 'pred': pred, 'prob': prob, 'model_type': _MODEL}
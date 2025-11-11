import time
import hmac
import hashlib
import requests
from typing import Optional

BASE_URL = "https://mock-api.roostoo.com"

def _ts_ms() -> str:
    return str(int(time.time() * 1000))

def _build_sorted_params(payload: dict) -> str:
    # sort keys and join as key=value&...
    keys = sorted(payload.keys())
    return "&".join(f"{k}={payload[k]}" for k in keys)

class RoostooClient:
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, base_url: str = BASE_URL):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip('/')

    def _sign(self, total_params: str) -> str:
        if not self.secret_key:
            return ""
        return hmac.new(self.secret_key.encode('utf-8'),
                        total_params.encode('utf-8'),
                        hashlib.sha256).hexdigest()

    def _symbol_to_pair(self, symbol: str) -> str:
        # Accept BTCUSDT, BTCUSD, BTC -> convert to BTC/USD
        s = symbol.upper()
        if '/' in s:
            return s
        if s.endswith('USDT'):
            base = s[:-4]
        elif s.endswith('USD'):
            base = s[:-3]
        else:
            base = s
        return f"{base}/USD"

    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = 'MARKET', price: Optional[float] = None):
        """
        POST /v3/place_order (RCL_TopLevelCheck)
        Parameters (body x-www-form-urlencoded): pair, side, type, quantity, timestamp, [price]
        Headers: RST-API-KEY, MSG-SIGNATURE
        """
        url = f"{self.base_url}/v3/place_order"
        pair = self._symbol_to_pair(symbol)

        payload = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
            "timestamp": _ts_ms()
        }
        if order_type and order_type.upper() == "LIMIT" and price is not None:
            payload["price"] = str(price)

        total_params = _build_sorted_params(payload)
        signature = self._sign(total_params)

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        if self.api_key:
            headers["RST-API-KEY"] = self.api_key
        if signature:
            headers["MSG-SIGNATURE"] = signature

        resp = requests.post(url, headers=headers, data=total_params, timeout=15)
        resp.raise_for_status()
        return resp.json()
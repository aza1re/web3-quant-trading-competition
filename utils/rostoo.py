import time
import hmac
import hashlib
import requests
from typing import Optional, Dict, Tuple, Any

BASE_URL = "https://mock-api.roostoo.com"


def _ts_ms() -> str:
    return str(int(time.time() * 1000))


def _build_sorted_params(payload: Dict[str, Any]) -> str:
    # sort keys and join as key=value&...
    keys = sorted(payload.keys())
    return "&".join(f"{k}={payload[k]}" for k in keys)


class RoostooClient:
    """
    Lightweight client for Roostoo mock API that follows the public docs:
      - GET /v3/serverTime
      - GET /v3/exchangeInfo
      - GET /v3/ticker (requires timestamp param)
      - Signed endpoints (RCL_TopLevelCheck) use RST-API-KEY + MSG-SIGNATURE header
    """

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

    def _signed_headers_and_body(self, payload: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Any], str]:
        """
        Return (headers, payload_dict, total_params_string) for signed endpoints.
        Previously only returned (headers, total_params) which caused GET requests
        to pass a raw string as params=..., preventing proper query encoding.
        """
        pl = dict(payload)
        if 'timestamp' not in pl:
            pl['timestamp'] = _ts_ms()
        total_params = _build_sorted_params(pl)
        signature = self._sign(total_params)
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if self.api_key:
            headers["RST-API-KEY"] = self.api_key
        if signature:
            headers["MSG-SIGNATURE"] = signature
        return headers, pl, total_params

    def _symbol_to_pair(self, symbol: str) -> str:
        # Accept BTCUSDT, BTCUSD, BTC or BTC/USD -> convert to BTC/USD
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

    # Public endpoints ----------------------------------------------------
    def server_time(self) -> Optional[Dict]:
        try:
            resp = requests.get(f"{self.base_url}/v3/serverTime", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def exchange_info(self) -> Optional[Dict]:
        try:
            resp = requests.get(f"{self.base_url}/v3/exchangeInfo", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def ticker(self, pair: Optional[str] = None) -> Optional[Dict]:
        params = {'timestamp': _ts_ms()}
        if pair:
            params['pair'] = pair
        try:
            resp = requests.get(f"{self.base_url}/v3/ticker", params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    # Signed endpoints ----------------------------------------------------
    def balance(self) -> Optional[Dict]:
        headers, payload, _ = self._signed_headers_and_body({})
        try:
            resp = requests.get(f"{self.base_url}/v3/balance", headers=headers, params=payload, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def pending_count(self) -> Optional[Dict]:
        headers, payload, _ = self._signed_headers_and_body({})
        try:
            resp = requests.get(f"{self.base_url}/v3/pending_count", headers=headers, params=payload, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def place_order(self, pair_or_coin: Optional[str] = None, side: Optional[str] = None,
                    quantity: Optional[float] = None, price: Optional[float] = None,
                    order_type: Optional[str] = None, symbol: Optional[str] = None, **kwargs) -> Optional[Dict]:
        """
        Place a LIMIT or MARKET order.
        Accepts pair like "BTC/USD" or base coin "BTC" (will be converted to BTC/USD).
        Compatible with callers using keyword 'symbol' or 'pair'.
        """
        # support symbol= or pair= aliases
        pair_or_coin = pair_or_coin or symbol or kwargs.get('pair') or kwargs.get('symbol')
        if pair_or_coin is None:
            raise TypeError("pair or symbol must be provided")

        # normalize quantity possibly passed as string
        try:
            qty_val = float(quantity) if quantity is not None else None
        except Exception:
            qty_val = None

        pair = pair_or_coin if '/' in pair_or_coin else self._symbol_to_pair(pair_or_coin)
        if order_type is None:
            order_type = "LIMIT" if price is not None else "MARKET"

        if order_type.upper() == 'LIMIT' and price is None:
            raise ValueError("LIMIT orders require a price")

        if qty_val is None:
            raise TypeError("quantity must be provided and numeric")

        payload = {
            'pair': pair,
            'side': side.upper() if side is not None else "BUY",
            'type': order_type.upper(),
            'quantity': str(qty_val)
        }
        if order_type.upper() == 'LIMIT' and price is not None:
            payload['price'] = str(price)

        headers, _, total_params = self._signed_headers_and_body(payload)
        try:
            resp = requests.post(f"{self.base_url}/v3/place_order", headers=headers, data=total_params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            # bubble up dict-like error for visibility
            try:
                return {"error": str(e), "text": e.response.text if getattr(e, "response", None) is not None else None}
            except Exception:
                return {"error": str(e)}

    def query_order(self, order_id: Optional[str] = None, pair: Optional[str] = None, pending_only: Optional[bool] = None) -> Optional[Dict]:
        payload: Dict[str, Any] = {}
        if order_id:
            payload['order_id'] = str(order_id)
        elif pair:
            payload['pair'] = pair
            if pending_only is not None:
                payload['pending_only'] = 'TRUE' if pending_only else 'FALSE'

        headers, _, total_params = self._signed_headers_and_body(payload)
        try:
            resp = requests.post(f"{self.base_url}/v3/query_order", headers=headers, data=total_params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def cancel_order(self, order_id: Optional[str] = None, pair: Optional[str] = None) -> Optional[Dict]:
        payload: Dict[str, Any] = {}
        if order_id:
            payload['order_id'] = str(order_id)
        elif pair:
            payload['pair'] = pair
        headers, _, total_params = self._signed_headers_and_body(payload)
        try:
            resp = requests.post(f"{self.base_url}/v3/cancel_order", headers=headers, data=total_params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None
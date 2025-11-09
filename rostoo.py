import requests
import time
import hmac
import hashlib

BASE_URL = "https://mock-api.roostoo.com"

class RoostooClient:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key

    def _get_timestamp(self):
        return str(int(time.time() * 1000))

    def _get_signed_headers(self, payload: dict = {}):
        payload = dict(payload)
        payload['timestamp'] = self._get_timestamp()
        sorted_keys = sorted(payload.keys())
        total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)

        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            total_params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        headers = {
            'RST-API-KEY': self.api_key,
            'MSG-SIGNATURE': signature
        }

        return headers, payload, total_params

    # Public endpoints
    def check_server_time(self):
        url = f"{BASE_URL}/v3/serverTime"
        res = requests.get(url)
        res.raise_for_status()
        return res.json()

    def get_exchange_info(self):
        url = f"{BASE_URL}/v3/exchangeInfo"
        res = requests.get(url)
        res.raise_for_status()
        return res.json()

    def get_ticker(self, pair=None):
        url = f"{BASE_URL}/v3/ticker"
        params = {'timestamp': self._get_timestamp()}
        if pair:
            params['pair'] = pair
        res = requests.get(url, params=params)
        res.raise_for_status()
        return res.json()

    # Signed endpoints
    def get_balance(self):
        url = f"{BASE_URL}/v3/balance"
        headers, payload, _ = self._get_signed_headers()
        res = requests.get(url, headers=headers, params=payload)
        res.raise_for_status()
        return res.json()

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None, params: dict = None):
        url = f"{BASE_URL}/v3/order"
        data = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'price': price
        }
        if params:
            data.update(params)
        headers, payload, _ = self._get_signed_headers(data)
        res = requests.post(url, headers=headers, json=data)
        res.raise_for_status()
        return res.json()

    def cancel_order(self, order_id: str):
        url = f"{BASE_URL}/v3/order"
        data = {'orderId': order_id}
        headers, payload, _ = self._get_signed_headers(data)
        res = requests.delete(url, headers=headers, json=data)
        res.raise_for_status()
        return res.json()

    def get_order_status(self, order_id: str):
        url = f"{BASE_URL}/v3/order"
        params = {'orderId': order_id}
        headers, payload, _ = self._get_signed_headers(params)
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        return res.json()

    def get_open_orders(self, symbol: str = None):
        url = f"{BASE_URL}/v3/openOrders"
        params = {}
        if symbol:
            params['symbol'] = symbol
        headers, payload, _ = self._get_signed_headers(params)
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        return res.json()

    def get_all_orders(self, symbol: str = None):
        url = f"{BASE_URL}/v3/allOrders"
        params = {}
        if symbol:
            params['symbol'] = symbol
        headers, payload, _ = self._get_signed_headers(params)
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        return res.json()
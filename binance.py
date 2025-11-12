import requests
import pandas as pd

def fetch_klines(symbol='BTCUSDT', interval='1h', limit=1000):
    """
    Fetches historical k-line (candlestick) data from Binance.
    """
    url_primary = "https://api.binance.com/api/v3/klines"
    url_fallback = "https://api.binance.us/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    headers = {
        "User-Agent": "Mozilla/5.0 (alpha-bot)",
        "Accept": "application/json"
    }
    for url in (url_primary, url_fallback):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            if r.status_code == 451:
                continue  # try fallback
            r.raise_for_status()
            data = r.json()
            if not data:
                continue
            rows = []
            for k in data:
                rows.append({
                    "open_time": pd.to_datetime(k[0], unit="ms", utc=True),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5])
                })
            return pd.DataFrame(rows)
        except Exception:
            continue
    return pd.DataFrame()  # empty on failure
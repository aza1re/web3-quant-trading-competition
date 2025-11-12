import time
import requests
import pandas as pd

HORUS_BASE = "https://api-horus.com"

# Map non-standard ticker fragments to Horus asset codes
ASSET_ALIASES = {
    "S": "SONIC",          # result of SUSDT stripping -> S
    "SUSDT": "SONIC",
    "SONIC": "SONIC",
    "SONICUSDT": "SONIC"
}

def fetch_klines(symbol: str = 'BTCUSDT', interval: str = '1h', limit: int = 500, api_key: str = None):
    """
    Fetch simple kline-like DataFrame from Horus price endpoint.
    Note: Horus returns price series (timestamp + price). This function
    maps price -> OHLC by duplicating close into open/high/low when OHLC
    not available. That is an approximation — use true OHLC when provider supports it.
    Returns DataFrame with columns: open_time, open, high, low, close, volume
    """
    if interval not in ('1d', '1h', '15m'):
        raise ValueError("interval must be one of '1d','1h','15m'")

    asset_raw = symbol.upper()
    asset = asset_raw.replace('USDT', '').replace('USD', '')
    # apply alias mapping
    asset = ASSET_ALIASES.get(asset_raw, ASSET_ALIASES.get(asset, asset))

    interval_seconds = 86400 if interval == '1d' else 3600 if interval == '1h' else 900
    end_ts = int(time.time())
    start_ts = end_ts - int(limit) * interval_seconds

    url = f"{HORUS_BASE}/market/price"
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    params = {
        "asset": asset,
        "interval": interval,
        "start": start_ts,
        "end": end_ts,
        "format": "json"
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Horus fetch error: {e}")
        return None
    except ValueError:
        print("Horus returned non-JSON response")
        return None

    if not data:
        return None

    df = pd.DataFrame(data)

    # common Horus payload: {'timestamp': ..., 'price': ...}
    if 'timestamp' in df.columns and ('price' in df.columns or 'close' in df.columns):
        price_col = 'price' if 'price' in df.columns else 'close'
        df['open_time'] = pd.to_datetime(df['timestamp'], unit='s')
        df['close'] = pd.to_numeric(df[price_col], errors='coerce')
        # approximate OHLC where only close exists
        df['open'] = df['close']
        df['high'] = df['close']
        df['low'] = df['close']
        df['volume'] = 0.0 if 'volume' not in df.columns else pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)
        return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]

    # if Horus returns different shape, try to infer: first col = timestamp, second = price
    cols = list(df.columns)
    if len(cols) >= 2:
        df['open_time'] = pd.to_datetime(df[cols[0]], unit='s', errors='coerce')
        df['close'] = pd.to_numeric(df[cols[1]], errors='coerce')
        df['open'] = df['close']
        df['high'] = df['close']
        df['low'] = df['close']
        df['volume'] = 0.0
        return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]

    print("Unexpected Horus response format, cannot build klines")
    return None
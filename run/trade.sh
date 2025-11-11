#!/usr/bin/env bash
API_KEY="S5dF9gHjK3lL1ZxCV7bN2mQwE0rT4yUiP6oA8sDdF2gJ0hKlZ9xC5vBnM4qW3eRt"
API_SECRET="Y3uI7oPaS9dF1gHjK5lL6ZxCV0bN2mQwE4rT8yUiP6oA3sDdF7gJ0hKlZ1xC5vBn"

# Edit quantity and side as needed. By default this will execute (force).
python3 trade.py --api-key "$API_KEY" --api-secret "$API_SECRET" --symbol BTCUSDT --quantity 0.001 --side BUY --force
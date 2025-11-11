#!/usr/bin/env bash
# Run converted BTC backtest (one-week / one-month examples)
YOUR_HORUS_KEY="a0ff981638cc60f41d91bcd588b782088d28d04a614a8ad633cee70f660b967a"

# default: 1h bars, one-week
python btc_converted/main.py --source binance --symbol BTCUSDT --interval 1h --limit 168 --apikey "$YOUR_HORUS_KEY" --capital 50000 --risk-mult 1 

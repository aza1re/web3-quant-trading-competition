import os
import argparse
from rostoo import RoostooClient

def main():
    parser = argparse.ArgumentParser(description="Place one MARKET trade and immediately close it (dry-run by default)")
    parser.add_argument('--api-key', '-k', default=os.environ.get('API_KEY'))
    parser.add_argument('--api-secret', '-s', default=os.environ.get('API_SECRET'))
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--quantity', type=float, default=0.001, help='Quantity to buy/sell')
    parser.add_argument('--side', choices=['BUY','SELL'], default='BUY', help='First order side')
    parser.add_argument('--force', action='store_true', help='Execute orders (omit for dry-run)')
    args = parser.parse_args()

    if not args.api_key or not args.api_secret:
        print("Missing API credentials. Provide --api-key/--api-secret or set env vars API_KEY/API_SECRET")
        return

    client = RoostooClient(api_key=args.api_key, secret_key=args.api_secret)

    print(f"Prepared: MARKET {args.side} {args.symbol} qty={args.quantity:.8f}")
    if not args.force:
        print("Dry-run mode. Use --force to actually place orders.")
        return

    try:
        resp = client.place_order(symbol=args.symbol, side=args.side, quantity=args.quantity, order_type='MARKET')
        print("Order response:", resp)
    except Exception as e:
        print("First order failed:", e)
        return

    # place opposite order to close the position
    opp = 'SELL' if args.side == 'BUY' else 'BUY'
    try:
        resp2 = client.place_order(symbol=args.symbol, side=opp, quantity=args.quantity, order_type='MARKET')
        print("Opposite order response:", resp2)
    except Exception as e:
        print("Opposite order failed:", e)

if __name__ == "__main__":
    main()
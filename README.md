# Web3 Quant Trading Competition Bot

## Overview

This repository implements a momentum and volatility breakout trading bot for the Web3 Quant Trading Competition. The bot is designed for both live trading and robust backtesting, trading BTC, ETH, and TRX with advanced risk controls.

## Trading Strategy

- **Volume Surge Filter:** Trades are considered only when the current bar’s volume exceeds a moving average by a configurable multiplier.
- **Momentum Entry:** Entry signals require positive momentum above a threshold, adjusted for volatility (ATR).
- **ATR (Average True Range):** Used to scale momentum thresholds and filter out noise.
- **Stop-Loss:** Positions are exited if price drops below a configurable percentage from entry or average price.
- **Trailing Stop:** Locks in profits by selling if price falls a set percentage from the peak since entry.
- **Take-Profit:** Optionally exits positions when a profit threshold is reached.
- **Minimum Hold Bars:** Ensures trades are held for a minimum number of bars before exit.
- **Cooldown Bars:** Prevents immediate re-entry after a trade to avoid whipsaw.
- **Risk Controls:** Limits on max open symbols and daily drawdown to prevent overexposure.

All strategy parameters (volume multiplier, ATR multiplier, momentum thresholds, stop-loss, take-profit, etc.) were configured in run/btc.sh with environment variables in the AWS instance.

## Backtesting workflow

We originally backtested the strategy using Binance historical OHLCV because Binance provides full OHLCV (open/high/low/close/volume) data and the strategy uses a volume‑surge filter. After preparing for live deployment on Roostoo, we discovered an important mismatch:

- Roostoo's live market endpoints do not provide volume in their ticker/quote data. That means a backtest run on Binance OHLCV will exercise volume-based logic that will not be available in live trading on Roostoo, producing unrealistic expectations.

Because of that mismatch we changed our backtesting approach:

- Keep Binance as a volume‑rich data source for experiments where volume‑based filters must be validated.
- For validating the exact live behavior we run backtests with data shaped like Roostoo (i.e., without volume). We use Horus for these runs so the offline replay mirrors the live API shape and limitations.
  - In practice this means running two kinds of backtests:
    1. Binance (with volume) — to validate and tune volume‑sensitive rules.
    2. Roostoo‑shaped with Horus (no volume) — to validate live‑deployment behavior and ensure signal→order mappings match what will happen on Roostoo.

## Live Trading Workflow

- For live trading, the bot uses the Roostoo API for price and order execution.
- Since Roostoo does not provide volume, the volume surge filter is disabled or bypassed in live mode (the code seeds or bypasses volume checks to reflect the live feed).
- The live loop:
  1. Normalize symbol -> pair via `_pair_from_symbol`.
  2. Instantiate `HybridAlphaConverted` per symbol with configured params.
  3. Warm up alpha windows using historical bars when available (Binance/Horus warmup).
  4. Aggregate bars at the configured interval and call `alpha.update(bar)` on each closed bar.
  5. When a 'buy' or 'sell' signal is emitted, apply sizing, risk filters, rounding, and call `_place_order_safe` with the normalized pair and mocked/real `RoostooClient`.
  6. On fills, update the local portfolio and reconcile with exchange balances.

### How parameters are applied in live
- `volume_multiplier` / `vol-mult`:
  - Controls the volume surge gate in `HybridAlphaConverted`. In live on Roostoo, missing volume leads to seeding from warmup data or bypassing the gate.
- `atr_multiplier`, `atr_mom_mult`:
  - Scale ATR contributions to momentum thresholds used to decide entries/exits.
- `momentum_period`, `momentum_epsilon`, `entry_epsilon`, `exit_epsilon`:
  - Define momentum calculation window and thresholds for triggering signals.
- `cooldown_bars`:
  - Minimum bars between trades enforced by alpha.
- `min_hold_bars`:
  - Minimum bars to hold a position before allowing normal exit.
- `stop_loss_pct`, `trailing_stop_pct`, `take_profit_pct`, `tp_immediate`:
  - Manage exits: immediate stop-loss, trailing stop behavior, and optional take-profit.
- `alloc` (allocation fraction) and `risk_mult`:
  - Determine position sizing: target_value = portfolio_value * alloc * risk_mult; converted to quantity and rounded to exchange step size.
- `fee`:
  - Applied when updating local portfolio after fills and when calculating affordability.
- `max_open_symbols`, `max_daily_dd_pct`:
  - Prevent new entries when limits are reached.
- `poll_secs` / `interval`:
  - Control fetch and aggregation timing.
- `force` / `do_check`:
  - `force` controls whether real orders are submitted; `do_check` runs a small connectivity/order-signature test on

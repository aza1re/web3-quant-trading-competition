import sys, os, argparse, traceback, math, json
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# repo root
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

# module paths
sys.path.insert(0, os.path.join(REPO_ROOT, "btc_converted"))

from btc_converted.alpha import HybridAlphaConverted  # uses relaxed momentum already
from utils.horus import fetch_klines as fetch_horus_klines
from utils.binance import fetch_klines as fetch_binance_klines

class SimplePortfolio:
    def __init__(self, cash=100000.0, fee=0.0001, risk_mult=1.0):
        self.cash = float(cash)
        self.fee = float(fee)
        self.positions = 0.0
        self.avg_price = None
        self.trade_count = 0
        self.risk_mult = float(risk_mult)
        self.trade_sizes: List[float] = []
        self.trade_values: List[float] = []

    def portfolio_value(self, price: float) -> float:
        return self.cash + self.positions * price

    def buy_allocation(self, price: float, allocation: float):
        pv = self.portfolio_value(price)
        target_value = pv * allocation * self.risk_mult
        if target_value <= 0: return
        max_affordable = self.cash / (1.0 + self.fee) if self.cash > 0 else 0.0
        trade_value = min(target_value, max_affordable)
        if trade_value <= 0: return
        qty = trade_value / price
        cost = qty * price
        fee_amt = abs(cost) * self.fee
        self.cash -= (cost + fee_amt)
        prev_qty = self.positions
        if prev_qty == 0 or self.avg_price is None:
            self.avg_price = price
        else:
            self.avg_price = (self.avg_price * prev_qty + cost) / (prev_qty + qty)
        self.positions += qty
        self.trade_count += 1
        self.trade_sizes.append(qty)
        self.trade_values.append(cost)

    def sell_all(self, price: float):
        if self.positions <= 0: return
        proceeds = self.positions * price
        fee_amt = abs(proceeds) * self.fee
        self.cash += (proceeds - fee_amt)
        self.trade_sizes.append(self.positions)
        self.trade_values.append(proceeds)
        self.positions = 0.0
        self.avg_price = None
        self.trade_count += 1

def to_pd_timestamp(t) -> Optional[pd.Timestamp]:
    try:
        if isinstance(t, (int, float)):
            if t > 1e12:
                return pd.to_datetime(int(t), unit="ms", utc=True)
            if t > 1e9:
                return pd.to_datetime(int(t), unit="s", utc=True)
        return pd.to_datetime(t, utc=True)
    except Exception:
        try:
            return pd.to_datetime(t)
        except Exception:
            return None

def fetch_data(source: str, symbol: str, interval: str, limit: int, apikey: Optional[str]) -> pd.DataFrame:
    if source.lower() == "horus":
        df = fetch_horus_klines(symbol=symbol, interval=interval, limit=limit, api_key=apikey)
    else:
        df = fetch_binance_klines(symbol=symbol, interval=interval, limit=limit)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned from source for {symbol}.")
    if "open_time" in df.columns:
        df = df.rename(columns={"open_time": "time"})
    return df.sort_values("time").reset_index(drop=True)

def build_bar(row) -> Dict[str, Any]:
    return {
        "time": row["time"],
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]) if "volume" in row and not pd.isna(row["volume"]) else 0.0
    }

class MultiPortfolio:
    def __init__(self, cash=100000.0, fee=0.0001, risk_mult=1.0):
        self.cash = float(cash)
        self.fee = float(fee)
        self.risk_mult = float(risk_mult)
        self.pos = {}        # sym -> qty
        self.avgp = {}       # sym -> avg price
        self.trade_count = 0
        self.trade_sizes = []
        self.trade_values = []

    def value(self, prices: Dict[str, float]) -> float:
        v = self.cash
        for s, q in self.pos.items():
            px = prices.get(s)
            if px is not None:
                v += q * px
        return v

    def buy(self, sym: str, price: float, allocation: float, prices: Dict[str, float]):
        pv = self.value(prices)
        target_value = pv * allocation * self.risk_mult
        if target_value <= 0 or price <= 0: return
        max_affordable = self.cash / (1.0 + self.fee)
        trade_value = min(target_value, max_affordable)
        if trade_value <= 0: return
        qty = trade_value / price
        cost = qty * price
        fee_amt = abs(cost) * self.fee
        self.cash -= (cost + fee_amt)
        prev_qty = self.pos.get(sym, 0.0)
        prev_avg = self.avgp.get(sym)
        new_avg = price if prev_qty == 0 or prev_avg is None else (prev_avg * prev_qty + cost) / (prev_qty + qty)
        self.pos[sym] = prev_qty + qty
        self.avgp[sym] = new_avg
        self.trade_count += 1
        self.trade_sizes.append(qty)
        self.trade_values.append(cost)

    def sell_all(self, sym: str, price: float):
        qty = self.pos.get(sym, 0.0)
        if qty <= 0: return
        proceeds = qty * price
        fee_amt = abs(proceeds) * self.fee
        self.cash += (proceeds - fee_amt)
        self.trade_count += 1
        self.trade_sizes.append(qty)
        self.trade_values.append(proceeds)
        self.pos[sym] = 0.0
        self.avgp[sym] = None

def run_backtest_single(symbol: str,
                        interval: str,
                        limit: int,
                        source: str,
                        capital: float,
                        fee: float,
                        risk_mult: float,
                        allocation: float,
                        apikey: Optional[str],
                        vol_mult: float,
                        atr_mult: float,
                        mom_period: int,
                        momentum_epsilon: float,
                        stop_loss: float,
                        entry_epsilon: float,
                        exit_epsilon: float,
                        cooldown_bars: int,
                        atr_mom_mult: float,
                        min_hold_bars: int,
                        take_profit_pct: float,
                        trailing_stop_pct: float,
                        tp_immediate: bool,
                        verbose: bool):
    df = fetch_data(source, symbol, interval, limit, apikey)
    alpha = HybridAlphaConverted(
        volume_period=5, atr_period=10, momentum_period=mom_period,
        volume_multiplier=vol_mult, atr_multiplier=atr_mult,
        stop_loss_pct=stop_loss, momentum_epsilon=momentum_epsilon,
        entry_epsilon=entry_epsilon, exit_epsilon=exit_epsilon,
        cooldown_bars=int(cooldown_bars), atr_mom_mult=atr_mom_mult,
        min_hold_bars=min_hold_bars, take_profit_pct=take_profit_pct,
        trailing_stop_pct=trailing_stop_pct, tp_immediate=tp_immediate
    )
    port = SimplePortfolio(cash=capital, fee=fee, risk_mult=risk_mult)
    times: List[Any] = []
    prices: List[float] = []
    equity: List[float] = []
    buys: List[Tuple[Any, float]] = []
    sells: List[Tuple[Any, float]] = []

    for _, row in df.iterrows():
        bar = build_bar(row)
        sig = alpha.update(bar)
        price = bar["close"]
        times.append(bar["time"])
        prices.append(price)

        if sig == "buy":
            port.buy_allocation(price, allocation)
            buys.append((bar["time"], price))
            if verbose: print(f"[BUY] {bar['time']} {symbol} {price:.2f}")
        elif sig == "sell":
            port.sell_all(price)
            sells.append((bar["time"], price))
            if verbose: print(f"[SELL] {bar['time']} {symbol} {price:.2f}")

        equity.append(port.portfolio_value(price))

    if port.positions > 0:
        port.sell_all(prices[-1])
        equity[-1] = port.portfolio_value(prices[-1])

    # --- print single-asset summary like multi ---
    start = times[0]
    end = times[-1]
    final_value = equity[-1]
    total_return = final_value / capital - 1.0
    days = max((pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 86400.0, 1.0)
    ann_return = (final_value / capital) ** (365.0 / days) - 1.0
    result_df = pd.DataFrame({"time": times, "equity": equity}).sort_values("time")
    rets = result_df["equity"].pct_change().dropna()
    ann_vol = (rets.std() if not rets.empty else 0.0) * math.sqrt(365.0)
    peak = result_df["equity"].cummax()
    drawdown = (result_df["equity"] - peak) / peak
    max_dd = drawdown.min() if not drawdown.empty else 0.0

    print("\nMomentum Backtest (Single-asset):")
    print(f"Symbol: {symbol}")
    print(f"Source: {source}")
    print(f"Period: {start} to {end}")
    print(f"Initial Capital: {capital:.2f}")
    print(f"Final Portfolio Value: {final_value:.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Total Trades: {len(buys) + len(sells)}  (buys={len(buys)}, sells={len(sells)})")
    print(f"Annualized Return (approx): {ann_return:.2%}")
    print(f"Annualized Volatility (approx): {ann_vol:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")

    summary = {
        "symbol": symbol,
        "source": source,
        "start": str(start),
        "end": str(end),
        "initial_capital": capital,
        "final_value": final_value,
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "max_drawdown": float(max_dd),
        "total_trades": len(buys) + len(sells),
    }
    try:
        print("\nJSON Summary:")
        print(json.dumps(summary, indent=2))
    except Exception:
        pass

    return {
        "symbol": symbol,
        "df": df,
        "times": times,
        "prices": prices,
        "equity": equity,
        "trades": len(buys) + len(sells),
        "final_cash": equity[-1],
        "avg_qty": np.mean(port.trade_sizes) if port.trade_sizes else 0.0,
        "avg_val": np.mean(port.trade_values) if port.trade_values else 0.0
    }

def run_backtest_multi(symbols: List[str],
                       interval: str,
                       limit: int,
                       source: str,
                       capital: float,
                       fee: float,
                       risk_mult: float,
                       allocation: float,
                       apikey: Optional[str],
                       vol_mult: float,
                       atr_mult: float,
                       mom_period: int,
                       momentum_epsilon: float,
                       stop_loss: float,
                       entry_epsilon: float,
                       exit_epsilon: float,
                       cooldown_bars: int,
                       atr_mom_mult: float,
                       min_hold_bars: int,
                       take_profit_pct: float,
                       trailing_stop_pct: float,
                       tp_immediate: bool,
                       max_open_symbols: int,          # NEW
                       max_daily_dd_pct: float,        # NEW
                       verbose: bool):
    data = {}
    for sym in symbols:
        try:
            df = fetch_data(source, sym, interval, limit, apikey)
            data[sym] = df
        except Exception as e:
            print(f"[WARN] skipping {sym}: {e}")
    if not data:
        raise RuntimeError("No symbols had data.")

    # per-symbol alpha
    alphas = {
        sym: HybridAlphaConverted(
            volume_period=5, atr_period=10, momentum_period=mom_period,
            volume_multiplier=vol_mult, atr_multiplier=atr_mult,
            stop_loss_pct=stop_loss, momentum_epsilon=momentum_epsilon,
            entry_epsilon=entry_epsilon, exit_epsilon=exit_epsilon,
            cooldown_bars=int(cooldown_bars), atr_mom_mult=atr_mom_mult,
            min_hold_bars=min_hold_bars, take_profit_pct=take_profit_pct,
            trailing_stop_pct=trailing_stop_pct, tp_immediate=tp_immediate
        )
        for sym in data.keys()
    }

    # unify time grid (outer union)
    dfs_idx = {sym: df.set_index("time") for sym, df in data.items()}
    all_times = sorted(set().union(*[set(df.index) for df in dfs_idx.values()]))

    port = MultiPortfolio(cash=capital, fee=fee, risk_mult=risk_mult)
    last_prices: Dict[str, float] = {sym: float(dfs_idx[sym]["close"].iloc[0]) for sym in dfs_idx}
    equity_series: List[float] = []
    time_series: List[Any] = []
    total_trades = 0
    day_equity_open = {}

    for t in all_times:
        # process bars available at this time per symbol
        for sym, df in dfs_idx.items():
            if t in df.index:
                row = df.loc[t]
                bar = {
                    "time": t,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]) if "volume" in row and not pd.isna(row["volume"]) else 0.0
                }
                sig = alphas[sym].update(bar)
                price = bar["close"]
                last_prices[sym] = price
                if sig == "buy":
                    # risk controls
                    if len([s for s,q in port.pos.items() if q > 0]) >= max_open_symbols:
                        continue
                    day_key = pd.to_datetime(t).date()
                    if day_key not in day_equity_open:
                        day_equity_open[day_key] = port.value(last_prices)
                    cur_equity = port.value(last_prices)
                    if (cur_equity - day_equity_open[day_key]) / day_equity_open[day_key] <= -max_daily_dd_pct:
                        continue
                    # per-symbol cash allocation (not portfolio value)
                    alloc_cash = port.cash * allocation * port.risk_mult
                    if alloc_cash <= 0:
                        continue
                    trade_value = min(alloc_cash, port.cash / (1 + port.fee))
                    qty = trade_value / price if price > 0 else 0
                    if qty <= 0:
                        continue
                    # execute buy (reusing existing method but override sizing logic)
                    port.cash -= trade_value * (1 + port.fee)
                    prev_qty = port.pos.get(sym, 0.0)
                    prev_avg = port.avgp.get(sym)
                    new_qty = prev_qty + qty
                    port.avgp[sym] = price if prev_qty == 0 or prev_avg is None else (prev_avg * prev_qty + trade_value) / new_qty
                    port.pos[sym] = new_qty
                    port.trade_count += 1
                    port.trade_sizes.append(qty)
                    port.trade_values.append(trade_value)
                    total_trades += 1
                    if verbose: print(f"[BUY] {t} {sym} qty={qty:.6f} px={price:.2f}")
                elif sig == "sell":
                    port.sell_all(sym, price)
                    total_trades += 1
                    if verbose: print(f"[SELL] {t} {sym} {price:.2f}")
        # snapshot equity after processing this timestamp
        time_series.append(t)
        equity_series.append(port.value(last_prices))

    # close remaining positions at last known prices for reporting
    for sym, qty in list(port.pos.items()):
        if qty > 0:
            port.sell_all(sym, last_prices.get(sym, 0.0))

    # KPIs
    start = time_series[0]
    end = time_series[-1]
    final_value = equity_series[-1]
    total_return = final_value / capital - 1.0
    days = max((pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 86400.0, 1.0)
    ann_return = (final_value / capital) ** (365.0 / days) - 1.0
    result_df = pd.DataFrame({"time": time_series, "equity": equity_series}).sort_values("time")
    rets = result_df["equity"].pct_change().dropna()
    ann_vol = (rets.std() if not rets.empty else 0.0) * math.sqrt(365.0)
    peak = result_df["equity"].cummax()
    drawdown = (result_df["equity"] - peak) / peak
    max_dd = drawdown.min() if not drawdown.empty else 0.0
    avg_trades_per_day = total_trades / days if days > 0 else 0.0
    avg_qty = np.mean(port.trade_sizes) if port.trade_sizes else 0.0
    avg_val = np.mean(port.trade_values) if port.trade_values else 0.0

    print("\nMomentum Backtest (Multi-asset, Horus-compatible):")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Source: {source}")
    print(f"Period: {start} to {end}")
    print(f"Initial Capital: {capital:.2f}")
    print(f"Final Portfolio Value: {final_value:.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Total Trades: {total_trades}")
    print(f"Avg Trades / Day: {avg_trades_per_day:.2f}")
    print(f"Annualized Return (approx): {ann_return:.2%}")
    print(f"Annualized Volatility (approx): {ann_vol:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Avg trade qty: {avg_qty:.8f}   Avg trade value (USD): {avg_val:.2f}")

    summary = {
        "symbols": symbols,
        "source": source,
        "start": str(start),
        "end": str(end),
        "initial_capital": capital,
        "final_value": final_value,
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "max_drawdown": float(max_dd),
        "total_trades": total_trades,
        "avg_trades_per_day": avg_trades_per_day,
        "avg_trade_qty": avg_qty,
        "avg_trade_value": avg_val
    }
    try:
        print("\nJSON Summary:")
        print(json.dumps(summary, indent=2))
    except Exception:
        pass

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default=os.environ.get("SYMBOL"))
    p.add_argument("--symbols", default=os.environ.get("SYMBOLS"))  # comma-separated
    p.add_argument("--interval", default=os.environ.get("INTERVAL", "1h"))
    p.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "168")))
    p.add_argument("--source", default=os.environ.get("SOURCE", "horus"))
    p.add_argument("--capital", type=float, default=float(os.environ.get("CAPITAL", "50000")))
    p.add_argument("--risk-mult", type=float, default=float(os.environ.get("RISK_MULT", "2.0")))
    p.add_argument("--allocation", type=float, default=float(os.environ.get("ALLOCATION", "0.5")))
    p.add_argument("--fee", type=float, default=float(os.environ.get("FEE", "0.0001")))
    p.add_argument("--apikey", default=os.environ.get("HORUS_API_KEY"))
    p.add_argument("--vol-mult", type=float, default=1.0)
    p.add_argument("--atr-mult", type=float, default=1.0)
    p.add_argument("--mom-period", type=int, default=3)
    p.add_argument("--momentum-epsilon", type=float, default=0.0005)
    p.add_argument("--stop-loss", type=float, default=0.05)
    p.add_argument("--entry-epsilon", type=float, default=0.0008)
    p.add_argument("--exit-epsilon", type=float, default=0.0005)
    p.add_argument("--cooldown-bars", type=int, default=1)
    p.add_argument("--atr-mom-mult", type=float, default=0.0)
    p.add_argument("--min-hold-bars", type=int, default=2)
    p.add_argument("--take-profit-pct", type=float, default=0.01)
    p.add_argument("--trailing-stop-pct", type=float, default=0.02)
    p.add_argument("--tp-immediate", action="store_true")
    p.add_argument("--max-open-symbols", type=int, default=3, help="limit concurrent long positions (multi-asset)")
    p.add_argument("--max-daily-dd-pct", type=float, default=0.05, help="daily drawdown limit; block new buys if exceeded")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        if args.symbols:
            syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
            run_backtest_multi(symbols=syms,
                               interval=args.interval,
                               limit=args.limit,
                               source=args.source,
                               capital=args.capital,
                               fee=args.fee,
                               risk_mult=args.risk_mult,
                               allocation=args.allocation,
                               apikey=args.apikey,
                               vol_mult=args.vol_mult,
                               atr_mult=args.atr_mult,
                               mom_period=args.mom_period,
                               momentum_epsilon=args.momentum_epsilon,
                               stop_loss=args.stop_loss,
                               entry_epsilon=args.entry_epsilon,
                               exit_epsilon=args.exit_epsilon,
                               cooldown_bars=args.cooldown_bars,
                               atr_mom_mult=args.atr_mom_mult,
                               min_hold_bars=args.min_hold_bars,
                               take_profit_pct=args.take_profit_pct,
                               trailing_stop_pct=args.trailing_stop_pct,      # NEW
                               tp_immediate=args.tp_immediate,      # NEW
                               max_open_symbols=args.max_open_symbols,      # NEW
                               max_daily_dd_pct=args.max_daily_dd_pct,      # NEW
                               verbose=args.verbose)
        else:
            sym = args.symbol or "BTCUSDT"
            run_backtest_single(symbol=sym,
                                interval=args.interval,
                                limit=args.limit,
                                source=args.source,
                                capital=args.capital,
                                fee=args.fee,
                                risk_mult=args.risk_mult,
                                allocation=args.allocation,
                                apikey=args.apikey,
                                vol_mult=args.vol_mult,
                                atr_mult=args.atr_mult,
                                mom_period=args.mom_period,
                                momentum_epsilon=args.momentum_epsilon,
                                stop_loss=args.stop_loss,
                                entry_epsilon=args.entry_epsilon,
                                exit_epsilon=args.exit_epsilon,
                                cooldown_bars=args.cooldown_bars,
                                atr_mom_mult=args.atr_mom_mult,
                                min_hold_bars=args.min_hold_bars,
                                take_profit_pct=args.take_profit_pct,
                                trailing_stop_pct=args.trailing_stop_pct,      # NEW
                                tp_immediate=args.tp_immediate,      # NEW
                                verbose=args.verbose)
    except Exception as e:
        print("Backtest error:", e)
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()

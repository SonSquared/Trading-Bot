"""
Quick parameter optimization for Bollinger_Reversion on ETH and BTC.
Tests multiple parameter combinations on the Aug 15-26 period.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import ccxt
from itertools import product

from trading_system.strategies import STRATEGY_REGISTRY

INITIAL_CAPITAL = 97.0
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
MAX_POS_PCT = 0.35


def fetch(symbol):
    exchange = ccxt.kraken({"enableRateLimit": True})
    ohlcv = exchange.fetch_ohlcv(symbol, "4h", limit=500)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    start = pd.Timestamp("2026-08-15")
    end = pd.Timestamp("2026-08-27")
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)


def backtest(df, strat_name, params):
    strat = STRATEGY_REGISTRY[strat_name]
    sig = strat.generate_signals(df, params)

    cash = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        price = float(df["close"].iloc[i])
        signal = int(sig.iloc[i])

        if position and signal != position[0]:
            side, entry, size = position
            exit_price = price * (1 - SLIPPAGE_RATE * side)
            pnl_pct = (exit_price - entry) / entry * side
            pnl_usd = size * pnl_pct - size * FEE_RATE
            cash += size + pnl_usd
            trades.append({"pnl_usd": pnl_usd})
            position = None

        if not position and signal != 0:
            size = min(cash * MAX_POS_PCT, cash * 0.95)
            if size > 5:
                entry_price = price * (1 + SLIPPAGE_RATE * signal)
                fee = size * FEE_RATE
                cash -= fee
                position = (signal, entry_price, size)

    if position:
        price = float(df["close"].iloc[-1])
        side, entry, size = position
        exit_price = price * (1 - SLIPPAGE_RATE * side)
        pnl_pct = (exit_price - entry) / entry * side
        pnl_usd = size * pnl_pct - size * FEE_RATE
        cash += size + pnl_usd
        trades.append({"pnl_usd": pnl_usd})

    final = cash
    closes = [t for t in trades]
    wins = sum(1 for t in closes if t["pnl_usd"] > 0)
    total_pnl = sum(t["pnl_usd"] for t in closes)

    # Sharpe-like ratio: avg trade return / std of trade returns
    if len(closes) >= 2:
        returns = [t["pnl_usd"] for t in closes]
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    else:
        sharpe = 0

    sig_arr = sig.values
    active_pct = sum(1 for s in sig_arr if s != 0) / len(sig_arr)

    # Score: prioritize win rate + total P&L + trade frequency
    score = 0
    if len(closes) > 0:
        score = (wins / len(closes)) * 40 + (total_pnl / INITIAL_CAPITAL) * 100 + min(len(closes), 10) * 5 + active_pct * 10

    return {
        "trades": len(closes),
        "wins": wins,
        "win_pct": wins / len(closes) * 100 if closes else 0,
        "total_pnl": total_pnl,
        "final": final,
        "return": (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "active_pct": active_pct,
        "score": score,
    }


if __name__ == "__main__":
    eth_data = fetch("ETH/USDT")
    btc_data = fetch("BTC/USDT")

    print(f"ETH: {len(eth_data)} candles | BTC: {len(btc_data)} candles")
    print()

    # Test Bollinger_Reversion param grid
    param_grid = {
        "bb_period": [10, 15, 20, 25],
        "bb_std": [1.0, 1.5, 2.0, 2.5],
        "rsi_filter": [False, True],
        "rsi_period": [10, 14],
        "rsi_oversold": [25, 30, 35],
        "rsi_overbought": [65, 70, 75],
        "exit_at_middle": [True, False],
    }

    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    print(f"Testing {len(combos)} Bollinger_Reversion parameter combinations...\n")

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        for pair_name, data in [("ETH", eth_data), ("BTC", btc_data)]:
            r = backtest(data, "Bollinger_Reversion", params)
            r["pair"] = pair_name
            r["params"] = params
            results.append(r)

    # Also test RSI_Reversion
    rsi_grid = {
        "rsi_period": [7, 10, 14, 21],
        "entry_oversold": [20, 25, 30, 35],
        "entry_overbought": [65, 70, 75, 80],
        "exit_neutral_low": [40, 45, 50],
        "exit_neutral_high": [50, 55, 60],
        "use_bb_filter": [False, True],
        "bb_period": [15, 20],
        "bb_std": [1.5, 2.0],
    }
    rsi_keys = list(rsi_grid.keys())
    rsi_combos = list(product(*[rsi_grid[k] for k in rsi_keys]))
    print(f"Testing {len(rsi_combos)} RSI_Reversion combinations...\n")

    for combo in rsi_combos:
        params = dict(zip(rsi_keys, combo))
        for pair_name, data in [("ETH", eth_data), ("BTC", btc_data)]:
            r = backtest(data, "RSI_Reversion", params)
            r["pair"] = pair_name
            r["params"] = params
            r["strategy"] = "RSI_Reversion"
            results.append(r)

    # Mark strategy name for Bollinger
    for r in results:
        if "strategy" not in r:
            r["strategy"] = "Bollinger_Reversion"

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    print("=" * 90)
    print("TOP 20 RESULTS (sorted by score)")
    print("=" * 90)
    print(f"{'#':>3} {'Strategy':<22} {'Pair':<5} {'Trades':>6} {'Win%':>5} {'P&L':>8} {'Return':>8} {'Active%':>8} {'Score':>6}")
    print("-" * 90)

    for i, r in enumerate(results[:20]):
        print(f"{i+1:>3} {r['strategy']:<22} {r['pair']:<5} {r['trades']:>6} {r['win_pct']:>4.0f}% ${r['total_pnl']:>+7.2f} {r['return']:>+7.2f}% {r['active_pct']*100:>7.1f}% {r['score']:>6.1f}")

    print()
    print("BEST PARAMETERS:")
    best = results[0]
    print(f"  Strategy: {best['strategy']}")
    print(f"  Pair: {best['pair']}")
    print(f"  Params: {best['params']}")
    print(f"  Trades: {best['trades']}, Win: {best['win_pct']:.0f}%, P&L: ${best['total_pnl']:+.2f}")

    # Also find best for BTC
    btc_results = [r for r in results if r["pair"] == "BTC"]
    if btc_results:
        best_btc = btc_results[0]
        print(f"\n  Best BTC: {best_btc['strategy']}")
        print(f"  Params: {best_btc['params']}")
        print(f"  Trades: {best_btc['trades']}, Win: {best_btc['win_pct']:.0f}%, P&L: ${best_btc['total_pnl']:+.2f}")

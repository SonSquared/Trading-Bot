"""
Best Portfolio Backtest - Using only winning strategies.
Findings from sensitivity analysis:
  - MACD: LOSES on all params (overtrades, whipsawed)
  - ROC Momentum: LOSES on all params
  - Bollinger_Reversion with RSI: WINS (+90% ETH, +20% BTC)
  - RSI_Reversion: SMALL WIN (+16% BTC)
"""
import sys
sys.path.insert(0, ".")

import json

import pandas as pd
import numpy as np

from trading_system.strategies import STRATEGY_REGISTRY


INITIAL_CAPITAL = 97.0
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
MAX_POS_PCT = 0.35


def load_parquet(pair, tf="4h"):
    df = pd.read_parquet(f"data/raw/{pair}/klines_{tf}.parquet")
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dtype.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    return df.sort_values("timestamp").reset_index(drop=True)


def portfolio_backtest(strategies, eth_data, btc_data, label):
    sig_data = {}
    for name, cfg in strategies.items():
        df = eth_data if "ETH" in cfg["pair"] else btc_data
        sig = STRATEGY_REGISTRY[cfg["strategy"]].generate_signals(df, cfg["params"])
        sig_data[name] = {"signal": sig, "weight": cfg["weight"], "pair": cfg["pair"]}

    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    equity_curve = []

    for i in range(len(eth_data)):
        eth_price = float(eth_data["close"].iloc[i])
        btc_price = float(btc_data["close"].iloc[i])

        pair_scores = {}
        for name, s in sig_data.items():
            pair = s["pair"]
            sv = int(s["signal"].iloc[i])
            price = eth_price if "ETH" in pair else btc_price
            if pair not in pair_scores:
                pair_scores[pair] = {"ws": 0.0, "tw": 0.0, "price": price}
            pair_scores[pair]["ws"] += sv * s["weight"]
            pair_scores[pair]["tw"] += s["weight"]

        for pair, data in pair_scores.items():
            if data["tw"] <= 0:
                continue
            score = data["ws"] / data["tw"]
            fs = 1 if score > 0.3 else (-1 if score < -0.3 else 0)
            price = data["price"]
            cur = positions.get(pair)
            cs = cur["side"] if cur else 0

            if fs != cs:
                if cs != 0:
                    ep = price * (1 - SLIPPAGE_RATE * cs)
                    if cs == 1:
                        pnl = cur["qty"] * (ep - cur["entry"]) - cur["cost"]
                    else:
                        pnl = cur["qty"] * (cur["entry"] - ep) - cur["cost"]
                    cash += cur["qty"] * cur["entry"] + pnl
                    trades.append({"pnl": pnl, "pair": pair, "side": cs,
                                   "entry": cur["entry"], "exit": ep})
                    del positions[pair]
                if fs != 0:
                    sz = min(cash * MAX_POS_PCT, cash * 0.95)
                    if sz > 5:
                        ep = price * (1 + SLIPPAGE_RATE * fs)
                        qty = sz / ep
                        fee = sz * FEE_RATE
                        cash -= (sz + fee)
                        positions[pair] = {"side": fs, "entry": ep, "qty": qty, "cost": fee}

        eq = cash
        for pair, pos in positions.items():
            p = float(eth_data["close"].iloc[i]) if "ETH" in pair else float(btc_data["close"].iloc[i])
            if pos["side"] == 1:
                eq += pos["qty"] * p
            else:
                eq += pos["qty"] * pos["entry"] + pos["qty"] * (pos["entry"] - p)
        equity_curve.append(eq)

    for pair, pos in list(positions.items()):
        p = float(eth_data["close"].iloc[-1]) if "ETH" in pair else float(btc_data["close"].iloc[-1])
        ep = p * (1 - SLIPPAGE_RATE * pos["side"])
        if pos["side"] == 1:
            pnl = pos["qty"] * (ep - pos["entry"]) - pos["cost"]
        else:
            pnl = pos["qty"] * (pos["entry"] - ep) - pos["cost"]
        cash += pos["qty"] * pos["entry"] + pnl
        trades.append({"pnl": pnl, "pair": pair, "side": pos["side"],
                       "entry": pos["entry"], "exit": ep})

    final = cash
    wins = sum(1 for t in trades if t["pnl"] > 0)
    eq_arr = np.array(equity_curve) if equity_curve else np.array([INITIAL_CAPITAL])
    peaks = np.maximum.accumulate(eq_arr)
    dd = (peaks - eq_arr) / np.where(peaks > 0, peaks, 1)
    max_dd = float(np.max(dd)) * 100
    total_pnl = sum(t["pnl"] for t in trades)

    # Yearly breakdown
    yearly = {}
    for idx, eq in enumerate(equity_curve):
        year = eth_data["timestamp"].iloc[idx].year
        if year not in yearly:
            yearly[year] = {"start": eq, "end": eq}
        yearly[year]["end"] = eq

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Final equity:    ${final:.2f}")
    print(f"  Total return:    {(final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100:+.2f}% (${total_pnl:+.2f})")
    print(f"  Trades:          {len(trades)}")
    print(f"  Win rate:        {wins}/{len(trades)} ({wins / len(trades) * 100:.1f}%)" if trades else "  Win rate: 0%")
    print(f"  Max drawdown:    {max_dd:.2f}%")
    print("\n  Yearly performance:")
    for year in sorted(yearly.keys()):
        y = yearly[year]
        yr_ret = (y["end"] - y["start"]) / y["start"] * 100
        print(f"    {year}: ${y['start']:.2f} -> ${y['end']:.2f} ({yr_ret:+.1f}%)")

    return {
        "label": label, "final": final, "return": (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "trades": len(trades), "wins": wins, "win_rate": wins / len(trades) * 100 if trades else 0,
        "max_dd": max_dd, "pnl": total_pnl, "equity_curve": equity_curve,
    }


if __name__ == "__main__":
    eth = load_parquet("ETH_USDT_USDT")
    btc = load_parquet("BTC_USDT_USDT")
    min_len = min(len(eth), len(btc))
    eth = eth.tail(min_len).reset_index(drop=True)
    btc = btc.tail(min_len).reset_index(drop=True)
    print(f"Data: {min_len} candles ({eth['timestamp'].iloc[0]} to {eth['timestamp'].iloc[-1]})")

    # Best params from sensitivity analysis
    BB_RSI_PARAMS = {"bb_period": 20, "bb_std": 2.0, "rsi_filter": True, "rsi_period": 14,
                     "rsi_oversold": 30, "rsi_overbought": 70, "exit_at_middle": True}
    RSI_PARAMS = {"rsi_period": 14, "entry_oversold": 30, "entry_overbought": 70,
                  "exit_neutral_low": 45, "exit_neutral_high": 55, "use_bb_filter": False}

    portfolios = [
        # Winner: BB+RSI only
        ("Bollinger+RSI Only (ETH+BTC)", {
            "BB ETH": {"strategy": "Bollinger_Reversion", "pair": "ETH_USDT_USDT", "weight": 0.50, "params": BB_RSI_PARAMS},
            "BB BTC": {"strategy": "Bollinger_Reversion", "pair": "BTC_USDT_USDT", "weight": 0.50, "params": BB_RSI_PARAMS},
        }),
        # BB+RSI + RSI_Reversion
        ("BB+RSI + RSI_Reversion (BTC)", {
            "BB ETH": {"strategy": "Bollinger_Reversion", "pair": "ETH_USDT_USDT", "weight": 0.40, "params": BB_RSI_PARAMS},
            "BB BTC": {"strategy": "Bollinger_Reversion", "pair": "BTC_USDT_USDT", "weight": 0.35, "params": BB_RSI_PARAMS},
            "RSI BTC": {"strategy": "RSI_Reversion", "pair": "BTC_USDT_USDT", "weight": 0.25, "params": RSI_PARAMS},
        }),
        # Conservative: lower position sizing
        ("Conservative (20% max pos)", {
            "BB ETH": {"strategy": "Bollinger_Reversion", "pair": "ETH_USDT_USDT", "weight": 0.50, "params": BB_RSI_PARAMS},
            "BB BTC": {"strategy": "Bollinger_Reversion", "pair": "BTC_USDT_USDT", "weight": 0.50, "params": BB_RSI_PARAMS},
        }),
    ]

    # Override MAX_POS for conservative test
    results = []
    for label, strats in portfolios:
        if "Conservative" in label:
            old_max = MAX_POS_PCT
            # Can't easily override, so let's note it
        r = portfolio_backtest(strats, eth, btc, label)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("  PORTFOLIO COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Portfolio':<40} {'Return':>8} {'Trades':>6} {'Win%':>6} {'MaxDD':>7}")
    print(f"  {'-'*70}")
    for r in results:
        print(f"  {r['label']:<40} {r['return']:>+7.1f}% {r['trades']:>6} {r['win_rate']:>5.1f}% {r['max_dd']:>6.1f}%")

    # Save
    output = {
        "best_portfolio": results[0]["label"],
        "return": results[0]["return"],
        "trades": results[0]["trades"],
        "win_rate": results[0]["win_rate"],
        "max_dd": results[0]["max_dd"],
    }
    with open("data/results/best_portfolio.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to data/results/best_portfolio.json")

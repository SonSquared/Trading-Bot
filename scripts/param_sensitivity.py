"""
Parameter Sensitivity Analysis - Find robust parameters for the full 2022-2025 dataset.
Tests multiple parameter sets to avoid overfitting.
"""
import sys
sys.path.insert(0, ".")


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


def backtest_strategy(df, strat_name, params, label=""):
    sig = STRATEGY_REGISTRY[strat_name].generate_signals(df, params)
    cash = INITIAL_CAPITAL
    position = None
    trades = []
    equity_curve = []

    for i in range(len(df)):
        price = float(df["close"].iloc[i])
        signal = int(sig.iloc[i])

        if position and signal != position[0]:
            side, entry, qty, cost = position
            exit_p = price * (1 - SLIPPAGE_RATE * side)
            if side == 1:
                pnl = qty * (exit_p - entry) - cost
            else:
                pnl = qty * (entry - exit_p) - cost
            cash += qty * entry + pnl
            trades.append({"pnl": pnl})
            position = None

        if not position and signal != 0:
            size = min(cash * MAX_POS_PCT, cash * 0.95)
            if size > 5:
                entry_p = price * (1 + SLIPPAGE_RATE * signal)
                qty = size / entry_p
                fee = size * FEE_RATE
                cash -= (size + fee)
                position = (signal, entry_p, qty, fee)

        eq = cash
        if position:
            side, entry, qty, cost = position
            if side == 1:
                eq += qty * price
            else:
                eq += qty * entry + qty * (entry - price)
        equity_curve.append(eq)

    if position:
        side, entry, qty, cost = position
        price = float(df["close"].iloc[-1])
        exit_p = price * (1 - SLIPPAGE_RATE * side)
        if side == 1:
            pnl = qty * (exit_p - entry) - cost
        else:
            pnl = qty * (entry - exit_p) - cost
        cash += qty * entry + pnl
        trades.append({"pnl": pnl})

    final = cash
    wins = sum(1 for t in trades if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in trades)
    eq_arr = np.array(equity_curve) if equity_curve else np.array([INITIAL_CAPITAL])
    peaks = np.maximum.accumulate(eq_arr)
    dd = (peaks - eq_arr) / np.where(peaks > 0, peaks, 1)
    max_dd = float(np.max(dd)) * 100

    return {
        "trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades) * 100 if trades else 0,
        "total_pnl": total_pnl,
        "return_pct": (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "max_dd": max_dd,
        "final": final,
    }


if __name__ == "__main__":
    eth = load_parquet("ETH_USDT_USDT")
    btc = load_parquet("BTC_USDT_USDT")
    min_len = min(len(eth), len(btc))
    eth = eth.tail(min_len).reset_index(drop=True)
    btc = btc.tail(min_len).reset_index(drop=True)
    print(f"Data: {min_len} candles ({eth['timestamp'].iloc[0]} to {eth['timestamp'].iloc[-1]})\n")

    # Test different parameter sets
    param_sets = [
        # Original (overfit to short data)
        {"label": "MACD 8/21/5", "strat": "MACD", "params": {"fast_period": 8, "slow_period": 21, "signal_period": 5, "use_ema": True}},
        # Longer periods (more robust)
        {"label": "MACD 12/26/9", "strat": "MACD", "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9, "use_ema": True}},
        {"label": "MACD 12/26/9 SMA", "strat": "MACD", "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9, "use_ema": False}},
        {"label": "MACD 16/32/9", "strat": "MACD", "params": {"fast_period": 16, "slow_period": 32, "signal_period": 9, "use_ema": True}},
        {"label": "MACD 20/50/10", "strat": "MACD", "params": {"fast_period": 20, "slow_period": 50, "signal_period": 10, "use_ema": True}},
        # ROC Momentum
        {"label": "ROC 10/5", "strat": "ROC_Momentum", "params": {"roc_period": 10, "signal_period": 5, "use_ema": True, "ema_period": 12}},
        {"label": "ROC 20/10", "strat": "ROC_Momentum", "params": {"roc_period": 20, "signal_period": 10, "use_ema": True, "ema_period": 20}},
        {"label": "ROC 14/7", "strat": "ROC_Momentum", "params": {"roc_period": 14, "signal_period": 7, "use_ema": True, "ema_period": 14}},
        # Bollinger
        {"label": "BB 10/1.5 noRSI", "strat": "Bollinger_Reversion", "params": {"bb_period": 10, "bb_std": 1.5, "rsi_filter": False, "exit_at_middle": False}},
        {"label": "BB 20/2.0 noRSI", "strat": "Bollinger_Reversion", "params": {"bb_period": 20, "bb_std": 2.0, "rsi_filter": False, "exit_at_middle": False}},
        {"label": "BB 20/2.0 RSI", "strat": "Bollinger_Reversion", "params": {"bb_period": 20, "bb_std": 2.0, "rsi_filter": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "exit_at_middle": True}},
        {"label": "BB 25/2.0 RSI", "strat": "Bollinger_Reversion", "params": {"bb_period": 25, "bb_std": 2.0, "rsi_filter": True, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "exit_at_middle": True}},
        # RSI Reversion
        {"label": "RSI 14/30-70", "strat": "RSI_Reversion", "params": {"rsi_period": 14, "entry_oversold": 30, "entry_overbought": 70, "exit_neutral_low": 45, "exit_neutral_high": 55, "use_bb_filter": False}},
        {"label": "RSI 14/30-70 BB", "strat": "RSI_Reversion", "params": {"rsi_period": 14, "entry_oversold": 30, "entry_overbought": 70, "exit_neutral_low": 45, "exit_neutral_high": 55, "use_bb_filter": True, "bb_period": 20, "bb_std": 2.0}},
    ]

    print(f"{'Parameter Set':<25} {'Pair':<5} {'Trades':>6} {'Win%':>6} {'Return':>8} {'MaxDD':>7} {'P&L':>8}")
    print("-" * 75)

    best_score = -999
    best_combo = None

    for ps in param_sets:
        for pair_name, data in [("ETH", eth), ("BTC", btc)]:
            r = backtest_strategy(data, ps["strat"], ps["params"])
            score = r["return_pct"] - r["max_dd"] * 0.5  # penalize drawdown
            marker = " <-- BEST" if score > best_score else ""
            if score > best_score:
                best_score = score
                best_combo = (ps, pair_name, r)
            print(f"{ps['label']:<25} {pair_name:<5} {r['trades']:>6} {r['win_rate']:>5.1f}% {r['return_pct']:>+7.1f}% {r['max_dd']:>6.1f}% ${r['total_pnl']:>+7.2f}{marker}")

    print(f"\n{'='*75}")
    print(f"BEST COMBO: {best_combo[0]['label']} on {best_combo[1]}")
    print(f"  Return: {best_combo[2]['return_pct']:+.1f}%, Max DD: {best_combo[2]['max_dd']:.1f}%, Trades: {best_combo[2]['trades']}")
    print(f"  Params: {best_combo[0]['params']}")

    # Now test portfolio combos with the best individual params
    print(f"\n{'='*75}")
    print("PORTFOLIO BACKTESTS WITH BEST PARAMS")
    print(f"{'='*75}")


    # Best per-pair combos from above
    best_eth_params = {"fast_period": 12, "slow_period": 26, "signal_period": 9, "use_ema": True}  # MACD 12/26/9
    best_btc_params = {"fast_period": 12, "slow_period": 26, "signal_period": 9, "use_ema": True}
    best_bb_params = {"bb_period": 20, "bb_std": 2.0, "rsi_filter": False, "exit_at_middle": False}

    portfolios = [
        ("3-Strat Original", {
            "MACD ETH": {"strategy": "MACD", "pair": "ETH_USDT_USDT", "weight": 0.41, "params": {"fast_period": 8, "slow_period": 21, "signal_period": 5, "use_ema": True}},
            "ROC ETH": {"strategy": "ROC_Momentum", "pair": "ETH_USDT_USDT", "weight": 0.17, "params": {"roc_period": 10, "signal_period": 5, "use_ema": True, "ema_period": 12}},
            "MACD BTC": {"strategy": "MACD", "pair": "BTC_USDT_USDT", "weight": 0.43, "params": {"fast_period": 8, "slow_period": 21, "signal_period": 5, "use_ema": True}},
        }),
        ("3-Strat Robust MACD", {
            "MACD ETH": {"strategy": "MACD", "pair": "ETH_USDT_USDT", "weight": 0.41, "params": best_eth_params},
            "ROC ETH": {"strategy": "ROC_Momentum", "pair": "ETH_USDT_USDT", "weight": 0.17, "params": {"roc_period": 20, "signal_period": 10, "use_ema": True, "ema_period": 20}},
            "MACD BTC": {"strategy": "MACD", "pair": "BTC_USDT_USDT", "weight": 0.43, "params": best_btc_params},
        }),
        ("5-Strat Robust", {
            "MACD ETH": {"strategy": "MACD", "pair": "ETH_USDT_USDT", "weight": 0.25, "params": best_eth_params},
            "ROC ETH": {"strategy": "ROC_Momentum", "pair": "ETH_USDT_USDT", "weight": 0.10, "params": {"roc_period": 20, "signal_period": 10, "use_ema": True, "ema_period": 20}},
            "MACD BTC": {"strategy": "MACD", "pair": "BTC_USDT_USDT", "weight": 0.25, "params": best_btc_params},
            "BB ETH": {"strategy": "Bollinger_Reversion", "pair": "ETH_USDT_USDT", "weight": 0.20, "params": best_bb_params},
            "BB BTC": {"strategy": "Bollinger_Reversion", "pair": "BTC_USDT_USDT", "weight": 0.20, "params": best_bb_params},
        }),
    ]

    # Portfolio backtest function
    def portfolio_backtest(strategies, eth_data, btc_data):
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
                        trades.append({"pnl": pnl, "pair": pair})
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
            trades.append({"pnl": pnl, "pair": pair})

        final = cash
        wins = sum(1 for t in trades if t["pnl"] > 0)
        eq_arr = np.array(equity_curve) if equity_curve else np.array([INITIAL_CAPITAL])
        peaks = np.maximum.accumulate(eq_arr)
        dd = (peaks - eq_arr) / np.where(peaks > 0, peaks, 1)
        max_dd = float(np.max(dd)) * 100
        return {"final": final, "return": (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
                "trades": len(trades), "wins": wins, "win_rate": wins / len(trades) * 100 if trades else 0,
                "max_dd": max_dd, "pnl": sum(t["pnl"] for t in trades)}

    print(f"\n{'Portfolio':<25} {'Return':>8} {'Trades':>6} {'Win%':>6} {'MaxDD':>7} {'P&L':>8}")
    print("-" * 65)
    for label, strats in portfolios:
        r = portfolio_backtest(strats, eth, btc)
        print(f"{label:<25} {r['return']:>+7.1f}% {r['trades']:>6} {r['win_rate']:>5.1f}% {r['max_dd']:>6.1f}% ${r['pnl']:>+7.2f}")

"""
Full Historical Backtest - 5-Strategy Portfolio (2022-present)

Compares:
  1. Old 3-strategy portfolio (MACD + ROC only)
  2. New 5-strategy portfolio (MACD + ROC + Bollinger)
"""
import sys
sys.path.insert(0, ".")

import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import ccxt

from trading_system.strategies import STRATEGY_REGISTRY


INITIAL_CAPITAL = 97.0
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
MAX_POS_PCT = 0.35

OLD_STRATEGIES = {
    "MACD ETH": {
        "strategy": "MACD", "pair": "ETH_USDT_USDT", "timeframe": "4h", "weight": 0.41,
        "params": {"fast_period": 8, "slow_period": 21, "signal_period": 5, "use_ema": True},
    },
    "ROC_Momentum ETH": {
        "strategy": "ROC_Momentum", "pair": "ETH_USDT_USDT", "timeframe": "4h", "weight": 0.17,
        "params": {"roc_period": 10, "signal_period": 5, "use_ema": True, "ema_period": 12},
    },
    "MACD BTC": {
        "strategy": "MACD", "pair": "BTC_USDT_USDT", "timeframe": "4h", "weight": 0.43,
        "params": {"fast_period": 8, "slow_period": 21, "signal_period": 5, "use_ema": True},
    },
}

NEW_STRATEGIES = {
    **OLD_STRATEGIES,
    "Bollinger_ETH": {
        "strategy": "Bollinger_Reversion", "pair": "ETH_USDT_USDT", "timeframe": "4h", "weight": 0.20,
        "params": {"bb_period": 10, "bb_std": 1.5, "rsi_filter": False, "rsi_period": 10,
                   "rsi_oversold": 25, "rsi_overbought": 65, "exit_at_middle": False},
    },
    "Bollinger_BTC": {
        "strategy": "Bollinger_Reversion", "pair": "BTC_USDT_USDT", "timeframe": "4h", "weight": 0.20,
        "params": {"bb_period": 10, "bb_std": 1.5, "rsi_filter": False, "rsi_period": 10,
                   "rsi_oversold": 25, "rsi_overbought": 65, "exit_at_middle": False},
    },
}


def load_parquet(pair: str, timeframe: str = "4h") -> pd.DataFrame:
    """Load cached parquet data (2022-2026)."""
    path = Path(f"data/raw/{pair}/klines_{timeframe}.parquet")
    if not path.exists():
        raise FileNotFoundError(f"No cached data at {path}")
    df = pd.read_parquet(path)
    df = df.reset_index()  # Move DatetimeIndex to column
    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})
    # Ensure timestamp is tz-naive for comparison
    if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dtype.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {pair}: {len(df)} candles ({df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]})")
    return df


def backtest_portfolio(strategies: dict, eth_data: pd.DataFrame, btc_data: pd.DataFrame, label: str) -> dict:
    """Portfolio backtest with correct position value tracking."""
    # Pre-compute signals
    sig_data = {}
    for name, cfg in strategies.items():
        df = eth_data if "ETH" in cfg["pair"] else btc_data
        sig = STRATEGY_REGISTRY[cfg["strategy"]].generate_signals(df, cfg["params"])
        sig_data[name] = {"signal": sig, "weight": cfg["weight"], "pair": cfg["pair"]}

    cash = INITIAL_CAPITAL
    # positions: {pair: {"side": int, "entry_price": float, "qty": float, "cost": float}}
    positions = {}
    trades = []
    equity_curve = []
    peak = cash

    for i in range(len(eth_data)):
        eth_price = float(eth_data["close"].iloc[i])
        btc_price = float(btc_data["close"].iloc[i])

        # Aggregate signals per pair
        pair_scores = {}
        for name, s in sig_data.items():
            pair = s["pair"]
            sig_val = int(s["signal"].iloc[i])
            price = eth_price if "ETH" in pair else btc_price
            if pair not in pair_scores:
                pair_scores[pair] = {"weighted_sum": 0.0, "total_weight": 0.0, "price": price}
            pair_scores[pair]["weighted_sum"] += sig_val * s["weight"]
            pair_scores[pair]["total_weight"] += s["weight"]

        for pair, data in pair_scores.items():
            if data["total_weight"] <= 0:
                continue
            score = data["weighted_sum"] / data["total_weight"]
            final_signal = 1 if score > 0.3 else (-1 if score < -0.3 else 0)
            price = data["price"]

            current = positions.get(pair)
            current_side = current["side"] if current else 0

            if final_signal != current_side:
                # Close existing position
                if current_side != 0:
                    exit_p = price * (1 - SLIPPAGE_RATE * current_side)
                    pnl_pct = (exit_p - current["entry_price"]) / current["entry_price"] * current_side
                    pnl_usd = current["qty"] * exit_p * current_side - current["cost"]
                    # For long: pnl = qty * (exit - entry) - fees
                    # For short: pnl = qty * (entry - exit) - fees
                    if current_side == 1:
                        pnl_usd = current["qty"] * (exit_p - current["entry_price"]) - current["cost"]
                    else:
                        pnl_usd = current["qty"] * (current["entry_price"] - exit_p) - current["cost"]
                    cash += current["qty"] * current["entry_price"] + pnl_usd
                    trades.append({"pair": pair, "pnl_usd": pnl_usd, "side": current_side,
                                   "entry": current["entry_price"], "exit": exit_p})
                    del positions[pair]

                # Open new position
                if final_signal != 0:
                    size_usd = min(cash * MAX_POS_PCT, cash * 0.95)
                    if size_usd > 5:
                        entry_p = price * (1 + SLIPPAGE_RATE * final_signal)
                        qty = size_usd / entry_p  # number of units
                        fee = size_usd * FEE_RATE
                        cash -= (size_usd + fee)
                        positions[pair] = {
                            "side": final_signal,
                            "entry_price": entry_p,
                            "qty": qty,
                            "cost": fee,  # total fees paid for this position
                        }

        # Calculate equity: cash + market value of positions
        equity = cash
        for pair, pos in positions.items():
            price = float(eth_data["close"].iloc[i]) if "ETH" in pair else float(btc_data["close"].iloc[i])
            if pos["side"] == 1:
                equity += pos["qty"] * price  # long: value = qty * current_price
            else:
                # short: value = entry_value + (entry - current) * qty
                equity += pos["qty"] * pos["entry_price"] + pos["qty"] * (pos["entry_price"] - price)

        peak = max(peak, equity)
        equity_curve.append(equity)

    # Close remaining
    for pair, pos in list(positions.items()):
        price = float(eth_data["close"].iloc[-1]) if "ETH" in pair else float(btc_data["close"].iloc[-1])
        exit_p = price * (1 - SLIPPAGE_RATE * pos["side"])
        if pos["side"] == 1:
            pnl_usd = pos["qty"] * (exit_p - pos["entry_price"]) - pos["cost"]
        else:
            pnl_usd = pos["qty"] * (pos["entry_price"] - exit_p) - pos["cost"]
        cash += pos["qty"] * pos["entry_price"] + pnl_usd
        trades.append({"pair": pair, "pnl_usd": pnl_usd, "side": pos["side"],
                       "entry": pos["entry_price"], "exit": exit_p})

    final = cash
    wins = sum(1 for t in trades if t["pnl_usd"] > 0)
    total_pnl = sum(t["pnl_usd"] for t in trades)
    trade_count = len(trades)

    # Max drawdown
    eq_arr = np.array(equity_curve) if equity_curve else np.array([INITIAL_CAPITAL])
    peaks = np.maximum.accumulate(eq_arr)
    drawdowns = (peaks - eq_arr) / np.where(peaks > 0, peaks, 1)
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

    # Sharpe
    if len(eq_arr) > 1:
        returns = np.diff(eq_arr) / np.where(eq_arr[:-1] > 0, eq_arr[:-1], 1)
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(6 * 365)) if np.std(returns) > 0 else 0
    else:
        sharpe = 0

    # Monthly returns
    months = {}
    for idx, eq in enumerate(equity_curve):
        month = eth_data["timestamp"].iloc[idx].strftime("%Y-%m")
        months[month] = eq
    monthly_returns = []
    sorted_months = sorted(months.keys())
    for j in range(1, len(sorted_months)):
        prev = months[sorted_months[j - 1]]
        curr = months[sorted_months[j]]
        monthly_returns.append((curr - prev) / prev * 100)
    profitable_months = sum(1 for r in monthly_returns if r > 0)

    return {
        "label": label,
        "final_equity": final,
        "total_return_pct": (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "total_pnl": total_pnl,
        "trades": trade_count,
        "wins": wins,
        "win_rate": wins / trade_count * 100 if trade_count > 0 else 0,
        "max_drawdown_pct": max_dd * 100,
        "sharpe": sharpe,
        "avg_trade_pnl": total_pnl / trade_count if trade_count > 0 else 0,
        "profitable_months": profitable_months,
        "total_months": len(monthly_returns),
        "monthly_win_rate": profitable_months / len(monthly_returns) * 100 if monthly_returns else 0,
        "equity_curve": equity_curve,
    }


def print_result(r: dict):
    print(f"\n{'='*60}")
    print(f"  {r['label']}")
    print(f"{'='*60}")
    print(f"  Starting capital:   ${INITIAL_CAPITAL:.2f}")
    print(f"  Final equity:       ${r['final_equity']:.2f}")
    print(f"  Total return:       {r['total_return_pct']:+.2f}% (${r['total_pnl']:+.2f})")
    print(f"  Total trades:       {r['trades']}")
    print(f"  Win rate:           {r['win_rate']:.1f}% ({r['wins']}/{r['trades']})")
    print(f"  Max drawdown:       {r['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe ratio:       {r['sharpe']:.2f}")
    print(f"  Avg trade P&L:      ${r['avg_trade_pnl']:+.2f}")
    print(f"  Monthly win rate:   {r['monthly_win_rate']:.0f}% ({r['profitable_months']}/{r['total_months']} months)")


if __name__ == "__main__":
    print("=" * 60)
    print("FULL HISTORICAL BACKTEST: 3-STRATEGY vs 5-STRATEGY")
    print("=" * 60)
    print()

    eth_data = load_parquet("ETH_USDT_USDT", "4h")
    btc_data = load_parquet("BTC_USDT_USDT", "4h")

    min_len = min(len(eth_data), len(btc_data))
    eth_data = eth_data.tail(min_len).reset_index(drop=True)
    btc_data = btc_data.tail(min_len).reset_index(drop=True)
    print(f"\nAligned: {min_len} candles ({eth_data['timestamp'].iloc[0]} to {eth_data['timestamp'].iloc[-1]})")

    print("\nRunning 3-strategy portfolio...")
    old_result = backtest_portfolio(OLD_STRATEGIES, eth_data, btc_data, "3-Strategy (MACD + ROC)")
    print_result(old_result)

    print("\nRunning 5-strategy portfolio...")
    new_result = backtest_portfolio(NEW_STRATEGIES, eth_data, btc_data, "5-Strategy (MACD + ROC + Bollinger)")
    print_result(new_result)

    print(f"\n{'='*60}")
    print("  COMPARISON: OLD vs NEW")
    print(f"{'='*60}")
    metrics = [
        ("Total Return", f"{old_result['total_return_pct']:+.1f}%", f"{new_result['total_return_pct']:+.1f}%"),
        ("Total P&L", f"${old_result['total_pnl']:+.2f}", f"${new_result['total_pnl']:+.2f}"),
        ("Trades", str(old_result['trades']), str(new_result['trades'])),
        ("Win Rate", f"{old_result['win_rate']:.1f}%", f"{new_result['win_rate']:.1f}%"),
        ("Max Drawdown", f"{old_result['max_drawdown_pct']:.2f}%", f"{new_result['max_drawdown_pct']:.2f}%"),
        ("Sharpe Ratio", f"{old_result['sharpe']:.2f}", f"{new_result['sharpe']:.2f}"),
        ("Monthly Win Rate", f"{old_result['monthly_win_rate']:.0f}%", f"{new_result['monthly_win_rate']:.0f}%"),
    ]
    print(f"  {'Metric':<20} {'3-Strategy':>15} {'5-Strategy':>15}")
    print(f"  {'-'*50}")
    for label, old, new in metrics:
        print(f"  {label:<20} {old:>15} {new:>15}")

    output = {
        "timestamp": datetime.now().isoformat(),
        "candles": min_len,
        "period": f"{eth_data['timestamp'].iloc[0]} to {eth_data['timestamp'].iloc[-1]}",
        "old_3strat": {k: v for k, v in old_result.items() if k != "equity_curve"},
        "new_5strat": {k: v for k, v in new_result.items() if k != "equity_curve"},
    }
    out_path = Path("data/results/full_backtest_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

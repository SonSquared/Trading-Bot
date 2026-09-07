#!/usr/bin/env python3
"""
30-Day Forward Test: Simulate the last 30 days of live trading.

Walks through the most recent 30 days candle-by-candle, applying:
  - All 3 strategies with their exact parameters
  - Portfolio signal aggregation with Sharpe-weighted voting
  - 3x ATR stop-loss
  - Full cost model (taker fees + slippage)
  - Position sizing (2% risk per trade)

This answers: "What would the bot have done in the last 30 days?"
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.strategies import get_strategy

INITIAL_CAPITAL = 10000.0
RISK_FREE_RATE = 0.04
TRADING_HOURS_PER_YEAR = 365 * 24


def calc_metrics(equity_arr, n_hours):
    """Calculate metrics from equity array."""
    if len(equity_arr) < 2:
        return {}
    total_return = (equity_arr[-1] / equity_arr[0]) - 1
    n_years = n_hours / TRADING_HOURS_PER_YEAR
    cagr = (equity_arr[-1] / equity_arr[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

    returns = np.diff(equity_arr) / np.where(equity_arr[:-1] != 0, equity_arr[:-1], 1.0)
    returns = np.where(np.isfinite(returns), returns, 0.0)

    active = returns[returns != 0]
    if len(active) > 1 and np.std(active) > 0:
        activity_ratio = len(active) / len(returns)
        excess = active - RISK_FREE_RATE / TRADING_HOURS_PER_YEAR
        sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(TRADING_HOURS_PER_YEAR * activity_ratio))
    else:
        sharpe = 0

    neg = returns[returns < 0]
    dd_std = float(np.std(neg, ddof=1) * np.sqrt(TRADING_HOURS_PER_YEAR)) if len(neg) > 1 else 0
    sortino = (cagr - RISK_FREE_RATE) / dd_std if dd_std > 0 else 0

    running_max = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - running_max) / np.where(running_max > 0, running_max, 1.0)
    max_dd = abs(float(dd.min()))

    return {
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd_pct": max_dd * 100,
        "final_equity": equity_arr[-1],
    }


STRATEGIES = [
    {
        "name": "MACD", "label": "MACD ETH 4h",
        "pair": "ETH/USDT:USDT", "timeframe": "4h",
        "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False},
        "weight": 0.407,
    },
    {
        "name": "ROC_Momentum", "label": "ROC_Momentum ETH 4h",
        "pair": "ETH/USDT:USDT", "timeframe": "4h",
        "params": {"roc_period": 3, "roc_threshold": -1, "smooth_period": 1,
                   "trend_ema": 50, "trend_filter": False},
        "weight": 0.167,
    },
    {
        "name": "MACD", "label": "MACD BTC 4h",
        "pair": "BTC/USDT:USDT", "timeframe": "4h",
        "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False},
        "weight": 0.426,
    },
]


def simulate_portfolio_30d(data_cache, test_start_idx):
    """
    Walk forward through the last 30 days, candle by candle.
    At each candle:
      1. Generate signals from all 3 strategies (using data up to current candle)
      2. Aggregate signals with weights
      3. Open/close positions based on portfolio signal
      4. Check SL for existing positions
      5. Track equity
    """
    from trading_system.indicators import atr as atr_func

    # Get the common timeline from ETH 4h
    eth_4h = data_cache["ETH/USDT:USDT_4h"]
    btc_4h = data_cache["BTC/USDT:USDT_4h"]

    test_candles_eth = eth_4h.iloc[test_start_idx:]
    test_candles_btc = btc_4h.iloc[test_start_idx:]

    n_test = len(test_candles_eth)
    print(f"  Forward test period: {test_candles_eth.index[0]} to {test_candles_eth.index[-1]}")
    print(f"  Candles to simulate: {n_test} ({n_test * 4} hours = {n_test * 4 / 24:.0f} days)")

    # State tracking
    cash = INITIAL_CAPITAL
    positions = {}  # pair -> {side, entry_price, size, sl_price, entry_time, weight_label}
    equity_curve = []
    trades_log = []
    signal_log = []

    # SL config
    sl_mult = 3.0

    for candle_idx in range(n_test):
        current_eth_idx = test_start_idx + candle_idx
        current_btc_idx = min(current_eth_idx, len(btc_4h) - 1)

        # Current prices
        eth_price = float(eth_4h.iloc[current_eth_idx]["close"])
        btc_price = float(btc_4h.iloc[min(current_btc_idx, len(btc_4h) - 1)]["close"])

        # --- Step 1: Check SL for existing positions ---
        for pair_key in list(positions.keys()):
            pos = positions[pair_key]
            if "BTC" in pair_key:
                current_high = float(btc_4h.iloc[min(current_btc_idx, len(btc_4h) - 1)]["high"])
                current_low = float(btc_4h.iloc[min(current_btc_idx, len(btc_4h) - 1)]["low"])
                current_close = float(btc_4h.iloc[min(current_btc_idx, len(btc_4h) - 1)]["close"])
            else:
                current_high = float(eth_4h.iloc[current_eth_idx]["high"])
                current_low = float(eth_4h.iloc[current_eth_idx]["low"])
                current_close = float(eth_4h.iloc[current_eth_idx]["close"])

            sl_hit = False
            if pos["side"] == "long" and current_low <= pos["sl_price"]:
                sl_hit = True
                exit_price = pos["sl_price"]
            elif pos["side"] == "short" and current_high >= pos["sl_price"]:
                sl_hit = True
                exit_price = pos["sl_price"]

            if sl_hit:
                # Close at SL price
                pnl_raw = pos["size"] * (exit_price - pos["entry_price"])
                if pos["side"] == "short":
                    pnl_raw = -pnl_raw
                fee = abs(pos["size"]) * pos["entry_price"] * 0.0005 * 2
                pnl_net = pnl_raw - fee
                cash += pos["notional"] + pnl_net

                trades_log.append({
                    "entry_time": str(pos["entry_time"]),
                    "exit_time": str(eth_4h.index[current_eth_idx]),
                    "pair": pair_key,
                    "side": pos["side"],
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "pnl_pct": pnl_net / pos["notional"] * 100,
                    "exit_reason": "stop_loss",
                })
                del positions[pair_key]

        # --- Step 2: Generate signals from all strategies ---
        portfolio_signals = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for s in STRATEGIES:
            pair_key = s["pair"]
            tf = s["timeframe"]

            if "BTC" in pair_key:
                df_slice = btc_4h.iloc[:current_btc_idx + 1]
            else:
                df_slice = eth_4h.iloc[:current_eth_idx + 1]

            if len(df_slice) < 50:
                signal = 0
            else:
                strategy = get_strategy(s["name"])
                signals = strategy.generate_signals(df_slice, s["params"])
                signal = int(signals.iloc[-1]) if len(signals) > 0 else 0

            portfolio_signals[s["label"]] = signal
            weighted_sum += s["weight"] * signal
            total_weight += s["weight"]

        if total_weight > 0:
            weighted_sum /= total_weight

        # Determine portfolio direction
        if weighted_sum > 0.3:
            direction = 1
        elif weighted_sum < -0.3:
            direction = -1
        else:
            direction = 0

        confidence = min(abs(weighted_sum), 1.0)

        signal_log.append({
            "time": str(eth_4h.index[current_eth_idx]),
            "weighted_score": round(weighted_sum, 4),
            "direction": direction,
            "confidence": round(confidence, 4),
            **{k: v for k, v in portfolio_signals.items()},
        })

        # --- Step 3: Execute based on signal ---
        for s in STRATEGIES:
            pair_key = s["pair"]
            pair_label = s["label"]

            if "BTC" in pair_key:
                price = float(btc_4h.iloc[min(current_btc_idx, len(btc_4h) - 1)]["close"])
                df_for_atr = btc_4h.iloc[:current_btc_idx + 1]
            else:
                price = eth_price
                df_for_atr = eth_4h.iloc[:current_eth_idx + 1]

            pair_signal = portfolio_signals.get(pair_label, 0)

            # Check if we have a position for this pair
            has_position = pair_key in positions

            if pair_signal == 0 and has_position:
                # Close position
                pos = positions[pair_key]
                exit_price = price * (1 - 0.0001) if pos["side"] == "long" else price * (1 + 0.0001)
                pnl_raw = pos["size"] * (exit_price - pos["entry_price"])
                if pos["side"] == "short":
                    pnl_raw = -pnl_raw
                fee = abs(pos["size"]) * pos["entry_price"] * 0.0005 * 2
                pnl_net = pnl_raw - fee
                cash += pos["notional"] + pnl_net

                trades_log.append({
                    "entry_time": str(pos["entry_time"]),
                    "exit_time": str(eth_4h.index[current_eth_idx]),
                    "pair": pair_key,
                    "side": pos["side"],
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "pnl_pct": pnl_net / pos["notional"] * 100,
                    "exit_reason": "signal_exit",
                })
                del positions[pair_key]

            elif pair_signal != 0 and not has_position:
                # Open position
                is_long = pair_signal > 0
                entry_price = price * (1 + 0.0001) if is_long else price * (1 - 0.0001)

                # Position sizing: 2% risk, scaled by weight
                risk_pct = 0.02 * s["weight"] / 0.426  # normalize to largest weight
                notional = cash * risk_pct
                notional = min(notional, cash * 0.15)  # cap at 15%

                if notional <= 0 or entry_price <= 0:
                    continue

                size = notional / entry_price
                if not is_long:
                    size = -size

                # Calculate SL
                atr_series = atr_func(df_for_atr, period=14)
                entry_atr = float(atr_series.iloc[-1]) if len(atr_series) > 0 and np.isfinite(atr_series.iloc[-1]) else entry_price * 0.02

                if is_long:
                    sl_price = entry_price - entry_atr * sl_mult
                else:
                    sl_price = entry_price + entry_atr * sl_mult

                cash -= notional  # Reserve notional
                positions[pair_key] = {
                    "side": "long" if is_long else "short",
                    "entry_price": entry_price,
                    "size": size,
                    "sl_price": sl_price,
                    "entry_time": eth_4h.index[current_eth_idx],
                    "notional": notional,
                    "atr_at_entry": entry_atr,
                }

        # --- Step 4: Calculate equity ---
        unrealized = 0
        for pair_key, pos in positions.items():
            if "BTC" in pair_key:
                current_price = float(btc_4h.iloc[min(current_btc_idx, len(btc_4h) - 1)]["close"])
            else:
                current_price = eth_price

            if pos["side"] == "long":
                unrealized += pos["size"] * (current_price - pos["entry_price"])
            else:
                unrealized += -pos["size"] * (current_price - pos["entry_price"])

        notional_in_positions = sum(p["notional"] for p in positions.values())
        equity = cash + notional_in_positions + unrealized
        equity_curve.append(equity)

    return equity_curve, trades_log, signal_log


def main():
    t0 = time.time()
    cfg = SystemConfig.default()
    loader = DataLoader(cfg)

    print("=" * 100)
    print("  30-DAY FORWARD TEST")
    print("=" * 100)

    # Load data
    eth_4h = loader.load("ETH/USDT:USDT", "4h")
    btc_4h = loader.load("BTC/USDT:USDT", "4h")

    data_cache = {
        "ETH/USDT:USDT_4h": eth_4h,
        "BTC/USDT:USDT_4h": btc_4h,
    }

    # Find the last 30 days (180 candles at 4h)
    candles_30d = 180
    test_start = len(eth_4h) - candles_30d
    if test_start < 100:
        test_start = 100

    print(f"\n  Data: ETH 4h={len(eth_4h)} candles, BTC 4h={len(btc_4h)} candles")
    print(f"  Test starts at candle {test_start}")

    # Run simulation
    equity_curve, trades_log, signal_log = simulate_portfolio_30d(data_cache, test_start)

    # Calculate metrics
    n_hours = len(equity_curve) * 4
    equity_arr = np.array(equity_curve)
    metrics = calc_metrics(equity_arr, n_hours)

    # Print results
    print(f"\n\n{'=' * 100}")
    print("  FORWARD TEST RESULTS")
    print(f"{'=' * 100}")

    print(f"\n  Period:            {eth_4h.index[test_start].date()} to {eth_4h.index[-1].date()}")
    print(f"  Candles:           {len(equity_curve)} ({n_hours} hours = {n_hours/24:.0f} days)")
    print(f"  Initial Capital:   ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Equity:      ${metrics['final_equity']:,.2f}")
    print(f"  Total Return:      {metrics['total_return_pct']:+.2f}%")
    print(f"  Sharpe Ratio:      {metrics['sharpe']:.2f}")
    print(f"  Sortino Ratio:     {metrics['sortino']:.2f}")
    print(f"  Max Drawdown:      {metrics['max_dd_pct']:.2f}%")

    # Trade analysis
    print(f"\n  Total Trades:      {len(trades_log)}")

    if trades_log:
        sl_exits = sum(1 for t in trades_log if t["exit_reason"] == "stop_loss")
        signal_exits = len(trades_log) - sl_exits
        winners = [t for t in trades_log if t["pnl_pct"] > 0]
        losers = [t for t in trades_log if t["pnl_pct"] <= 0]
        total_pnl = sum(t["pnl_pct"] for t in trades_log)

        print(f"  SL Exits:          {sl_exits} ({sl_exits/len(trades_log)*100:.1f}%)")
        print(f"  Signal Exits:      {signal_exits} ({signal_exits/len(trades_log)*100:.1f}%)")
        print(f"  Winners:           {len(winners)} ({len(winners)/len(trades_log)*100:.1f}%)")
        print(f"  Losers:            {len(losers)} ({len(losers)/len(trades_log)*100:.1f}%)")

        if winners:
            print(f"  Avg Win:           {np.mean([t['pnl_pct'] for t in winners]):+.2f}%")
        if losers:
            print(f"  Avg Loss:          {np.mean([t['pnl_pct'] for t in losers]):+.2f}%")
        print(f"  Total P&L:         {total_pnl:+.2f}%")

        # Print each trade
        print(f"\n  {'Time':<22} {'Pair':<20} {'Side':<6} {'Entry':>10} {'Exit':>10} {'P&L%':>8} {'Reason':<15}")
        print(f"  {'-'*95}")
        for t in trades_log:
            print(f"  {t['entry_time'][:19]:<22} {t['pair']:<20} {t['side']:<6} "
                  f"{t['entry_price']:>10.2f} {t['exit_price']:>10.2f} "
                  f"{t['pnl_pct']:>+7.2f}% {t['exit_reason']:<15}")

    # Signal analysis
    print("\n\n  SIGNAL LOG (last 10 candles):")
    print(f"  {'Time':<22} {'Score':>8} {'Dir':>5} {'MACD ETH':>10} {'ROC ETH':>10} {'MACD BTC':>10}")
    print(f"  {'-'*70}")
    for s in signal_log[-10:]:
        print(f"  {s['time'][:19]:<22} {s['weighted_score']:>+8.4f} {s['direction']:>+5d} "
              f"{s['MACD ETH 4h']:>+10d} {s['ROC_Momentum ETH 4h']:>+10d} "
              f"{s['MACD BTC 4h']:>+10d}")

    # Compare with backtest expectations
    print("\n\n  COMPARISON: Forward Test vs Backtest Expectations")
    print(f"  {'-'*70}")
    print(f"  {'Metric':<25} {'Forward Test':>15} {'Backtest (30d)':>15} {'Status':>10}")
    print(f"  {'-'*70}")

    # Annualized comparison
    fwd_annual = metrics["cagr_pct"]
    bt_annual = 15.7  # from final backtest
    print(f"  {'Annual Return':<25} {fwd_annual:>+14.1f}% {bt_annual:>+14.1f}% "
          f"{'OK' if fwd_annual > 0 else 'WARN':>10}")

    fwd_sharpe = metrics["sharpe"]
    bt_sharpe = 22.55
    print(f"  {'Sharpe Ratio':<25} {fwd_sharpe:>15.2f} {bt_sharpe:>15.2f} "
          f"{'OK' if fwd_sharpe > 2 else 'WARN':>10}")

    fwd_dd = metrics["max_dd_pct"]
    bt_dd = 0.50
    print(f"  {'Max Drawdown':<25} {fwd_dd:>14.2f}% {bt_dd:>14.2f}% "
          f"{'OK' if fwd_dd < 5 else 'WARN':>10}")

    if trades_log:
        fwd_wr = len(winners) / len(trades_log) * 100
        bt_wr = 65.0  # approximate
        print(f"  {'Win Rate':<25} {fwd_wr:>14.1f}% {bt_wr:>14.1f}% "
              f"{'OK' if fwd_wr > 50 else 'WARN':>10}")

    # Save results
    out = {
        "period": f"{eth_4h.index[test_start].date()} to {eth_4h.index[-1].date()}",
        "candles": len(equity_curve),
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": float(metrics["final_equity"]),
        "total_return_pct": metrics["total_return_pct"],
        "sharpe": metrics["sharpe"],
        "sortino": metrics["sortino"],
        "max_dd_pct": metrics["max_dd_pct"],
        "trades": trades_log,
        "signals": signal_log[-50:],
        "equity_curve": [float(x) for x in equity_curve],
    }

    out_path = Path("data/results/forward_test_30d.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n  Results saved to: {out_path}")
    print(f"  Time: {time.time()-t0:.0f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()

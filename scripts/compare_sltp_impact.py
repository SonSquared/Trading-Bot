#!/usr/bin/env python3
"""
Compare Backtest: Signal-Only Exits vs ATR-Based SL/TP.

For each of the 3 strategies, runs the backtest and then
replays the trades with simulated SL/TP to see the impact.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.backtester.engine import BacktestEngine
from trading_system.strategies import get_strategy
from trading_system.indicators import atr

RISK_FREE_RATE = 0.04
TRADING_HOURS_PER_YEAR = 365 * 24
INITIAL_CAPITAL = 10000.0

STRATEGIES = [
    {
        "name": "MACD",
        "label": "MACD ETH 4h",
        "pair": "ETH/USDT:USDT",
        "timeframe": "4h",
        "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False},
    },
    {
        "name": "ROC_Momentum",
        "label": "ROC ETH 4h",
        "pair": "ETH/USDT:USDT",
        "timeframe": "4h",
        "params": {"roc_period": 3, "roc_threshold": -1, "smooth_period": 1,
                   "trend_ema": 50, "trend_filter": False},
    },
    {
        "name": "MACD",
        "label": "MACD BTC 4h",
        "pair": "BTC/USDT:USDT",
        "timeframe": "4h",
        "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False},
    },
]

SLTP_CONFIGS = [
    {"name": "No SL/TP", "sl_mult": 0, "tp_mult": 0, "trailing": False, "breakeven": False},
    {"name": "SL only (2xATR)", "sl_mult": 2.0, "tp_mult": 0, "trailing": False, "breakeven": False},
    {"name": "SL+TP (2x/3x ATR)", "sl_mult": 2.0, "tp_mult": 3.0, "trailing": False, "breakeven": False},
    {"name": "Full (SL+TP+BE+Trail)", "sl_mult": 2.0, "tp_mult": 3.0, "trailing": True, "breakeven": True},
]


def calc_metrics_from_equity(equity_arr, n_hours):
    """Calculate metrics from a numpy equity array."""
    if len(equity_arr) < 2:
        return {"error": "no data"}

    total_return = (equity_arr[-1] / equity_arr[0]) - 1
    n_years = n_hours / TRADING_HOURS_PER_YEAR if n_hours > 0 else 1
    cagr = (equity_arr[-1] / equity_arr[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

    returns = np.diff(equity_arr) / np.where(equity_arr[:-1] != 0, equity_arr[:-1], 1.0)
    returns = np.where(np.isfinite(returns), returns, 0.0)

    vol = float(np.std(returns, ddof=1) * np.sqrt(TRADING_HOURS_PER_YEAR)) if len(returns) > 1 else 0
    excess = returns - RISK_FREE_RATE / TRADING_HOURS_PER_YEAR
    sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(TRADING_HOURS_PER_YEAR)) if len(excess) > 1 and np.std(excess) > 0 else 0

    neg_rets = returns[returns < 0]
    downside_dev = float(np.std(neg_rets, ddof=1) * np.sqrt(TRADING_HOURS_PER_YEAR)) if len(neg_rets) > 1 else 0
    sortino = (cagr - RISK_FREE_RATE) / downside_dev if downside_dev > 0 else 0

    running_max = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - running_max) / np.where(running_max > 0, running_max, 1.0)
    max_dd = abs(float(drawdown.min()))
    calmar = cagr / max_dd if max_dd > 0 else 0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "volatility": vol,
    }


def simulate_with_sltp(trades, df, atr_series, sl_mult, tp_mult,
                       trailing, breakeven):
    """
    Replay trades with SL/TP logic applied to each trade.

    For each trade:
    - Check every candle's high/low against SL/TP levels
    - Exit at the first SL or TP hit during the trade
    - Optionally apply breakeven and trailing stop

    Returns simulated equity curve and trade list.
    """
    if not trades or sl_mult == 0 and tp_mult == 0:
        return None, trades

    simulated_trades = []
    equity = INITIAL_CAPITAL

    for trade in trades:
        entry_time = trade["entry_time"]
        exit_time = trade["exit_time"]
        entry_price = trade["entry_price"]
        position_size = trade["position_size"]
        is_long = position_size > 0

        # Get ATR at entry
        entry_idx = df.index.get_indexer([entry_time], method="nearest")[0]
        entry_atr = float(atr_series.iloc[entry_idx]) if entry_idx < len(atr_series) else entry_price * 0.02

        # Set SL/TP levels
        if is_long:
            sl_price = entry_price - entry_atr * sl_mult if sl_mult > 0 else 0
            tp_price = entry_price + entry_atr * tp_mult if tp_mult > 0 else float("inf")
        else:
            sl_price = entry_price + entry_atr * sl_mult if sl_mult > 0 else float("inf")
            tp_price = entry_price - entry_atr * tp_mult if tp_mult > 0 else 0

        # Get trade window
        trade_mask = (df.index >= entry_time) & (df.index <= exit_time)
        trade_candles = df[trade_mask]

        if trade_candles.empty:
            # No candles in range, use original trade
            equity += trade["pnl"]
            simulated_trades.append(trade)
            continue

        # Track for trailing stop
        highest_pnl = 0.0
        breakeven_hit = False
        current_sl = sl_price

        # Check each candle in the trade
        exit_price = None
        exit_reason = "signal"

        for idx_i in range(len(trade_candles)):
            candle = trade_candles.iloc[idx_i]
            high = candle["high"]
            low = candle["low"]

            # Check breakeven
            if breakeven and not breakeven_hit:
                if is_long:
                    unrealized = (high - entry_price) / entry_price
                else:
                    unrealized = (entry_price - low) / entry_price
                if unrealized >= 0.015:  # 1.5% profit
                    breakeven_hit = True
                    current_sl = entry_price * 1.001  # Slight profit

            # Check trailing stop
            if trailing:
                if is_long:
                    unrealized_high = (high - entry_price) / entry_price
                    if unrealized_high > highest_pnl:
                        highest_pnl = unrealized_high
                    if highest_pnl * entry_price > entry_atr:
                        new_trail = high - entry_atr * 2.5
                        if new_trail > current_sl:
                            current_sl = new_trail
                else:
                    unrealized_high = (entry_price - low) / entry_price
                    if unrealized_high > highest_pnl:
                        highest_pnl = unrealized_high
                    if highest_pnl * entry_price > entry_atr:
                        new_trail = low + entry_atr * 2.5
                        if new_trail < current_sl:
                            current_sl = new_trail

            # Check stop-loss
            if sl_mult > 0:
                if is_long and low <= current_sl:
                    exit_price = current_sl
                    exit_reason = "trailing_stop" if breakeven_hit and current_sl > sl_price else "stop_loss"
                    break
                elif not is_long and high >= current_sl:
                    exit_price = current_sl
                    exit_reason = "trailing_stop" if breakeven_hit and current_sl < sl_price else "stop_loss"
                    break

            # Check take-profit
            if tp_mult > 0:
                if is_long and high >= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"
                    break
                elif not is_long and low <= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"
                    break

        # If no SL/TP hit, use original exit
        if exit_price is None:
            exit_price = trade["exit_price"]
            exit_reason = "signal"

        # Calculate P&L
        pnl_raw = position_size * (exit_price - entry_price)
        # Simple fee approximation
        notional = abs(position_size * entry_price)
        fee = notional * 0.0004 * 2  # taker fee both sides
        pnl_net = pnl_raw - fee

        equity += pnl_net
        simulated_trades.append({
            **trade,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl": pnl_net,
            "original_pnl": trade["pnl"],
        })

    return equity, simulated_trades


def main():
    t0 = time.time()
    cfg = SystemConfig.default()
    loader = DataLoader(cfg)
    bt_config = cfg.backtest
    engine = BacktestEngine(bt_config)

    print("=" * 110)
    print("  SL/TP IMPACT COMPARISON")
    print("=" * 110)

    results_table = []

    for strat_def in STRATEGIES:
        strat_name = strat_def["name"]
        label = strat_def["label"]
        pair = strat_def["pair"]
        tf = strat_def["timeframe"]
        params = strat_def["params"]

        print(f"\n{'=' * 110}")
        print(f"  {label} ({strat_name} on {pair} {tf})")
        print(f"  Params: {json.dumps(params)}")
        print(f"{'=' * 110}")

        # Load data and run base backtest
        strategy = get_strategy(strat_name)
        df = loader.load(pair, tf)
        funding = loader.load_funding_rates(pair)
        signals = strategy.generate_signals(df, params)
        base_result = engine.run(df, signals, strat_name, params, pair, tf, funding)
        atr_series = atr(df, period=14)

        n_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
        trades = base_result.trades

        print(f"  Base backtest: {len(trades)} trades, Sharpe={base_result.sharpe:.2f}, "
              f"Return={base_result.total_return*100:.1f}%, MaxDD={base_result.max_drawdown*100:.2f}%")

        for sltp_cfg in SLTP_CONFIGS:
            if sltp_cfg["sl_mult"] == 0 and sltp_cfg["tp_mult"] == 0:
                # No SL/TP - use base results
                metrics = {
                    "total_return": base_result.total_return,
                    "cagr": base_result.cagr,
                    "sharpe": base_result.sharpe,
                    "sortino": base_result.sortino,
                    "max_drawdown": base_result.max_drawdown,
                    "trades": len(trades),
                    "sl_exits": 0,
                    "tp_exits": 0,
                    "be_exits": 0,
                    "signal_exits": len(trades),
                }
            else:
                final_equity, sim_trades = simulate_with_sltp(
                    trades, df, atr_series,
                    sltp_cfg["sl_mult"], sltp_cfg["tp_mult"],
                    sltp_cfg["trailing"], sltp_cfg["breakeven"],
                )

                if final_equity is None:
                    continue

                # Build equity curve from simulated trades
                sim_equity = np.full(len(df), INITIAL_CAPITAL, dtype=np.float64)
                running = INITIAL_CAPITAL
                for t in sim_trades:
                    running += t["pnl"]
                    # Mark exit candle
                    try:
                        exit_idx = df.index.get_indexer([t["exit_time"]], method="nearest")[0]
                        sim_equity[exit_idx:] = running
                    except:
                        pass

                sim_metrics = calc_metrics_from_equity(sim_equity, n_hours)

                # Count exit reasons
                sl_exits = sum(1 for t in sim_trades if t.get("exit_reason") == "stop_loss")
                tp_exits = sum(1 for t in sim_trades if t.get("exit_reason") == "take_profit")
                trail_exits = sum(1 for t in sim_trades if t.get("exit_reason") == "trailing_stop")
                signal_exits = sum(1 for t in sim_trades if t.get("exit_reason") == "signal")
                be_exits = trail_exits  # trailing stop includes breakeven exits

                metrics = {
                    **sim_metrics,
                    "trades": len(sim_trades),
                    "sl_exits": sl_exits,
                    "tp_exits": tp_exits,
                    "be_exits": be_exits,
                    "signal_exits": signal_exits,
                }

            # Print result
            print(f"\n  {sltp_cfg['name']}:")
            print(f"    Return:     {metrics['total_return']*100:>+7.1f}%  "
                  f"({(metrics['total_return']-base_result.total_return)*100:>+.1f}% vs base)")
            print(f"    Sharpe:     {metrics['sharpe']:>+7.2f}  "
                  f"({metrics['sharpe']-base_result.sharpe:>+.2f} vs base)")
            print(f"    MaxDD:      {metrics['max_drawdown']*100:>7.2f}%  "
                  f"({(metrics['max_drawdown']-base_result.max_drawdown)*100:>+.2f}% vs base)")
            print(f"    Sortino:    {metrics['sortino']:>+7.2f}")
            print(f"    Trades:     {metrics['trades']:>7d}")
            print(f"    Exit breakdown: SL={metrics['sl_exits']}  TP={metrics['tp_exits']}  "
                  f"Trail={metrics['be_exits']}  Signal={metrics['signal_exits']}")

            results_table.append({
                "strategy": label,
                "sltp_config": sltp_cfg["name"],
                **metrics,
            })

    # Summary table
    print(f"\n{'=' * 110}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 110}")

    print(f"\n  {'Strategy':<20s} {'SL/TP Config':<22s} {'Return':>8s} {'Sharpe':>8s} "
          f"{'MaxDD':>8s} {'Sortino':>8s} {'SL':>4s} {'TP':>4s} {'Sig':>4s}")
    print(f"  {'-' * 100}")

    for r in results_table:
        print(f"  {r['strategy']:<20s} {r['sltp_config']:<22s} "
              f"{r['total_return']*100:>+7.1f}% {r['sharpe']:>+8.2f} "
              f"{r['max_drawdown']*100:>7.2f}% {r['sortino']:>+8.2f} "
              f"{r['sl_exits']:>4d} {r['tp_exits']:>4d} {r['signal_exits']:>4d}")

    # Key insights
    print(f"\n{'=' * 110}")
    print("  KEY INSIGHTS")
    print(f"{'=' * 110}")

    for strat_def in STRATEGIES:
        label = strat_def["label"]
        base = next((r for r in results_table
                      if r["strategy"] == label and r["sltp_config"] == "No SL/TP"), None)
        full = next((r for r in results_table
                      if r["strategy"] == label and r["sltp_config"] == "Full (SL+TP+BE+Trail)"), None)
        if base and full:
            ret_impact = (full["total_return"] - base["total_return"]) * 100
            sharpe_impact = full["sharpe"] - base["sharpe"]
            dd_impact = (full["max_drawdown"] - base["max_drawdown"]) * 100

            print(f"\n  {label}:")
            print(f"    Return impact:  {ret_impact:>+.1f}%")
            print(f"    Sharpe impact:  {sharpe_impact:>+.2f}")
            print(f"    MaxDD impact:   {dd_impact:>+.2f}%")
            print(f"    SL exits:       {full['sl_exits']}/{full['trades']} trades ({full['sl_exits']/max(full['trades'],1)*100:.0f}%)")
            print(f"    TP exits:       {full['tp_exits']}/{full['trades']} trades ({full['tp_exits']/max(full['trades'],1)*100:.0f}%)")
            print(f"    Signal exits:   {full['signal_exits']}/{full['trades']} trades ({full['signal_exits']/max(full['trades'],1)*100:.0f}%)")

    # Save
    out_path = Path("data/results/sltp_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results_table, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print(f"  Time: {time.time()-t0:.0f}s")
    print("=" * 110)


if __name__ == "__main__":
    main()

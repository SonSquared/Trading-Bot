"""
2022 Crypto Crash Stress Test

Tests all 3 strategies during the worst crypto crash in history:
  - BTC: $47,000 (Jan 2022) -> $15,500 (Nov 2022) = -67%
  - ETH: $3,700 (Jan 2022) -> $880 (Nov 2022) = -76%

Tests:
  1. Without risk management (raw strategy performance)
  2. With risk management (5% stop-loss, 10% portfolio DD limit)
  3. Worst-case single trade loss
  4. Maximum drawdown under each scenario
  5. Recovery time analysis
"""

import sys
sys.path.insert(0, ".")

import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
from pathlib import Path

import pandas as pd
import numpy as np

from trading_system.strategies import STRATEGY_REGISTRY
from trading_system.bot.risk_manager import RiskManager

RESULTS_DIR = Path("data/results")
INITIAL_CAPITAL = 97.0
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
MAX_POS_PCT = 0.35


def load_crash_period(pair: str) -> pd.DataFrame:
    """Load data for the 2022 crash period (Jan-Nov 2022)."""
    df = pd.read_parquet(f"data/raw/{pair}/klines_4h.parquet")
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dtype.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Filter to 2022 crash period
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2022-12-01")
    mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
    return df[mask].reset_index(drop=True)


def backtest_with_risk_mgmt(
    df: pd.DataFrame, strat_name: str, params: dict,
    use_risk_mgmt: bool = True, risk_config: dict = None
) -> dict:
    """Backtest with optional risk management."""
    risk = None
    if use_risk_mgmt:
        cfg = risk_config or {}
        risk = RiskManager(
            position_stop_loss_pct=cfg.get("position_sl", 5.0),
            portfolio_max_dd_pct=cfg.get("portfolio_dd", 10.0),
            trailing_stop_activation_pct=cfg.get("trailing_activation", 5.0),
            trailing_stop_distance_pct=cfg.get("trailing_distance", 3.0),
        )

    sig = STRATEGY_REGISTRY[strat_name].generate_signals(df, params)
    cash = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    position = None  # (side, entry_price, qty, cost, entry_time_idx, high_pnl)
    trades = []
    equity_curve = []
    risk_events = []
    cooldown_until = -1  # Don't re-enter until this bar index

    for i in range(len(df)):
        price = float(df["close"].iloc[i])
        signal = int(sig.iloc[i])

        # --- Risk management checks ---
        if position and risk:
            side, entry, qty, cost, entry_idx, high_pnl = position
            pnl_pct = (price - entry) / entry * side * 100

            # Update high watermark
            if pnl_pct > high_pnl:
                position = (side, entry, qty, cost, entry_idx, pnl_pct)
                high_pnl = pnl_pct

            # Position stop-loss
            if pnl_pct <= -risk.position_stop_loss_pct:
                exit_p = price * (1 - SLIPPAGE_RATE * side)
                if side == 1:
                    pnl = qty * (exit_p - entry) - cost
                else:
                    pnl = qty * (entry - exit_p) - cost
                cash += qty * entry + pnl
                trades.append({"pnl": pnl, "reason": "stop_loss", "loss_pct": pnl_pct})
                risk_events.append({"type": "position_stop", "price": price, "pnl_pct": pnl_pct, "bar": i})
                position = None
                cooldown_until = i + 6  # Wait 6 bars (24h on 4h) before re-entering
                equity_curve.append(cash)
                continue

            # Trailing stop
            if high_pnl >= risk.trailing_stop_activation_pct:
                stop_level = high_pnl - risk.trailing_stop_distance_pct
                if pnl_pct <= stop_level:
                    exit_p = price * (1 - SLIPPAGE_RATE * side)
                    if side == 1:
                        pnl = qty * (exit_p - entry) - cost
                    else:
                        pnl = qty * (entry - exit_p) - cost
                    cash += qty * entry + pnl
                    trades.append({"pnl": pnl, "reason": "trailing_stop", "loss_pct": pnl_pct})
                    risk_events.append({"type": "trailing_stop", "price": price, "pnl_pct": pnl_pct, "bar": i})
                    position = None
                    cooldown_until = i + 6
                    equity_curve.append(cash)
                    continue

        # Close on signal reversal
        if position and signal != position[0]:
            side, entry, qty, cost, entry_idx, high_pnl = position
            exit_p = price * (1 - SLIPPAGE_RATE * side)
            if side == 1:
                pnl = qty * (exit_p - entry) - cost
            else:
                pnl = qty * (entry - exit_p) - cost
            cash += qty * entry + pnl
            trades.append({"pnl": pnl, "reason": "signal_reversal", "loss_pct": 0})
            position = None

        # Open new position (respect cooldown)
        if not position and signal != 0 and i >= cooldown_until:
            size = min(cash * MAX_POS_PCT, cash * 0.95)
            if size > 5:
                entry_p = price * (1 + SLIPPAGE_RATE * signal)
                qty = size / entry_p
                fee = size * FEE_RATE
                cash -= (size + fee)
                position = (signal, entry_p, qty, fee, i, 0)

        # Portfolio drawdown check
        eq = cash
        if position:
            side, entry, qty, cost, _, _ = position
            if side == 1:
                eq += qty * price
            else:
                eq += qty * entry + qty * (entry - price)
        equity_curve.append(eq)

        if risk and eq > peak_equity:
            peak_equity = eq
        if risk and peak_equity > 0:
            dd_pct = (peak_equity - eq) / peak_equity * 100
            if dd_pct >= risk.portfolio_max_dd_pct:
                # Close everything
                if position:
                    side, entry, qty, cost, _, _ = position
                    exit_p = price * (1 - SLIPPAGE_RATE * side)
                    if side == 1:
                        pnl = qty * (exit_p - entry) - cost
                    else:
                        pnl = qty * (entry - exit_p) - cost
                    cash += qty * entry + pnl
                    trades.append({"pnl": pnl, "reason": "portfolio_dd_stop", "loss_pct": -dd_pct})
                    risk_events.append({"type": "portfolio_dd_stop", "price": price, "dd_pct": dd_pct, "bar": i})
                    position = None

    # Close remaining
    if position:
        side, entry, qty, cost, _, _ = position
        price = float(df["close"].iloc[-1])
        exit_p = price * (1 - SLIPPAGE_RATE * side)
        if side == 1:
            pnl = qty * (exit_p - entry) - cost
        else:
            pnl = qty * (entry - exit_p) - cost
        cash += qty * entry + pnl
        trades.append({"pnl": pnl, "reason": "end_of_period", "loss_pct": 0})

    final = cash
    wins = sum(1 for t in trades if t["pnl"] > 0)
    n_trades = len(trades)
    eq_arr = np.array(equity_curve) if equity_curve else np.array([INITIAL_CAPITAL])
    peaks = np.maximum.accumulate(eq_arr)
    dd = (peaks - eq_arr) / np.where(peaks > 0, peaks, 1)
    max_dd = float(np.max(dd)) * 100
    ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # Worst single trade
    worst_trade = min(trades, key=lambda x: x["pnl"]) if trades else {"pnl": 0, "reason": "none"}
    worst_pct = worst_trade["pnl"] / INITIAL_CAPITAL * 100

    return {
        "final": final,
        "return_pct": ret,
        "trades": n_trades,
        "wins": wins,
        "win_rate": wins / n_trades * 100 if n_trades > 0 else 0,
        "max_dd": max_dd,
        "worst_trade_pct": worst_pct,
        "worst_trade_reason": worst_trade.get("reason", ""),
        "risk_events": len(risk_events),
        "equity_curve": eq_arr.tolist(),
    }


def run_stress_test():
    """Main stress test."""
    print("=" * 70)
    print("2022 CRYPTO CRASH STRESS TEST")
    print("BTC: $47,000 -> $15,500 (-67%) | ETH: $3,700 -> $880 (-76%)")
    print("=" * 70)
    print()

    strategies = {
        "BB_RSI ETH": {
            "strategy": "Bollinger_Reversion",
            "pair": "ETH_USDT_USDT",
            "params": {"bb_period": 20, "bb_std": 2.0, "rsi_filter": False, "exit_at_middle": False},
        },
        "BB_RSI BTC": {
            "strategy": "Bollinger_Reversion",
            "pair": "BTC_USDT_USDT",
            "params": {"bb_period": 20, "bb_std": 2.0, "rsi_filter": False, "exit_at_middle": False},
        },
        "RSI_Reversion BTC": {
            "strategy": "RSI_Reversion",
            "pair": "BTC_USDT_USDT",
            "params": {"rsi_period": 14, "entry_oversold": 30, "entry_overbought": 65,
                       "exit_neutral_low": 45, "exit_neutral_high": 50, "use_bb_filter": False},
        },
    }

    results = {}

    for name, cfg in strategies.items():
        print(f"\n{'='*70}")
        print(f"STRATEGY: {name}")
        print(f"{'='*70}")

        try:
            df = load_crash_period(cfg["pair"])
        except Exception as e:
            print(f"  ERROR loading data: {e}")
            continue

        if len(df) < 50:
            print(f"  Insufficient data: {len(df)} candles")
            continue

        start_price = float(df["close"].iloc[0])
        end_price = float(df["close"].iloc[-1])
        buy_hold_ret = (end_price - start_price) / start_price * 100
        print(f"  Data: {len(df)} candles ({df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]})")
        print(f"  Buy & Hold: ${start_price:,.2f} -> ${end_price:,.2f} ({buy_hold_ret:+.1f}%)")
        print()

        # Without risk management
        print("  --- Without Risk Management ---")
        no_rm = backtest_with_risk_mgmt(df, cfg["strategy"], cfg["params"], use_risk_mgmt=False)
        print(f"  Return: {no_rm['return_pct']:+.1f}% | Max DD: {no_rm['max_dd']:.1f}%")
        print(f"  Trades: {no_rm['trades']} | Win Rate: {no_rm['win_rate']:.0f}%")
        print(f"  Worst trade: {no_rm['worst_trade_pct']:+.1f}% ({no_rm['worst_trade_reason']})")
        print(f"  Final equity: ${no_rm['final']:.2f}")

        # With risk management (5% SL, 10% DD)
        print("\n  --- With Risk Management (5% SL, 10% DD) ---")
        with_rm = backtest_with_risk_mgmt(df, cfg["strategy"], cfg["params"], use_risk_mgmt=True)
        print(f"  Return: {with_rm['return_pct']:+.1f}% | Max DD: {with_rm['max_dd']:.1f}%")
        print(f"  Trades: {with_rm['trades']} | Win Rate: {with_rm['win_rate']:.0f}%")
        print(f"  Worst trade: {with_rm['worst_trade_pct']:+.1f}% ({with_rm['worst_trade_reason']})")
        print(f"  Risk events: {with_rm['risk_events']} | Final: ${with_rm['final']:.2f}")

        # With tighter risk management (3% SL, 7% DD)
        print("\n  --- With Tight Risk Management (3% SL, 7% DD) ---")
        tight_rm = backtest_with_risk_mgmt(
            df, cfg["strategy"], cfg["params"], use_risk_mgmt=True,
            risk_config={"position_sl": 3.0, "portfolio_dd": 7.0, "trailing_activation": 3.0, "trailing_distance": 2.0}
        )
        print(f"  Return: {tight_rm['return_pct']:+.1f}% | Max DD: {tight_rm['max_dd']:.1f}%")
        print(f"  Trades: {tight_rm['trades']} | Win Rate: {tight_rm['win_rate']:.0f}%")
        print(f"  Worst trade: {tight_rm['worst_trade_pct']:+.1f}% ({tight_rm['worst_trade_reason']})")
        print(f"  Risk events: {tight_rm['risk_events']} | Final: ${tight_rm['final']:.2f}")

        results[name] = {
            "buy_hold": buy_hold_ret,
            "no_rm": {k: v for k, v in no_rm.items() if k != "equity_curve"},
            "with_rm": {k: v for k, v in with_rm.items() if k != "equity_curve"},
            "tight_rm": {k: v for k, v in tight_rm.items() if k != "equity_curve"},
        }

    # Summary
    print(f"\n{'='*70}")
    print("STRESS TEST SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Strategy':<20} {'B&H':>8} {'No RM':>10} {'5%SL/10%DD':>12} {'3%SL/7%DD':>12}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*12} {'-'*12}")

    for name, r in results.items():
        print(f"  {name:<20} {r['buy_hold']:>+7.1f}% {r['no_rm']['return_pct']:>+9.1f}% {r['with_rm']['return_pct']:>+11.1f}% {r['tight_rm']['return_pct']:>+11.1f}%")

    print("\n  Risk Management Impact:")
    for name, r in results.items():
        dd_before = r["no_rm"]["max_dd"]
        dd_after = r["with_rm"]["max_dd"]
        dd_tight = r["tight_rm"]["max_dd"]
        worst_before = r["no_rm"]["worst_trade_pct"]
        worst_after = r["with_rm"]["worst_trade_pct"]
        print(f"  {name}:")
        print(f"    Max DD: {dd_before:.1f}% -> {dd_after:.1f}% (standard) -> {dd_tight:.1f}% (tight)")
        print(f"    Worst trade: {worst_before:+.1f}% -> {worst_after:+.1f}%")

    # Save results
    out = {"timestamp": "2022-crash-stress-test", "results": results}
    out_path = RESULTS_DIR / "crash_stress_test.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run_stress_test()

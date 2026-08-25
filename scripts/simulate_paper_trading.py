#!/usr/bin/env python3
"""
Simulated Paper Trading — 30-Day End-to-End Test.

Walks through 30 days of 4h candles one at a time, simulating
the portfolio bot's signal aggregation, position management,
and SL protection exactly as it would run live.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.strategies import get_strategy
from trading_system.indicators import atr

RISK_FREE_RATE = 0.04
TRADING_HOURS_PER_YEAR = 365 * 24
INITIAL_CAPITAL = 10000.0
SL_ATR_MULT = 3.0
RISK_PER_TRADE = 0.02
LEVERAGE = 1.0
TAKER_FEE = 0.0004

STRATEGIES = [
    {
        "name": "MACD",
        "label": "MACD ETH",
        "pair": "ETH/USDT:USDT",
        "timeframe": "4h",
        "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False},
        "weight": 0.374,
    },
    {
        "name": "ROC_Momentum",
        "label": "ROC ETH",
        "pair": "ETH/USDT:USDT",
        "timeframe": "4h",
        "params": {"roc_period": 3, "roc_threshold": -1, "smooth_period": 1,
                   "trend_ema": 50, "trend_filter": False},
        "weight": 0.253,
    },
    {
        "name": "MACD",
        "label": "MACD BTC",
        "pair": "BTC/USDT:USDT",
        "timeframe": "4h",
        "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False},
        "weight": 0.374,
    },
]


class PaperTradingSimulator:
    """Simulates the portfolio bot's live trading logic tick-by-tick."""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, dict] = {}  # pair -> position info
        self.trade_log: list[dict] = []
        self.equity_history: list[dict] = []
        self.signal_log: list[dict] = []

    def get_signal(self, pair: str, data: dict[str, pd.DataFrame]) -> dict:
        """Aggregate signals from all strategies for a pair."""
        strategies_for_pair = [s for s in STRATEGIES if s["pair"] == pair]
        if not strategies_for_pair:
            return {"direction": 0, "score": 0, "confidence": 0, "signals": {}}

        signals = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for s_def in strategies_for_pair:
            tf = s_def["timeframe"]
            df = data.get(tf)
            if df is None or df.empty or len(df) < 50:
                signals[s_def["label"]] = 0
                continue

            strategy = get_strategy(s_def["name"])
            sigs = strategy.generate_signals(df, s_def["params"])
            sig = int(sigs.iloc[-1]) if len(sigs) > 0 else 0
            signals[s_def["label"]] = sig
            weighted_sum += s_def["weight"] * sig
            total_weight += s_def["weight"]

        if total_weight > 0:
            weighted_sum /= total_weight

        confidence = min(abs(weighted_sum), 1.0)
        if weighted_sum > 0.3:
            direction = 1
        elif weighted_sum < -0.3:
            direction = -1
        else:
            direction = 0

        return {
            "direction": direction,
            "score": weighted_sum,
            "confidence": confidence,
            "signals": signals,
        }

    def check_sl(self, pair: str, current_price: float, high: float, low: float,
                 atr_val: float) -> list[dict]:
        """Check if any position's stop-loss is hit."""
        actions = []
        if pair not in self.positions:
            return actions

        pos = self.positions[pair]
        entry = pos["entry_price"]
        is_long = pos["side"] == "long"
        sl_price = pos["stop_loss"]

        if is_long and low <= sl_price:
            actions.append(self._close_position(pair, sl_price, "stop_loss"))
        elif not is_long and high >= sl_price:
            actions.append(self._close_position(pair, sl_price, "stop_loss"))

        return actions

    def process_signal(self, pair: str, direction: int, confidence: float,
                       price: float, atr_val: float, timestamp) -> list[dict]:
        """Process an aggregated signal and manage positions."""
        actions = []
        has_position = pair in self.positions

        if direction == 0:
            # Close any existing position
            if has_position:
                actions.append(self._close_position(pair, price, "signal_exit"))
            return actions

        is_long = direction > 0

        if has_position:
            pos = self.positions[pair]
            pos_is_long = pos["side"] == "long"

            if (is_long and pos_is_long) or (not is_long and not pos_is_long):
                return actions  # Already in right direction

            # Close and reverse
            actions.append(self._close_position(pair, price, "signal_reverse"))

        # Open new position
        risk_pct = RISK_PER_TRADE * min(confidence, 1.0)
        notional = self.cash * risk_pct * LEVERAGE
        notional = min(notional, self.cash * LEVERAGE * 0.95)

        if notional <= 0 or price <= 0:
            return actions

        amount = notional / price
        side = "long" if is_long else "short"

        # Calculate SL
        if is_long:
            sl = price - atr_val * SL_ATR_MULT
        else:
            sl = price + atr_val * SL_ATR_MULT

        # Apply entry fee
        fee = notional * TAKER_FEE
        self.cash -= fee

        self.positions[pair] = {
            "side": side,
            "entry_price": price,
            "size": amount,
            "notional": notional,
            "stop_loss": sl,
            "entry_time": timestamp,
            "entry_fee": fee,
        }

        self.trade_log.append({
            "timestamp": str(timestamp),
            "action": "open",
            "pair": pair,
            "side": side,
            "price": price,
            "size": amount,
            "notional": notional,
            "sl": sl,
            "confidence": confidence,
        })

        return actions

    def _close_position(self, pair: str, exit_price: float, reason: str) -> dict:
        """Close a position and calculate P&L."""
        pos = self.positions.pop(pair)
        entry = pos["entry_price"]
        size = pos["size"]
        is_long = pos["side"] == "long"

        if is_long:
            pnl_raw = size * (exit_price - entry)
        else:
            pnl_raw = size * (entry - exit_price)

        # Apply exit fee
        notional = abs(size * exit_price)
        fee = notional * TAKER_FEE
        pnl_net = pnl_raw - fee

        self.cash += pnl_net

        self.trade_log.append({
            "timestamp": str(exit_price),
            "action": "close",
            "pair": pair,
            "side": "sell" if is_long else "buy",
            "price": exit_price,
            "entry_price": entry,
            "pnl_raw": pnl_raw,
            "pnl_net": pnl_net,
            "fee": pos["entry_fee"] + fee,
            "reason": reason,
        })

        return {"action": "close", "pair": pair, "reason": reason, "pnl": pnl_net}

    def get_equity(self, prices: dict[str, float]) -> float:
        """Calculate current total equity."""
        equity = self.cash
        for pair, pos in self.positions.items():
            price = prices.get(pair, pos["entry_price"])
            if pos["side"] == "long":
                unrealized = pos["size"] * (price - pos["entry_price"])
            else:
                unrealized = pos["size"] * (pos["entry_price"] - price)
            equity += unrealized
        return equity


def run_simulation():
    t0 = time.time()
    cfg = SystemConfig.default()
    loader = DataLoader(cfg)

    print("=" * 100)
    print("  30-DAY SIMULATED PAPER TRADING")
    print("=" * 100)

    # Load data
    eth_4h = loader.load("ETH/USDT:USDT", "4h")
    btc_4h = loader.load("BTC/USDT:USDT", "4h")

    # Get the last 30 days of data (180 candles at 4h)
    n_candles = 180  # 30 days * 6 candles/day
    eth_slice = eth_4h.iloc[-n_candles:]
    btc_slice = btc_4h.iloc[-n_candles:]

    # Align to common index
    common_idx = eth_slice.index.intersection(btc_slice.index)
    eth_aligned = eth_4h.loc[:common_idx[-1]]
    btc_aligned = btc_4h.loc[:common_idx[-1]]

    print(f"\n  Simulation period: {common_idx[0]} to {common_idx[-1]}")
    print(f"  Candles: {len(common_idx)} (30 days at 4h)")
    print(f"  Initial capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  SL: {SL_ATR_MULT}x ATR | Risk/trade: {RISK_PER_TRADE:.0%} | Fee: {TAKER_FEE:.2%}")

    sim = PaperTradingSimulator()
    atr_cache = {}

    # Walk through each candle
    for i, ts in enumerate(common_idx):
        # Build data up to current candle (no look-ahead)
        eth_data = eth_aligned.loc[:ts]
        btc_data = btc_aligned.loc[:ts]

        # Get prices
        eth_price = float(eth_data["close"].iloc[-1])
        btc_price = float(btc_data["close"].iloc[-1])
        prices = {"ETH/USDT:USDT": eth_price, "BTC/USDT:USDT": btc_price}

        # Get ATR
        for pair, data in [("ETH/USDT:USDT", eth_data), ("BTC/USDT:USDT", btc_data)]:
            if len(data) >= 15:
                atr_s = atr(data, period=14)
                atr_cache[pair] = float(atr_s.iloc[-1])
            else:
                atr_cache[pair] = prices[pair] * 0.02

        # Check SL first (before signal generation)
        for pair in list(sim.positions.keys()):
            if pair in prices:
                candle_high = float(data["high"].iloc[-1]) if pair in [("ETH/USDT:USDT", eth_data), ("BTC/USDT:USDT", btc_data)] else prices[pair]
                candle_low = float(data["low"].iloc[-1]) if pair in [("ETH/USDT:USDT", eth_data), ("BTC/USDT:USDT", btc_data)] else prices[pair]
                data_ref = eth_data if pair == "ETH/USDT:USDT" else btc_data
                candle_high = float(data_ref["high"].iloc[-1])
                candle_low = float(data_ref["low"].iloc[-1])
                sim.check_sl(pair, prices[pair], candle_high, candle_low, atr_cache.get(pair, 0))

        # Generate signals
        for pair in ["ETH/USDT:USDT", "BTC/USDT:USDT"]:
            data = eth_data if pair == "ETH/USDT:USDT" else btc_data
            agg = sim.get_signal(pair, {STRATEGIES[0]["timeframe"]: data})

            if agg["direction"] != 0:
                sim.signal_log.append({
                    "timestamp": str(ts),
                    "pair": pair,
                    "direction": agg["direction"],
                    "score": agg["score"],
                    "confidence": agg["confidence"],
                    "signals": agg["signals"],
                })

            sim.process_signal(
                pair, agg["direction"], agg["confidence"],
                prices[pair], atr_cache.get(pair, prices[pair] * 0.02), ts,
            )

        # Record equity
        equity = sim.get_equity(prices)
        sim.equity_history.append({
            "timestamp": str(ts),
            "equity": equity,
            "cash": sim.cash,
            "n_positions": len(sim.positions),
            "eth_price": eth_price,
            "btc_price": btc_price,
        })

    # Close any remaining positions
    final_prices = {"ETH/USDT:USDT": float(eth_aligned["close"].iloc[-1]),
                    "BTC/USDT:USDT": float(btc_aligned["close"].iloc[-1])}
    for pair in list(sim.positions.keys()):
        sim._close_position(pair, final_prices[pair], "simulation_end")

    # ── Results ───────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("  SIMULATION RESULTS")
    print("=" * 100)

    eq_arr = np.array([e["equity"] for e in sim.equity_history])
    total_return = (eq_arr[-1] / eq_arr[0]) - 1
    max_dd = 0
    peak = eq_arr[0]
    for v in eq_arr:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    print(f"\n  Final Equity:  ${eq_arr[-1]:,.2f}")
    print(f"  Total Return:  {total_return*100:+.2f}%")
    print(f"  Max Drawdown:  {max_dd*100:.2f}%")
    print(f"  Total Trades:  {len([t for t in sim.trade_log if t['action'] == 'close'])}")

    # Trade breakdown
    closes = [t for t in sim.trade_log if t["action"] == "close"]
    sl_closes = [t for t in closes if t["reason"] == "stop_loss"]
    signal_closes = [t for t in closes if t["reason"] in ("signal_exit", "signal_reverse")]
    sim_closes = [t for t in closes if t["reason"] == "simulation_end"]

    print(f"\n  Exit Breakdown:")
    print(f"    Signal exits:   {len(signal_closes)}")
    print(f"    SL exits:       {len(sl_closes)}")
    print(f"    End-of-sim:     {len(sim_closes)}")

    if sl_closes:
        sl_pnls = [t["pnl_net"] for t in sl_closes]
        print(f"    SL avg P&L:     ${np.mean(sl_pnls):.2f}")

    # Trade log
    print(f"\n  Trade Log (last 20):")
    print(f"  {'Timestamp':<28s} {'Pair':<18s} {'Action':<8s} {'Side':<6s} {'Price':>10s} {'P&L':>10s} {'Reason'}")
    print(f"  {'-' * 95}")
    for t in sim.trade_log[-20:]:
        pnl = f"${t.get('pnl_net', 0):>+8.2f}" if "pnl_net" in t else ""
        print(f"  {t['timestamp']:<28s} {t['pair']:<18s} {t['action']:<8s} {t.get('side', ''):<6s} "
              f"{t['price']:>10.2f} {pnl:>10s} {t.get('reason', '')}")

    # Daily equity progression
    print(f"\n  Daily Equity Progression:")
    eq_df = pd.DataFrame(sim.equity_history)
    eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
    eq_df.set_index("timestamp", inplace=True)
    daily = eq_df["equity"].resample("1D").last().dropna()
    for date, eq in daily.items():
        bar = "#" * int((eq - INITIAL_CAPITAL) / 50) if eq >= INITIAL_CAPITAL else "-" * int((INITIAL_CAPITAL - eq) / 50)
        print(f"  {date.strftime('%Y-%m-%d')}  ${eq:>10,.2f}  {bar}")

    # Individual strategy comparison (same period)
    print(f"\n{'=' * 100}")
    print("  INDIVIDUAL STRATEGY COMPARISON (same 30-day period)")
    print("=" * 100)

    for s_def in STRATEGIES:
        pair = s_def["pair"]
        data = eth_aligned if pair == "ETH/USDT:USDT" else btc_aligned
        strategy = get_strategy(s_def["name"])
        sigs = strategy.generate_signals(data, s_def["params"])
        # Simple backtest: buy when signal=1, sell when signal=0
        position = sigs.replace(0, np.nan).ffill().fillna(0)
        returns = data["close"].pct_change().fillna(0) * position.shift(1).fillna(0)
        strat_equity = INITIAL_CAPITAL * (1 + returns).cumprod()
        strat_ret = (strat_equity.iloc[-1] / INITIAL_CAPITAL) - 1
        strat_dd = ((strat_equity.cummax() - strat_equity) / strat_equity.cummax()).max()
        print(f"  {s_def['label']:<15s} Return: {strat_ret*100:>+7.2f}%  MaxDD: {strat_dd*100:.2f}%  Final: ${strat_equity.iloc[-1]:,.0f}")

    # Save
    output = {
        "period": f"{common_idx[0]} to {common_idx[-1]}",
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": float(eq_arr[-1]),
        "total_return": total_return,
        "max_drawdown": max_dd,
        "total_trades": len(closes),
        "sl_exits": len(sl_closes),
        "signal_exits": len(signal_closes),
        "equity_history": sim.equity_history,
        "trade_log": sim.trade_log,
    }
    out_path = Path("data/results/paper_trading_simulation.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Results saved to: {out_path}")
    print(f"  Time: {elapsed:.0f}s")
    print("=" * 100)


if __name__ == "__main__":
    run_simulation()

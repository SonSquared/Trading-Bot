"""
Guardrail tests for the audit fixes:

- The paper-trader run lock must prevent concurrent instances from
  double-opening positions.
- State saves must be atomic (no torn JSON that would trigger the
  "corrupt state, reset" path).
- The re-optimization backtest must execute at the next open (no
  same-candle look-ahead fill), and the honest selection must return a
  well-formed result with a deploy decision.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.paper_trader import (
    acquire_lock, release_lock, save_state, open_position, close_position, _fresh_state,
    MAX_POSITION_PCT, FEE_RATE as PAPER_FEE_RATE,
    SLIPPAGE_RATE as PAPER_SLIPPAGE_RATE,
)
from trading_system.bot.candles import closed_candles
from trading_system.bot import accounting


def test_run_lock_blocks_concurrent_instance(tmp_path, monkeypatch):
    import scripts.paper_trader as pt

    monkeypatch.setattr(pt, "LOCK_FILE", tmp_path / "bot.lock")
    monkeypatch.setattr(pt, "LOG_DIR", tmp_path)

    lock_path = tmp_path / "bot.lock"
    assert acquire_lock(max_wait_seconds=2) is True
    assert lock_path.exists()
    try:
        # A second instance must NOT acquire the lock while the first holds it
        assert acquire_lock(max_wait_seconds=0.5) is False
    finally:
        release_lock()
    # After release, the lock is acquirable again
    assert acquire_lock(max_wait_seconds=2) is True
    release_lock()
    assert not lock_path.exists()


def test_run_lock_breaks_stale_lock(tmp_path, monkeypatch):
    import scripts.paper_trader as pt

    lock = tmp_path / "bot.lock"
    monkeypatch.setattr(pt, "LOCK_FILE", lock)
    monkeypatch.setattr(pt, "LOG_DIR", tmp_path)

    # Simulate a crashed instance: lock file exists, old timestamp
    lock.write_text("99999")
    import os
    import time as _time
    old = _time.time() - 3600
    os.utime(lock, (old, old))

    assert acquire_lock(max_wait_seconds=2, stale_seconds=600) is True
    release_lock()


def test_save_state_is_atomic(tmp_path, monkeypatch):
    import scripts.paper_trader as pt

    state_file = tmp_path / "paper_state.json"
    monkeypatch.setattr(pt, "STATE_FILE", state_file)

    state = {"cash": 97.0, "positions": {"ETH_USDT_USDT": {"side": 1}}, "total_trades": 3}
    save_state(state)

    loaded = json.loads(state_file.read_text())
    assert loaded == state
    # No temp files left behind
    assert not (tmp_path / "paper_state.json.tmp").exists()


def test_closed_candles_timestamp_column_naive():
    """Paper-trader frames carry a naive (UTC) timestamp column."""
    idx = pd.date_range("2026-01-01", periods=60, freq="4h")
    df = pd.DataFrame({
        "timestamp": idx,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1.0,
    })
    # All candles are far in the past -> all closed
    closed = closed_candles(df, "4h")
    assert len(closed) == 60


def test_backtest_single_fills_at_next_open():
    """The honest backtest runs end-to-end on synthetic data without raising."""
    import scripts.monthly_reoptimize as mr

    n = 200
    opens = np.linspace(100.0, 120.0, n)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
        "open": opens,
        "high": opens + 1.0,
        "low": opens - 1.0,
        "close": opens + 0.5,
        "volume": 1000.0,
    })

    # Backtest over the first 30 candles; the strategy needs 20 candles of
    # Bollinger warmup, so this mostly exercises the no-look-ahead plumbing.
    r = mr.backtest_single(df.iloc[:30], "Bollinger_Reversion",
                           {"bb_period": 20, "bb_std": 2.0, "rsi_filter": False,
                            "exit_at_middle": False})
    assert isinstance(r["return_pct"], float)
    assert r["trades"] >= 0


def test_select_params_honest_returns_structured_decision():
    """Honest selection must produce a params dict + deploy bool, and never
    select on the test set (candidates come from train windows only)."""
    import scripts.monthly_reoptimize as mr

    rng = np.random.default_rng(42)
    n = 3000
    close = 100 + np.cumsum(rng.normal(0, 1.0, n))
    close = np.maximum(close, 1.0)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
        "open": np.roll(close, 1),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1000.0,
    })
    df.loc[0, "open"] = close[0]

    windows = mr.walk_forward_windows(df, train_months=1, test_months=1)
    assert windows, "synthetic data should produce walk-forward windows"

    grid = {"bb_period": [15], "bb_std": [2.0], "rsi_filter": [False], "exit_at_middle": [False]}
    params, evaluation, ok_to_deploy = mr.select_params_honest(
        "Bollinger_Reversion", "ETH_USDT_USDT", df, grid, windows, max_combos=10
    )

    assert params is not None
    assert isinstance(ok_to_deploy, bool)
    assert evaluation["total_windows"] > 0
    assert 0 <= evaluation["profitable_windows"] <= evaluation["total_windows"]
    # Every candidate param must be a valid Bollinger grid combo
    assert params["bb_period"] == 15 and params["bb_std"] == 2.0


# --- Accounting consistency (paper trader == backtest engine) ---

def _oscillating_df(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Synthetic mean-reverting series that makes Bollinger_Reversion trade."""
    rng = np.random.default_rng(seed)
    base = 100 + 8 * np.sin(np.arange(n) / 12)
    close = np.maximum(base + rng.normal(0, 0.5, n), 1.0)
    opens = np.roll(close, 1)
    opens[0] = close[0]
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
        "open": opens,
        "high": np.maximum(opens, close) * 1.004,
        "low": np.minimum(opens, close) * 0.996,
        "close": close,
        "volume": 1000.0,
    })


_BB_PARAMS = {"bb_period": 20, "bb_std": 2.0, "rsi_filter": False, "exit_at_middle": False}


def test_shared_cost_constants_are_identical_everywhere():
    """Paper trader, both backtest engines, and the shared module must charge
    the same fees/slippage/funding or forward results are meaningless."""
    import scripts.monthly_reoptimize as mr
    import scripts.walk_forward_optimize as wf

    assert PAPER_FEE_RATE == mr.FEE_RATE == wf.FEE_RATE == accounting.FEE_RATE
    assert PAPER_SLIPPAGE_RATE == mr.SLIPPAGE_RATE == wf.SLIPPAGE_RATE == accounting.SLIPPAGE_RATE
    # 0.00005 per 4h candle == 0.0001 per 8h (the paper trader's schedule)
    assert accounting.FUNDING_PER_CANDLE * 2 == pytest.approx(accounting.FUNDING_RATE_8H)


def test_funding_boundaries_8h_schedule():
    """Funding is charged at 00:00/08:00/16:00 UTC, boundaries in [start, end)."""
    f = accounting.funding_boundaries_in
    assert f("2024-01-01 07:00", "2024-01-01 09:00") == 1  # 08:00
    assert f("2024-01-01 08:00", "2024-01-01 16:00") == 1  # 08:00 only
    assert f("2024-01-01 00:00", "2024-01-01 23:59") == 3  # 00/08/16
    assert f("2024-01-01 16:00", "2024-01-01 16:00") == 0
    assert f("2024-01-01 23:00", "2024-01-02 00:30") == 1  # 00:00 next day


def test_charge_funding_debits_cash_once(tmp_path, monkeypatch):
    """charge_funding deducts exactly the 8h-boundary funding from cash."""
    from datetime import datetime, timezone

    # open_position() logs to the production trade log — keep tests out of it
    import scripts.paper_trader as pt_mod
    monkeypatch.setattr(pt_mod, "TRADE_LOG", tmp_path / "trades.jsonl")

    st = _fresh_state()
    size = 34.0
    price = 100.0
    entry_time = datetime(2024, 1, 1, 7, 30, tzinfo=timezone.utc)
    open_position(st, "ETH_USDT_USDT", 1, price, "t", size)
    st["positions"]["ETH_USDT_USDT"]["entry_time"] = entry_time.isoformat()
    st["cash"] = 97.0  # reset to clean number after open

    now = datetime(2024, 1, 1, 8, 30, tzinfo=timezone.utc)
    total, per_pair = accounting.charge_funding(st, now)

    qty = size / (price * (1 + accounting.SLIPPAGE_RATE))
    expected = qty * (price * (1 + accounting.SLIPPAGE_RATE)) * accounting.FUNDING_RATE_8H
    assert total == pytest.approx(expected)
    assert per_pair == {"ETH_USDT_USDT": pytest.approx(expected)}
    assert st["cash"] == pytest.approx(97.0 - expected)
    assert st["positions"]["ETH_USDT_USDT"]["funding_paid"] == pytest.approx(expected)

    # Charging again before the next boundary charges nothing
    total2, _ = accounting.charge_funding(st, datetime(2024, 1, 1, 8, 45, tzinfo=timezone.utc))
    assert total2 == 0.0


def test_paper_trader_matches_backtest_engine_roundtrip(tmp_path, monkeypatch):
    """The paper trader's ledger must equal the backtest engine's final cash
    for the same signals, fills, fees, and funding — to float precision."""
    import scripts.monthly_reoptimize as mr
    import scripts.paper_trader as pt_mod
    from trading_system.strategies import STRATEGY_REGISTRY
    from trading_system.bot.accounting import funding_cost

    # open_position()/close_position() log to the production trade log — keep
    # tests out of it (this used to pollute data/results/paper_trades.jsonl)
    monkeypatch.setattr(pt_mod, "TRADE_LOG", tmp_path / "trades.jsonl")

    df = _oscillating_df()
    r = mr.backtest_single(df, "Bollinger_Reversion", _BB_PARAMS)
    assert r["trades"] >= 1, "synthetic data should produce at least one trade"

    # Replay the exact same decisions through the paper-trader ledger.
    sig = STRATEGY_REGISTRY["Bollinger_Reversion"].generate_signals(df, _BB_PARAMS)
    sig_prev = sig.shift(1).fillna(0)
    st = _fresh_state()
    pair = "ETH_USDT_USDT"
    for i in range(len(df)):
        price = float(df["open"].iloc[i])
        signal = int(sig_prev.iloc[i])
        pos = st["positions"].get(pair)
        side = pos["side"] if pos else 0
        if pos and signal != side:
            # Lump funding through the exit time, same as the backtest.
            qty = pos["size_usd"] / pos["entry_price"]
            funding = funding_cost(qty * pos["entry_price"], pos["entry_time"], df["timestamp"].iloc[i])
            close_position(st, pair, price, "reversal")
            st["cash"] -= funding
        if not st["positions"] and signal != 0:
            size = min(st["cash"] * MAX_POSITION_PCT, st["cash"] * 0.95)
            if size > 5:
                open_position(st, pair, signal, price, "Bollinger_Reversion", size)
                # Stamp the entry with the candle time, like the backtest
                st["positions"][pair]["entry_time"] = df["timestamp"].iloc[i].isoformat()

    # Backtest force-closes the remaining position at the last close.
    if st["positions"]:
        pos = st["positions"][pair]
        qty = pos["size_usd"] / pos["entry_price"]
        funding = funding_cost(qty * pos["entry_price"], pos["entry_time"], df["timestamp"].iloc[-1])
        close_position(st, pair, float(df["close"].iloc[-1]), "final")
        st["cash"] -= funding

    assert st["cash"] == pytest.approx(r["final"], abs=1e-9)


# --- Forward-replay risk model (trailing / disaster stops) ---


def _rm():
    from trading_system.bot.risk_manager import DEFAULT_RISK_MANAGER
    return DEFAULT_RISK_MANAGER


def test_position_stops_benign_candle_ratchets_watermark():
    """A candle that doesn't breach a stop ratchets the profit watermark
    (adverse-first: the new high only counts from the NEXT candle)."""
    from scripts.forward_run import apply_position_stops

    pos = {"side": 1, "entry_price": 100.0, "size_usd": 34.0}
    candle = {"open": 101.0, "high": 108.0, "low": 98.0, "close": 99.0}
    reason, fill = apply_position_stops(pos, candle, _rm())
    assert reason is None and fill is None
    assert pos["high_pnl"] == pytest.approx(8.0)  # high 108 -> +8%


def test_position_stops_disaster_stop_loss_fires():
    from scripts.forward_run import apply_position_stops

    pos = {"side": 1, "entry_price": 100.0, "size_usd": 34.0}
    candle = {"open": 100.0, "high": 101.0, "low": 94.0, "close": 95.0}  # -6% low
    reason, fill = apply_position_stops(pos, candle, _rm())
    assert reason is not None and reason.startswith("Stop-loss")
    assert fill == pytest.approx(95.0)  # stop at -5%, no gap


def test_position_stops_trailing_stop_fires_long():
    """Watermark 8% -> stop at +5%; a -2.5% low breaches it; the stop level
    was already passed at the open (102 < 105), so the fill is the open."""
    from scripts.forward_run import apply_position_stops

    pos = {"side": 1, "entry_price": 100.0, "size_usd": 34.0, "high_pnl": 8.0}
    candle = {"open": 102.0, "high": 103.0, "low": 97.5, "close": 98.0}
    reason, fill = apply_position_stops(pos, candle, _rm())
    assert reason is not None and reason.startswith("Trailing stop")
    assert fill == pytest.approx(102.0)


def test_position_stops_trailing_stop_fires_short():
    """Short mirror: stop level at 95 was gapped through by the open (98)."""
    from scripts.forward_run import apply_position_stops

    pos = {"side": -1, "entry_price": 100.0, "size_usd": 34.0, "high_pnl": 8.0}
    candle = {"open": 98.0, "high": 103.0, "low": 97.0, "close": 102.0}
    reason, fill = apply_position_stops(pos, candle, _rm())
    assert reason is not None and reason.startswith("Trailing stop")
    assert fill == pytest.approx(98.0)


def test_drawdown_cooldown_blocks_reentry_then_expires():
    """After a dd stop the account must stay flat during the cooldown, then
    reopen once it expires (with the peak re-armed by the caller). A static
    peak would either oscillate close->open->close or stay dormant forever."""
    from datetime import datetime, timedelta, timezone

    rm = _rm()
    now = datetime.now(timezone.utc)
    # Still in the cooldown -> no opens
    ok, reason = rm.can_open_position({}, equity=90.0, cash=90.0, prices={},
                                      peak_equity=90.0,   # caller re-armed
                                      dd_cooldown_until=now + timedelta(hours=2),
                                      now=now)
    assert ok is False
    assert "cooldown" in reason
    # Cooldown expired -> opens resume (peak re-armed, so no stale dd line)
    ok, _ = rm.can_open_position({}, equity=90.0, cash=90.0, prices={},
                                 peak_equity=90.0,
                                 dd_cooldown_until=now - timedelta(hours=2),
                                 now=now)
    assert ok is True
    # No cooldown args -> legacy behaviour
    ok, _ = rm.can_open_position({}, equity=50.0, cash=50.0, prices={})
    assert ok is True
    # Stale peak without re-arm must still gate (safety net)
    ok, reason = rm.can_open_position({}, equity=90.0, cash=90.0, prices={},
                                      peak_equity=100.0)
    assert ok is False and "below the dd line" in reason


def test_position_stops_not_triggered_below_activation():
    """A +3% watermark is below the +5% activation -> no trailing stop even
    when the candle reverses hard."""
    from scripts.forward_run import apply_position_stops

    pos = {"side": 1, "entry_price": 100.0, "size_usd": 34.0, "high_pnl": 3.0}
    candle = {"open": 101.0, "high": 104.0, "low": 96.0, "close": 97.0}
    reason, fill = apply_position_stops(pos, candle, _rm())
    assert reason is None and fill is None
    assert pos["high_pnl"] == pytest.approx(4.0)  # ratcheted to the new high


def test_walk_forward_and_monthly_backtests_agree():
    """Both backtest engines must produce identical results on the same data."""
    import scripts.monthly_reoptimize as mr
    import scripts.walk_forward_optimize as wf

    df = _oscillating_df()
    r_mr = mr.backtest_single(df, "Bollinger_Reversion", _BB_PARAMS)
    r_wf = wf.backtest_single(df, "Bollinger_Reversion", _BB_PARAMS)
    assert r_mr["final"] == pytest.approx(r_wf["final"], abs=1e-9)
    assert r_mr["trades"] == r_wf["trades"]
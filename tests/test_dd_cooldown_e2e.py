"""
End-to-end test proving the paper bot stays flat for the full dd cooldown after
a drawdown stop and resumes trading only after the peak re-arm.

Scenario (synthetic 4h candles, deterministic — no randomness):

  candles 0-11   ramp +2%          -> bot opens a LONG at candle 1
  candle  12     crash -35% close  -> portfolio equity drops ~12% below peak,
                                      the 10% PORTFOLIO DRAWDOWN STOP fires
  candles 13-18  flat bottom       -> the 24h cooldown (6 x 4h candles);
                                      bot must stay FLAT
  candles 19-28  recovery +12%     -> bot may reopen after cooldown
  candles 29-44  flat              -> ride / time-stop churn

Why the dd stop (not the 5% position SL) fires: _replay evaluates the
portfolio drawdown stop BEFORE per-position stops at each candle boundary,
and the dd check uses the candle CLOSE. A single deep crash candle pushes
equity through the dd line at the close, so the dd branch runs and the
position stops never get evaluated on that candle.

Why the crash is -35%: the replay sizes at
``min(equity * 0.35, cash * 0.95, INITIAL_CAPITAL * 0.50)`` ~= $34 on a $97
account. For a 10% portfolio drawdown (~$9.8), the position must lose ~29%.
-35% gives comfortable margin.

How the test distinguishes peak RE-ARM from a static-peak account: with the
peak re-armed to post-stop equity, the cooldown is the ONLY gate, so the bot
reopens at the first boundary after the 24h cooldown expires (gap 24-28h).
With a static pre-crash peak, equity (~$86) would stay >10% below the old
peak (~$98) until the recovery lifts it, delaying any reopen to 40h+.
The test asserts 24h <= gap < 40h.

The two unit tests pin the risk_manager.can_open_position gating directly
(cooldown blocks; re-armed peak allows; static peak blocks forever).
"""

import json
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")

import pandas as pd

from trading_system.bot.risk_manager import RiskManager
from scripts.forward_run import _replay

PAIR = "ETH_USDT_USDT"
CRASH_IDX = 12          # crash candle index
CRASH_PCT = 0.35        # -35% close on the crash candle
COOLDOWN_H = 24         # dd_cooldown_hours in DEFAULT_RISK_MANAGER
CANDLE_H = 4


def _always_long_registry():
    """Register a trivial always-LONG strategy so the replay trades it."""
    import trading_system.strategies as st

    class AlwaysLong:
        name = "AlwaysLong"

        @staticmethod
        def generate_signals(df, params):
            return pd.Series(1, index=df.index)

    st.STRATEGY_REGISTRY["AlwaysLong"] = AlwaysLong()


def synthetic_crash_then_recover() -> pd.DataFrame:
    """Deterministic 4h OHLC series: ramp, crash, flat bottom, recovery, flat."""
    n_candles = 45
    start_price = 2000.0
    recover_pct = 0.12

    timestamps = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=CANDLE_H * i)
                  for i in range(n_candles)]

    # Phase 1 (candles 0..CRASH_IDX-1): deterministic +2% ramp.
    prices = [start_price * (1 + 0.02 * (i + 1) / CRASH_IDX) for i in range(CRASH_IDX)]
    crash_start = prices[-1]

    # Phase 2: the crash candle close (recorded in prices so later phases
    # start from the right level; the full OHLC row is built below).
    crash_close = crash_start * (1 - CRASH_PCT)
    prices.append(crash_close)

    # Phase 3: flat bottom for exactly the cooldown window (6 candles = 24h).
    bottom = crash_close
    prices.extend([bottom] * 6)

    # Phase 4: recovery +12% over 10 candles.
    recover_start = prices[-1]
    for i in range(10):
        prices.append(recover_start * (1 + recover_pct * (i + 1) / 10.0))

    # Phase 5: flat to the end.
    prices.extend([prices[-1]] * (n_candles - len(prices)))
    assert len(prices) == n_candles

    # Sanity: a -35% crash must be deep enough for the ~$34 position to
    # lose >= 10% of the ~$97 account at the crash close.
    position_usd = 97.0 * 0.35
    loss_frac = position_usd * CRASH_PCT / 97.0
    assert loss_frac >= 0.10, (
        f"crash too shallow: position loses {loss_frac:.1%} of equity, need >= 10%")

    rows = []
    for idx, ts in enumerate(timestamps):
        if idx == CRASH_IDX:
            rows.append({
                "timestamp": ts,
                "open": crash_start,
                "high": crash_start * 1.001,
                "low": crash_close * 0.99,   # below close; irrelevant: dd fires first
                "close": crash_close,
                "volume": 5_000_000.0,
            })
        else:
            px = prices[idx]
            rows.append({
                "timestamp": ts,
                "open": px,
                "high": px * 1.0005,
                "low": px * 0.9995,
                "close": px,
                "volume": 1_000_000.0,
            })
    return pd.DataFrame(rows)


def _run_replay_with_always_long(df: pd.DataFrame) -> dict:
    """Run _replay with the AlwaysLong strategy as the sole optimized config.

    _replay calls load_active_strategies(), which prefers
    bot_strategy_params.json over defaults, so we write a temp params file
    and restore the original afterwards.
    """
    from scripts.paper_trader import OPTIMIZED_PARAMS_FILE

    _always_long_registry()

    original = None
    if OPTIMIZED_PARAMS_FILE.exists():
        original = OPTIMIZED_PARAMS_FILE.read_text()
        OPTIMIZED_PARAMS_FILE.unlink()
    try:
        cfg = {
            "TestLong": {
                "strategy": "AlwaysLong",
                "pair": PAIR,
                "timeframe": "4h",
                "weight": 1.0,
                "params": {},
            }
        }
        OPTIMIZED_PARAMS_FILE.write_text(json.dumps(cfg))
        return _replay({PAIR: df}, label="dd cooldown e2e")
    finally:
        if original is not None:
            OPTIMIZED_PARAMS_FILE.write_text(original)
        else:
            OPTIMIZED_PARAMS_FILE.unlink(missing_ok=True)


def _trade_times(t: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Candle-time entry/exit for a replay trade record.

    _replay stamps ``_entry_time``/``_exit_time`` with real candle times and
    leaves the wall-clock ``entry_time``/``exit_time`` untouched.
    """
    entry = t.get("_entry_time") or t.get("entry_time", "")
    exit_ = t.get("_exit_time") or t.get("exit_time", "")
    return pd.Timestamp(entry), pd.Timestamp(exit_)


def test_dd_cooldown_stays_flat():
    """E2E: dd stop closes everything, bot stays flat >= 24h, then reopens
    promptly after the cooldown (proving the peak was re-armed)."""
    df = synthetic_crash_then_recover()
    result = _run_replay_with_always_long(df)

    trades = result["trades_detail"]

    print("=== DD Cooldown E2E ===")
    for t in sorted(trades, key=lambda x: _trade_times(x)[1]):
        entry, exit_ = _trade_times(t)
        print(f"  {entry.isoformat()} -> {exit_.isoformat()}  "
              f"pnl={t.get('pnl_usd', 0):+.2f}  {t.get('reason', '')}")
    print(f"  final capital ${result['capital']['final']:.2f} "
          f"({result['capital']['return_pct']:+.2f}%), "
          f"max DD {result['risk']['max_drawdown_pct']:.2f}%")

    # 1. The portfolio drawdown stop must have fired.
    dd_trades = [t for t in trades
                 if t.get("reason", "").startswith("Portfolio drawdown")]
    assert dd_trades, "expected a portfolio drawdown stop; crash was too shallow " \
                      f"or the position never opened (trades={len(trades)})"
    dd_entry, dd_exit = _trade_times(dd_trades[0])
    print(f"  dd stop closed at {dd_exit.isoformat()} "
          f"(position opened {dd_entry.isoformat()})")

    # The dd-closed position must have been opened BEFORE the crash candle.
    assert dd_entry < pd.Timestamp(df["timestamp"].iloc[CRASH_IDX])

    # 2. No new position may open during the 24h cooldown.
    cooldown_end = dd_exit + pd.Timedelta(hours=COOLDOWN_H)
    during = [t for t in trades if dd_exit < _trade_times(t)[0] < cooldown_end]
    assert not during, \
        f"bot opened {len(during)} position(s) during the {COOLDOWN_H}h cooldown"

    # 3. The bot must resume trading after the cooldown (not permanently
    #    dormant) — and promptly, which only happens if the peak was re-armed.
    reopens = [t for t in trades if _trade_times(t)[0] >= cooldown_end]
    assert reopens, "bot never resumed trading after the dd stop (dormancy bug)"
    first_reopen = min(_trade_times(t)[0] for t in reopens)
    gap_h = (first_reopen - dd_exit).total_seconds() / 3600.0
    print(f"  first reopen at {first_reopen.isoformat()} "
          f"({gap_h:.1f}h after the dd stop)")
    assert gap_h >= COOLDOWN_H, \
        f"bot reopened after only {gap_h:.1f}h < {COOLDOWN_H}h cooldown"
    assert gap_h < 40, (
        f"bot reopened after {gap_h:.1f}h — too late for a re-armed peak "
        f"(a static pre-crash peak would delay the reopen past the recovery)")


def test_can_open_drawdown_gating():
    """Unit: can_open_position blocks during the cooldown even with empty
    positions and plenty of cash, and allows once it expires."""
    risk = RiskManager(
        position_stop_loss_pct=5.0,
        trailing_stop_activation_pct=5.0,
        trailing_stop_distance_pct=3.0,
        max_position_hours=72.0,
        portfolio_max_dd_pct=10.0,
        dd_cooldown_hours=24.0,
        max_open_positions=5,
        max_portfolio_heat_pct=50.0,
    )

    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    cooldown_until = now + timedelta(hours=12)

    can_open, reason = risk.can_open_position(
        positions={}, equity=200.0, cash=200.0,
        prices={PAIR: 2000.0}, peak_equity=200.0,
        dd_cooldown_until=cooldown_until.isoformat(), now=now,
    )
    print(f"  during cooldown: can_open={can_open} ({reason})")
    assert not can_open
    assert "cooldown" in reason.lower()

    can_open2, reason2 = risk.can_open_position(
        positions={}, equity=200.0, cash=200.0,
        prices={PAIR: 2000.0}, peak_equity=200.0,
        dd_cooldown_until=cooldown_until.isoformat(),
        now=now + timedelta(hours=24),
    )
    print(f"  after cooldown:  can_open={can_open2} ({reason2})")
    assert can_open2, reason2


def test_peak_rearm():
    """Unit: after the dd stop the peak must be re-armed to post-stop equity,
    or a flat cash account can never clear the old dd line (permanent
    dormancy — the exact bug this fix removes)."""
    risk = RiskManager(
        position_stop_loss_pct=5.0,
        portfolio_max_dd_pct=10.0,
        dd_cooldown_hours=24.0,
    )

    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    cooldown_until = now + timedelta(hours=24)

    # During cooldown: blocked even with the re-armed peak.
    can_open, _ = risk.can_open_position(
        positions={}, equity=170.0, cash=170.0,
        prices={PAIR: 1700.0}, peak_equity=170.0,
        dd_cooldown_until=cooldown_until.isoformat(), now=now,
    )
    assert not can_open

    # After cooldown, re-armed peak (170) with equity at 170: dd = 0% -> allowed.
    can_open2, reason2 = risk.can_open_position(
        positions={}, equity=170.0, cash=170.0,
        prices={PAIR: 1700.0}, peak_equity=170.0,
        dd_cooldown_until=cooldown_until.isoformat(),
        now=now + timedelta(hours=48),
    )
    print(f"  re-armed peak after cooldown: can_open={can_open2} ({reason2})")
    assert can_open2, reason2

    # Contrast: static pre-crash peak (200) with equity 170 -> dd 15% ->
    # blocked forever. This is the bug.
    can_open3, reason3 = risk.can_open_position(
        positions={}, equity=170.0, cash=170.0,
        prices={PAIR: 1700.0}, peak_equity=200.0,
        dd_cooldown_until=None,
        now=now + timedelta(hours=48),
    )
    print(f"  static peak after cooldown:   can_open={can_open3} ({reason3})")
    assert not can_open3, "static peak should keep the account below the dd line"


if __name__ == "__main__":
    test_dd_cooldown_stays_flat()
    print("=" * 60)
    test_can_open_drawdown_gating()
    print("=" * 60)
    test_peak_rearm()
    print("=" * 60)
    print("ALL TESTS PASS")

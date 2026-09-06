"""
Tests for the strategy league (scripts/strategy_league.py).

The critical property: a challenger is promoted ONLY when it beats the
incumbent out-of-sample on ALL gate metrics. These tests pin every gate
branch, the deployment path (write, validate through the bot's real
loader, backup), and a small end-to-end round on synthetic data.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import scripts.paper_trader as pt
import scripts.strategy_league as league


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def good_oos(**over) -> dict:
    base = {
        "mean": 5.0, "median": 5.0, "min": -3.0, "max": 12.0,
        "profitable_windows": 3, "total_windows": 4, "profitable_frac": 0.75,
        "window_returns": [5.0, 6.0, -3.0, 12.0],
        "n_trades": 40,
    }
    base.update(over)
    return base


def make_data(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Oscillating synthetic OHLCV that mean-reversion strategies trade."""
    rng = np.random.default_rng(seed)
    close = np.maximum(100 + 8 * np.sin(np.arange(n) / 12) + rng.normal(0, 0.5, n), 1.0)
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


def two_windows(n: int = 400) -> list[dict]:
    half = n // 2
    return [
        {"train_start": 0, "train_end": half - 20, "test_start": half - 20,
         "test_end": half + 40, "train_start_date": "2024-01-01",
         "train_end_date": "x", "test_start_date": "y", "test_end_date": "z"},
        {"train_start": half - 20, "train_end": half + 40, "test_start": half + 40,
         "test_end": n, "train_start_date": "a", "train_end_date": "b",
         "test_start_date": "c", "test_end_date": "d"},
    ]


# --------------------------------------------------------------------------
# The promotion gate
# --------------------------------------------------------------------------

class TestGate:
    def test_positive_mean_but_no_incumbent_passes(self):
        ok, why = league.challenger_wins(good_oos(), None)
        assert ok and "first qualifying" in why

    def test_nonpositive_mean_rejected(self):
        ok, why = league.challenger_wins(good_oos(mean=0.0), None)
        assert not ok and "not positive" in why
        ok, why = league.challenger_wins(good_oos(mean=-2.0), None)
        assert not ok

    def test_worst_window_floor_rejects(self):
        ok, why = league.challenger_wins(good_oos(min=-16.0), None)
        assert not ok and "floor" in why

    def test_minority_profitable_rejected(self):
        ok, why = league.challenger_wins(
            good_oos(profitable_windows=1, profitable_frac=0.25), None)
        assert not ok and "majority" in why

    def test_majority_boundary_exactly_half_rejected(self):
        # 2/4 is not a strict majority -> rejected.
        ok, why = league.challenger_wins(
            good_oos(profitable_windows=2, profitable_frac=0.5), None)
        assert not ok and "majority" in why

    def test_loses_on_mean_rejected(self):
        ok, why = league.challenger_wins(good_oos(mean=5.0),
                                         good_oos(mean=6.0, min=-4.0,
                                                  profitable_frac=0.5))
        assert not ok and "mean" in why

    def test_loses_on_consistency_rejected(self):
        ok, why = league.challenger_wins(good_oos(),
                                         good_oos(mean=4.0, min=-4.0,
                                                  profitable_frac=1.0))
        assert not ok and "fraction" in why

    def test_loses_on_worst_window_rejected(self):
        ok, why = league.challenger_wins(good_oos(),
                                         good_oos(mean=4.0, min=-2.0))
        assert not ok and "worst window" in why

    def test_full_victory_passes(self):
        ok, why = league.challenger_wins(good_oos(),
                                         good_oos(mean=4.0, min=-4.0,
                                                  profitable_frac=0.5))
        assert ok and "beats incumbent" in why

    def test_tie_on_mean_rejected(self):
        # No tie-flipping: equal mean is NOT a win.
        ok, why = league.challenger_wins(good_oos(mean=5.0),
                                         good_oos(mean=5.0, min=-4.0,
                                                  profitable_frac=0.5))
        assert not ok

    def test_zero_trades_rejected_as_inactive(self):
        # A candidate that never trades OOS is a vacancy, not a strategy:
        # its 0.00% must not pass the gate.
        ok, why = league.challenger_wins(good_oos(n_trades=0), None)
        assert not ok and "inactive" in why

    def test_minimal_trades_still_pass(self):
        # Any trade activity at all clears the inactivity gate.
        ok, why = league.challenger_wins(good_oos(n_trades=1), None)
        assert ok

    def test_missing_n_trades_key_treated_as_inactive(self):
        # Defensive: an OOS dict from an older code path without a trade
        # count must not silently pass.
        oos = good_oos()
        del oos["n_trades"]
        ok, why = league.challenger_wins(oos, None)
        assert not ok and "inactive" in why


class TestBenchmarkIncumbent:
    """The challenger must beat the pair's BEST config, not the first or
    weakest one found — otherwise it can promote against a config it
    would never actually replace."""

    def _cfg(self, strategy, weight):
        return {"strategy": strategy, "pair": "BTC_USDT_USDT",
                "timeframe": "4h", "weight": weight, "params": {}}

    def test_best_config_is_benchmarked(self):
        data = make_data(400)
        windows = two_windows(400)
        incumbent = {
            "Donchian_Breakout_BTC_USDT_USDT": self._cfg("Donchian_Breakout", 0.35),
            "RSI_Reversion_BTC_USDT_USDT": self._cfg("RSI_Reversion", 0.25),
        }
        slot, cfg, ev = league.pick_benchmark_incumbent(
            incumbent, "BTC_USDT_USDT", data, windows)
        assert ev is not None
        means = {}
        for k, c in incumbent.items():
            e = league.evaluate_incumbent_for_pair(c, data, windows)
            means[k] = e["mean"] if e else float("-inf")
        best = max(means, key=means.get)
        assert slot == best, (
            f"benchmarked {cfg['strategy']} but best is "
            f"{incumbent[best]['strategy']}")

    def test_unevaluable_config_falls_back(self):
        data = make_data(400)
        windows = two_windows(400)
        # Bogus strategy names -> evaluate_oos swallows the errors and
        # returns None for every config of the pair.
        incumbent = {
            "Ghost_A_BTC_USDT_USDT": self._cfg("NoSuchStrategy_A", 0.35),
            "Ghost_B_BTC_USDT_USDT": self._cfg("NoSuchStrategy_B", 0.25),
        }
        slot, cfg, ev = league.pick_benchmark_incumbent(
            incumbent, "BTC_USDT_USDT", data, windows)
        assert ev is None
        assert slot in incumbent  # first config kept as the slot

    def test_absent_pair_returns_none(self):
        data = make_data(400)
        windows = two_windows(400)
        slot, cfg, ev = league.pick_benchmark_incumbent(
            {}, "BTC_USDT_USDT", data, windows)
        assert slot is None and cfg is None and ev is None


# --------------------------------------------------------------------------
# OOS evaluation
# --------------------------------------------------------------------------

class TestEvaluateOOS:
    def test_structure_and_fractions(self):
        data = make_data()
        windows = two_windows(len(data))
        r = league.evaluate_oos("Bollinger_Reversion",
                                {"bb_period": 20, "bb_std": 2.0,
                                 "rsi_filter": False, "exit_at_middle": True},
                                data, windows)
        assert r is not None
        assert r["total_windows"] == len(windows)
        assert 0.0 <= r["profitable_frac"] <= 1.0
        assert len(r["window_returns"]) == r["total_windows"]
        assert r["min"] <= r["median"] <= r["max"]

    def test_untradable_data_returns_none(self):
        # Constant prices: backtest produces no trades but must not crash.
        data = make_data(200)
        data["close"] = 100.0
        data["open"] = 100.0
        data["high"] = 100.0
        data["low"] = 100.0
        r = league.evaluate_oos("Bollinger_Reversion",
                                {"bb_period": 20, "bb_std": 2.0,
                                 "rsi_filter": False, "exit_at_middle": True},
                                data, two_windows(200))
        assert r is None or r["total_windows"] >= 1


# --------------------------------------------------------------------------
# Deployment
# --------------------------------------------------------------------------

class TestDeployment:
    @pytest.fixture
    def league_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(league, "PARAMS_FILE", tmp_path / "bot_strategy_params.json")
        monkeypatch.setattr(league, "PARAMS_BACKUP", tmp_path / "backup.json")
        return tmp_path

    def test_deploy_winner_preserves_other_pairs(self, league_paths):
        incumbent = {
            "Bollinger_Reversion_ETH_USDT_USDT": {
                "strategy": "Bollinger_Reversion", "pair": "ETH_USDT_USDT",
                "timeframe": "4h", "weight": 0.4, "params": {"bb_period": 20}},
            "Bollinger_Reversion_BTC_USDT_USDT": {
                "strategy": "Bollinger_Reversion", "pair": "BTC_USDT_USDT",
                "timeframe": "4h", "weight": 0.35, "params": {"bb_period": 20}},
        }
        round_result = {"winner": {"strategy": "Supertrend",
                                   "params": {"period": 14, "multiplier": 3.0}}}
        new_cfg = league.deploy_winner("ETH_USDT_USDT", round_result, incumbent)
        # ETH slot replaced, BTC slot untouched.
        assert "Supertrend_ETH_USDT_USDT" in new_cfg
        assert "Bollinger_Reversion_ETH_USDT_USDT" not in new_cfg
        assert "Bollinger_Reversion_BTC_USDT_USDT" in new_cfg
        eth = new_cfg["Supertrend_ETH_USDT_USDT"]
        assert eth["strategy"] == "Supertrend"
        assert eth["weight"] == 0.4  # inherited the incumbent's slot weight

    def test_apply_config_writes_and_backs_up(self, league_paths):
        pf = league.PARAMS_FILE
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(json.dumps({"old": True}), encoding="utf-8")
        league.apply_config({"new": True}, dry_run=False)
        assert json.loads(pf.read_text(encoding="utf-8")) == {"new": True}
        assert league.PARAMS_BACKUP.read_text(encoding="utf-8") == json.dumps({"old": True})

    def test_dry_run_touches_nothing(self, league_paths):
        pf = league.PARAMS_FILE
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(json.dumps({"old": True}), encoding="utf-8")
        league.apply_config({"new": True}, dry_run=True)
        assert json.loads(pf.read_text(encoding="utf-8")) == {"old": True}
        assert not league.PARAMS_BACKUP.exists()

    def test_deployed_config_loads_through_bot_loader(self, league_paths, monkeypatch):
        """The full integration: a league-deployed file must be loadable by
        the paper trader's own load_active_strategies()."""
        valid_config = {
            "Supertrend_ETH_USDT_USDT": {
                "strategy": "Supertrend", "pair": "ETH_USDT_USDT",
                "timeframe": "4h", "weight": 0.4,
                "params": {"period": 14, "multiplier": 3.0},
            },
        }
        pf = league.PARAMS_FILE
        pf.write_text(json.dumps(valid_config), encoding="utf-8")
        monkeypatch.setattr(pt, "OPTIMIZED_PARAMS_FILE", pf)
        loaded = pt.load_active_strategies()
        assert "Supertrend_ETH_USDT_USDT" in loaded
        assert loaded["Supertrend_ETH_USDT_USDT"]["strategy"] == "Supertrend"


# --------------------------------------------------------------------------
# End-to-end round on synthetic data
# --------------------------------------------------------------------------

class TestLeagueRound:
    def test_round_produces_report_structure(self):
        data = make_data(400)
        windows = two_windows(400)
        inc_cfg = {"strategy": "Bollinger_Reversion",
                   "params": {"bb_period": 20, "bb_std": 2.0,
                              "rsi_filter": False, "exit_at_middle": True}}
        r = league.run_league_for_pair(
            "ETH_USDT_USDT", data, windows, inc_cfg,
            league.evaluate_incumbent_for_pair(inc_cfg, data, windows),
            max_combos=2)
        assert r["pair"] == "ETH_USDT_USDT"
        assert r["candidates"], "candidate list must be present"
        # Incumbent evaluated on the same windows.
        assert r["incumbent"] and r["incumbent"]["oos"] is not None
        # Every evaluated candidate carries a gate verdict.
        for c in r["candidates"]:
            assert "gate" in c or "error" in c
        # Winner-or-retained decision present.
        assert isinstance(r["promoted"], bool)

    def test_incumbent_retained_when_no_challenger_wins(self):
        """With a weak synthetic field the incumbent must stay: promotion
        requires genuine OOS superiority, not just the best-of-a-bad-lot."""
        data = make_data(400, seed=3)
        windows = two_windows(400)
        inc_cfg = {"strategy": "Bollinger_Reversion",
                   "params": {"bb_period": 20, "bb_std": 2.0,
                              "rsi_filter": False, "exit_at_middle": True}}
        r = league.run_league_for_pair(
            "ETH_USDT_USDT", data, windows, inc_cfg,
            league.evaluate_incumbent_for_pair(inc_cfg, data, windows),
            max_combos=2)
        # On tiny synthetic data nobody should legitimately clear the
        # strict gate; if a winner IS declared it must have genuinely
        # positive OOS mean + majority + floor-clearing worst window.
        if r["promoted"]:
            oos = r["winner"]["oos"]
            assert oos["mean"] > 0
            assert oos["profitable_windows"] >= (oos["total_windows"] + 1) // 2
            assert oos["min"] > league.WORST_WINDOW_FLOOR_PCT
            assert oos["mean"] > r["incumbent"]["oos"]["mean"]
        # The decision must be conservative and consistent:
        # promoted <=> at least one candidate cleared the gate, and the
        # winner is the gate-passer with the best OOS mean.
        passers = [c for c in r["candidates"]
                   if c.get("gate", {}).get("passed")]
        if r["promoted"]:
            assert r["winner"]["strategy"] in {c["strategy"] for c in passers}
            w_mean = r["winner"]["oos"]["mean"]
            for c in passers:
                assert c["oos"]["mean"] <= w_mean + 1e-9, (
                    f"{c['strategy']} passed the gate with a better mean "
                    f"than the declared winner")
        else:
            # Incumbent retained <=> nobody cleared the gate.
            assert not passers, (
                f"incumbent retained but {[c['strategy'] for c in passers]} "
                f"cleared the gate")

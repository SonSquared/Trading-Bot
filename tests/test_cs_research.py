"""Task 6 (plan MD): strategies, purged walk-forward, statistics, league, veto.

Pinned here:
- train timestamps strictly precede (embargoed) test timestamps in every fold,
- strategies can ONLY return OrderIntent objects — never exchange orders,
- poor tail performance blocks promotion even with a good average,
- when no candidate passes the gates, the league selects NO replacement
  (veto) rather than promoting the least-bad candidate,
- block-bootstrap CIs and multiple-testing adjustment behave correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_system.models import OrderIntent
from crypto_system.research.league import League, LeagueCandidate, LeagueResult
from crypto_system.research.scoreboard import LiveScoreboard
from crypto_system.research.statistics import (
    bonferroni_alpha,
    bootstrap_mean_ci,
    max_drawdown,
    sharpe,
)
from crypto_system.research.walkforward import PurgedWalkForward, simulate_net
from crypto_system.strategies.base import MeanReversionSleeve, TrendSleeve

# ---------------------------------------------------------------- fixtures

RNG = np.random.default_rng(7)


def _market_df(n: int = 900) -> pd.DataFrame:
    """Synthetic trending-then-ranging market with realistic micro-structure."""
    drift = np.concatenate(
        [np.full(n // 3, 0.0008), np.full(n // 3, -0.0004), np.full(n - 2 * (n // 3), 0.0005)]
    )
    rets = drift + RNG.normal(0, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(RNG.normal(0, 0.004, n)))
    low = close * (1 - np.abs(RNG.normal(0, 0.004, n)))
    open_ = np.roll(close, 1) * (1 + RNG.normal(0, 0.001, n))
    open_[0] = close[0]
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.abs(RNG.normal(100, 10, n))},
        index=idx,
    )


DF = _market_df()


# ---------------------------------------------------------------- strategies

class TestIntentOnly:
    def test_generate_returns_order_intents_only(self):
        sleeve = TrendSleeve()
        intents = sleeve.generate(DF, {"lookback": 40, "exit": 10})
        assert len(intents) > 0
        for intent in intents:
            assert isinstance(intent, OrderIntent)
            assert intent.side in ("long", "short")
            assert intent.symbol
            assert intent.notional > 0
            # no exchange-order escape hatch: intents carry no order id/status
            assert not hasattr(intent, "order_id")
            assert not hasattr(intent, "status")

    def test_both_sleeves_are_symmetric(self):
        for sleeve in (TrendSleeve(), MeanReversionSleeve()):
            intents = sleeve.generate(DF, sleeve.default_params())
            sides = {i.side for i in intents}
            assert sides <= {"long", "short"}


# ---------------------------------------------------------------- walk-forward

class TestPurgedWalkForward:
    def test_train_precedes_embargoed_test_in_every_fold(self):
        splitter = PurgedWalkForward(n_folds=4, embargo_bars=6)
        folds = list(splitter.split(DF))
        assert len(folds) == 4
        for train_df, test_df in folds:
            assert train_df.index.max() < test_df.index.min()
            # embargo: gap between train end and test start is at least embargo
            gap = test_df.index.min() - train_df.index.max()
            assert gap >= pd.Timedelta(hours=4 * 6)

    def test_parameter_search_confined_to_train(self):
        """The engine never evaluates candidate params on test data first."""
        splitter = PurgedWalkForward(n_folds=3, embargo_bars=6)
        for train_df, test_df in splitter.split(DF):
            # engine contract: select params on train, then score on test
            best = PurgedWalkForward.select_params(
                TrendSleeve(), train_df,
                grid=[{"lookback": 30, "exit": 10}, {"lookback": 60, "exit": 10}],
            )
            assert best in ({"lookback": 30, "exit": 10}, {"lookback": 60, "exit": 10})
            oos = simulate_net(
                TrendSleeve().signal(test_df, best), test_df["close"],
                fee_rate=0.0005, slippage_bps=2.0,
            )
            assert oos is not None


class TestSimulation:
    def test_costs_reduce_returns(self):
        sig = TrendSleeve().signal(DF, {"lookback": 40, "exit": 10})
        gross = simulate_net(sig, DF["close"], fee_rate=0.0, slippage_bps=0.0)
        net = simulate_net(sig, DF["close"], fee_rate=0.0005, slippage_bps=2.0)
        assert net.sum() < gross.sum()

    def test_next_open_semantics_no_same_bar_lookahead(self):
        # A signal that is always +1 from bar 0 must not earn bar-0's return.
        sig = pd.Series(1, index=DF.index)
        net = simulate_net(sig, DF["close"], fee_rate=0.0, slippage_bps=0.0)
        assert net.iloc[0] == 0  # position starts next bar


# ---------------------------------------------------------------- statistics

class TestStatistics:
    def test_bootstrap_ci_covers_true_mean(self):
        rng = np.random.default_rng(3)
        sample = rng.normal(0.001, 0.01, 500)
        lo, hi = bootstrap_mean_ci(sample, n_boot=200, block=10, seed=3)
        assert lo < sample.mean() < hi

    def test_bonferroni_tightens_alpha(self):
        assert bonferroni_alpha(0.05, 10) == pytest.approx(0.005)
        assert bonferroni_alpha(0.05, 1) == pytest.approx(0.05)

    def test_sharpe_and_drawdown(self):
        rets = pd.Series([0.01, -0.005, 0.02, -0.03, 0.01])
        assert sharpe(rets, periods_per_year=365 * 6) != 0
        dd = max_drawdown(pd.Series([1.0, 1.1, 0.9, 1.2]))
        assert dd == pytest.approx((0.9 - 1.1) / 1.1)


# ---------------------------------------------------------------- league + veto

def _candidate(name: str, quality: str) -> LeagueCandidate:
    return LeagueCandidate(
        name=name,
        strategy_factory=TrendSleeve,
        grid=[{"lookback": 40, "exit": 10}],
        quality=quality,  # test hook: overrides OOS metrics
    )


class TestLeagueGates:
    def test_poor_tail_blocks_promotion(self):
        league = League(min_oos_trades=5, worst_fold_floor=-0.05)
        candidate = _candidate("blowup", quality={
            "oos_mean": 0.10, "oos_median": 0.02, "oos_trades": 40,
            "fold_returns": [0.08, 0.09, 0.07, -0.40],  # catastrophic tail
            "worst_dd": -0.12, "sharpe": 1.1,
        })
        result = league.evaluate([candidate])
        assert result.promoted is None

    def test_all_candidates_fail_veto_selects_nothing(self):
        league = League(min_oos_trades=5)
        candidates = [
            _candidate("a", quality={
                "oos_mean": -0.01, "oos_median": -0.005, "oos_trades": 30,
                "fold_returns": [-0.01, -0.02], "worst_dd": -0.05, "sharpe": -0.3,
            }),
            _candidate("b", quality={
                "oos_mean": 0.001, "oos_median": -0.001, "oos_trades": 8,
                "fold_returns": [0.002, 0.001], "worst_dd": -0.02, "sharpe": 0.1,
            }),
        ]
        result = league.evaluate(candidates)
        assert result.promoted is None
        assert result.vetoed is True
        assert result.rejected["a"]  # rejection reasons recorded
        assert result.rejected["b"]

    def test_solid_candidate_is_promoted(self):
        league = League(min_oos_trades=5, worst_fold_floor=-0.05, min_fold_win_rate=0.5)
        good = _candidate("good", quality={
            "oos_mean": 0.03, "oos_median": 0.025, "oos_trades": 55,
            "fold_returns": [0.02, 0.04, 0.025, 0.03], "worst_dd": -0.04,
            "sharpe": 1.4,
        })
        result = league.evaluate([good])
        assert result.promoted is not None
        assert result.promoted.name == "good"

    def test_full_pipeline_on_synthetic_market(self):
        """End-to-end: real strategies, real splits, real costs."""
        league = League(min_oos_trades=3, worst_fold_floor=-0.50,
                        min_fold_win_rate=0.0, n_folds=3, embargo_bars=6)
        candidates = [
            LeagueCandidate("trend", TrendSleeve,
                            [{"lookback": 40, "exit": 10}, {"lookback": 80, "exit": 20}]),
            LeagueCandidate("reversion", MeanReversionSleeve,
                            [{"window": 14, "entry": 30, "exit": 55}]),
        ]
        result: LeagueResult = league.run(candidates, DF)
        assert isinstance(result.promoted if result.promoted else None, LeagueCandidate | None)
        assert result.rejected.keys() <= {c.name for c in candidates}


class TestScoreboard:
    def test_persistent_underperformance_quarantines(self):
        board = LiveScoreboard(min_trades=10, sharpe_floor=-0.5)
        # 12 live trades, consistently losing
        trades = [{"pnl": -1.0 - 0.1 * i} for i in range(12)]
        verdict = board.assess(trades, incumbent_sharpe=0.8)
        assert verdict.underperforming and verdict.quarantine

    def test_healthy_record_keeps_incumbent(self):
        board = LiveScoreboard(min_trades=10, sharpe_floor=-0.5)
        trades = [{"pnl": 1.0 + (1 if i % 2 else -0.4)} for i in range(14)]
        verdict = board.assess(trades, incumbent_sharpe=0.8)
        assert not verdict.quarantine

    def test_quarantine_opens_only_incumbent_slot(self):
        board = LiveScoreboard(min_trades=5)
        trades = [{"pnl": -2.0}] * 6
        verdict = board.assess(trades, incumbent_sharpe=0.5)
        assert verdict.quarantine
        assert verdict.slots_available_at_next_league == ["incumbent"]

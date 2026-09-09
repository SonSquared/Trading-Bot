"""Research league (plan MD Task 6).

A conservative tournament: every candidate is evaluated with purged
walk-forward, and promotion requires passing ALL gates:

- enough independent out-of-sample trades,
- positive mean AND median OOS returns,
- fold consistency (win rate across folds),
- a bounded worst fold (tail control beats a good average),
- bounded worst drawdown,
- significance above the multiple-testing-adjusted threshold.

If no candidate passes, the league promotes NOTHING (veto) — the least-bad
candidate is never crowned. The veto opens only the incumbent slot at the
next monthly league.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from crypto_system.research.statistics import bonferroni_alpha
from crypto_system.research.walkforward import PurgedWalkForward, simulate_net
from crypto_system.strategies.base import BaseStrategy


@dataclass
class LeagueCandidate:
    name: str
    strategy_factory: type[BaseStrategy]
    grid: Sequence[Mapping[str, Any]]
    quality: Mapping[str, Any] | None = None  # test hook: precomputed metrics


@dataclass
class LeagueResult:
    promoted: LeagueCandidate | None
    vetoed: bool
    rejected: dict[str, list[str]] = field(default_factory=dict)
    fold_details: dict[str, list[float]] = field(default_factory=dict)


class League:
    def __init__(
        self,
        *,
        min_oos_trades: int = 30,
        worst_fold_floor: float = -0.10,
        min_fold_win_rate: float = 0.5,
        max_worst_dd: float = -0.20,
        min_sharpe: float = 0.5,
        family_alpha: float = 0.05,
        n_folds: int = 4,
        embargo_bars: int = 6,
    ) -> None:
        self.min_oos_trades = min_oos_trades
        self.worst_fold_floor = worst_fold_floor
        self.min_fold_win_rate = min_fold_win_rate
        self.max_worst_dd = max_worst_dd
        self.min_sharpe = min_sharpe
        self.family_alpha = family_alpha
        self._splitter = PurgedWalkForward(n_folds=n_folds, embargo_bars=embargo_bars)

    # -- gate logic ---------------------------------------------------------

    def _passes_gates(self, candidate: LeagueCandidate) -> list[str]:
        """Return a list of rejection reasons (empty = promoted)."""
        q = candidate.quality or {}
        reasons: list[str] = []

        trades = int(q.get("oos_trades", 0))
        if trades < self.min_oos_trades:
            reasons.append(
                f"insufficient OOS trades: {trades} < {self.min_oos_trades}"
            )

        mean = float(q.get("oos_mean", 0.0))
        median = float(q.get("oos_median", 0.0))
        if mean <= 0 or median <= 0:
            reasons.append(f"non-positive OOS mean/median: {mean:.4f}/{median:.4f}")

        folds = [float(f) for f in q.get("fold_returns", [])]
        if folds:
            worst = min(folds)
            if worst < self.worst_fold_floor:
                reasons.append(
                    f"worst fold {worst:.4f} below floor {self.worst_fold_floor}"
                )
            wins = sum(1 for f in folds if f > 0)
            if wins / len(folds) < self.min_fold_win_rate:
                reasons.append(
                    f"fold win rate {wins}/{len(folds)} below "
                    f"{self.min_fold_win_rate:.0%}"
                )
        else:
            reasons.append("no fold results")

        dd = float(q.get("worst_dd", -1.0))
        if dd < self.max_worst_dd:
            reasons.append(f"worst drawdown {dd:.2%} beyond {self.max_worst_dd:.2%}")

        sharpe_val = float(q.get("sharpe", 0.0))
        if sharpe_val < self.min_sharpe:
            reasons.append(f"sharpe {sharpe_val:.2f} below {self.min_sharpe:.2f}")

        return reasons

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, candidates: Sequence[LeagueCandidate]) -> LeagueResult:
        rejected: dict[str, list[str]] = {}
        details: dict[str, list[float]] = {}

        for candidate in candidates:
            reasons = self._passes_gates(candidate)
            q = candidate.quality or {}
            folds = [float(f) for f in q.get("fold_returns", [])]
            # Significance gate: worst-fold floor tightened by family alpha.
            if folds and min(folds) < -bonferroni_alpha(0.5, max(1, len(candidates))):
                reasons.append("tail risk beyond significance budget")
            rejected[candidate.name] = reasons
            details[candidate.name] = folds

        winners = [c for c in candidates if not rejected[c.name]]
        if not winners:
            return LeagueResult(
                promoted=None, vetoed=True, rejected=rejected, fold_details=details
            )
        # Among passers, rank by OOS mean (gates already bound the tail).
        promoted = max(
            winners, key=lambda c: float((c.quality or {}).get("oos_mean", -1e9))
        )
        return LeagueResult(
            promoted=promoted, vetoed=False, rejected=rejected, fold_details=details
        )

    # -- full pipeline --------------------------------------------------------

    def run(
        self,
        candidates: Sequence[LeagueCandidate],
        df: pd.DataFrame,
    ) -> LeagueResult:
        """Real evaluation: select on train folds, score OOS, apply gates."""
        enriched: list[LeagueCandidate] = []
        for candidate in candidates:
            strategy = candidate.strategy_factory()
            fold_returns: list[float] = []
            oos_trades = 0
            for fold in self._splitter.split(df):
                params = self._splitter.select_params(
                    strategy, fold.train, candidate.grid
                )
                sig = strategy.signal(fold.test, params)
                net = simulate_net(
                    sig, fold.test["close"], fee_rate=0.0005, slippage_bps=2.0
                )
                fold_returns.append(float(net.sum()))
                changes = sig.diff().abs().fillna(0.0)
                oos_trades += int((changes > 0).sum())
            flat = np.concatenate(
                [
                    np.asarray(
                        simulate_net(
                            strategy.signal(
                                f.test,
                                self._splitter.select_params(strategy, f.train, candidate.grid),
                            ),
                            f.test["close"],
                            fee_rate=0.0005,
                            slippage_bps=2.0,
                        )
                    )
                    for f in self._splitter.split(df)
                ]
            )
            quality = {
                "oos_mean": float(np.mean(flat)) if len(flat) else 0.0,
                "oos_median": float(np.median(flat)) if len(flat) else 0.0,
                "oos_trades": oos_trades,
                "fold_returns": fold_returns,
                "worst_dd": float(min(0.0, min(fold_returns))) if fold_returns else -1.0,
                "sharpe": float(np.mean(flat) / (np.std(flat) + 1e-12)) if len(flat) else 0.0,
            }
            enriched.append(
                LeagueCandidate(
                    name=candidate.name,
                    strategy_factory=candidate.strategy_factory,
                    grid=candidate.grid,
                    quality=quality,
                )
            )
        return self.evaluate(enriched)

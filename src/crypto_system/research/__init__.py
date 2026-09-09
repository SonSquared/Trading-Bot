"""Research layer: walk-forward, statistics, league, scoreboard."""

from crypto_system.research.league import League, LeagueCandidate, LeagueResult
from crypto_system.research.scoreboard import LiveScoreboard, ScoreboardVerdict
from crypto_system.research.statistics import (
    bonferroni_alpha,
    bootstrap_mean_ci,
    max_drawdown,
    sharpe,
)
from crypto_system.research.walkforward import Fold, PurgedWalkForward, simulate_net

__all__ = [
    "League",
    "LeagueCandidate",
    "LeagueResult",
    "LiveScoreboard",
    "ScoreboardVerdict",
    "bonferroni_alpha",
    "bootstrap_mean_ci",
    "max_drawdown",
    "sharpe",
    "Fold",
    "PurgedWalkForward",
    "simulate_net",
]

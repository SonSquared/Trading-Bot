"""Live scoreboard (plan MD Task 6): closed-paper-trade performance review.

Persistent live underperformance quarantines the incumbent: its slot opens
at the NEXT monthly league (never an immediate, reactive swap), and the
quarantine is the only way a slot frees up between scheduled promotions.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crypto_system.research.statistics import max_drawdown, sharpe


@dataclass(frozen=True)
class ScoreboardVerdict:
    underperforming: bool
    quarantine: bool
    live_sharpe: float
    live_drawdown: float
    n_trades: int
    reason: str
    slots_available_at_next_league: list[str]


class LiveScoreboard:
    def __init__(
        self,
        *,
        min_trades: int = 30,
        sharpe_floor: float = 0.0,
        dd_floor: float = -0.25,
    ) -> None:
        self.min_trades = min_trades
        self.sharpe_floor = sharpe_floor
        self.dd_floor = dd_floor

    def assess(
        self,
        closed_trades: list[dict[str, float]],
        *,
        incumbent_sharpe: float,
        periods_per_year: int = 365 * 6,
    ) -> ScoreboardVerdict:
        n = len(closed_trades)
        pnls = pd.Series([float(t.get("pnl", 0.0)) for t in closed_trades])
        reasons: list[str] = []

        if n >= self.min_trades:
            live_sharpe = sharpe(pnls, periods_per_year=periods_per_year)
            equity = pnls.cumsum() + 1.0
            live_dd = max_drawdown(equity)
            if live_sharpe < self.sharpe_floor:
                reasons.append(
                    f"live sharpe {live_sharpe:.2f} below floor {self.sharpe_floor:.2f}"
                )
            if live_dd < self.dd_floor:
                reasons.append(f"live drawdown {live_dd:.2%} beyond {self.dd_floor:.2%}")
            if live_sharpe < min(0.0, incumbent_sharpe - 1.0):
                reasons.append(
                    f"live sharpe {live_sharpe:.2f} far below incumbent "
                    f"{incumbent_sharpe:.2f}"
                )
        else:
            live_sharpe = 0.0
            live_dd = 0.0
            reasons.append(f"insufficient trades ({n}/{self.min_trades}) — monitoring")

        quarantine = bool(reasons) and n >= self.min_trades
        return ScoreboardVerdict(
            underperforming=bool(reasons),
            quarantine=quarantine,
            live_sharpe=live_sharpe,
            live_drawdown=live_dd,
            n_trades=n,
            reason="; ".join(reasons) or "healthy",
            slots_available_at_next_league=["incumbent"] if quarantine else [],
        )

"""Deterministic improvement runner (plan MD Task 7 / scripts/improve.py).

Modes:
- ``--quick``: replay built-in fixtures only — NO network, NO promotion,
  NO state push, NO Telegram. Deterministic: same input, same output.
- ``--report-only``: render and print the weekly report, nothing else.
- full mode (neither flag): acquire the state lease, verify the ledger,
  run the monthly league evaluation, promote ONLY if gates pass, render
  the report, optionally send it. Promotion cadence is monthly — this
  runner never promotes weekly.

Quick mode is the "run on my PC when I open it" entry point: it shows the
current standing and what the league WOULD do, without side effects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crypto_system.audit.ledger import Ledger  # noqa: E402
from crypto_system.config import Settings  # noqa: E402
from crypto_system.models import ExecutionMode  # noqa: E402
from crypto_system.reporting.dashboard import render_weekly_report  # noqa: E402
from crypto_system.reporting.telegram import TelegramReporter  # noqa: E402
from crypto_system.research.league import League, LeagueCandidate  # noqa: E402
from crypto_system.state_sync import StateLock, new_run_id  # noqa: E402
from crypto_system.strategies.base import (  # noqa: E402
    MeanReversionSleeve,
    TrendSleeve,
)

app = typer.Typer(help="Self-improvement runner: research, promote (monthly), report.")

QUICK_FIXTURE_STATE: dict[str, Any] = {
    "equity": 10_312.50,
    "start_equity": 10_000.00,
    "n_trades": 18,
    "positions": {
        "BTCUSDT": {"side": "long", "qty": 0.042, "entry": 64_100.0, "price": 65_250.0}
    },
    "equity_curve": [10_000, 10_050, 10_020, 10_120, 10_312.50],
    "halts": [],
    "vetoes": [],
}

QUICK_FIXTURE_CANDIDATES = [
    LeagueCandidate(
        name="trend",
        strategy_factory=TrendSleeve,
        grid=[{"lookback": 40, "exit": 10}],
        quality={
            "oos_mean": 0.021,
            "oos_median": 0.014,
            "oos_trades": 34,
            "fold_returns": [0.031, 0.018, 0.024, 0.011],
            "worst_dd": -0.062,
            "sharpe": 1.21,
        },
    ),
    LeagueCandidate(
        name="reversion",
        strategy_factory=MeanReversionSleeve,
        grid=[{"window": 14, "entry": 30, "exit": 55}],
        quality={
            "oos_mean": 0.004,
            "oos_median": -0.001,
            "oos_trades": 41,
            "fold_returns": [0.012, -0.004, 0.006, -0.002],
            "worst_dd": -0.031,
            "sharpe": 0.34,
        },
    ),
]


def _load_state(settings: Settings) -> dict[str, Any]:
    path = Path(settings.paper_state_dir) / "paper_state.json"
    if path.exists():
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data
    return dict(QUICK_FIXTURE_STATE)


def _fixture_market() -> "Any":
    """Deterministic synthetic market for offline league replay."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(11)
    n = 900
    drift = np.concatenate(
        [np.full(n // 3, 0.0008), np.full(n // 3, -0.0004),
         np.full(n - 2 * (n // 3), 0.0005)]
    )
    rets = drift + rng.normal(0, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.roll(close, 1),
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 100.0,
        },
        index=idx,
    )


@app.command()
def run(
    quick: bool = typer.Option(False, "--quick", help="Fixture replay: no network, no promotion."),
    report_only: bool = typer.Option(False, "--report-only", help="Print the report and exit."),
    send_telegram: bool = typer.Option(
        False, "--send", help="Send the report via Telegram (explicit opt-in)."
    ),
) -> int:
    settings = Settings.load()
    run_id = new_run_id() if not quick else "quick-deterministic"

    if quick or report_only:
        # Quick mode: deterministic fixture replay. No lock, no network,
        # no promotion, no sends — enforced, not promised.
        league = League()
        result = league.evaluate(QUICK_FIXTURE_CANDIDATES)
        state = dict(QUICK_FIXTURE_STATE)
        promotion = result.promoted.name if result.promoted else "VETOED — no replacement"
        state["league_promotion"] = promotion
        state["league_vetoed"] = result.vetoed
        report = render_weekly_report(state)
        typer.echo(report)
        typer.echo(
            "\nQuick mode: fixture replay only — no network, no promotion, "
            "no state push, no Telegram send."
        )
        return 0

    # Full mode: real work under the lease.
    lock = StateLock(
        Path(settings.paper_state_dir) / "improve.lock",
        owner=f"improve:{run_id}",
        ttl_seconds=900,
    )
    with lock:
        ledger = Ledger(
            Path(settings.paper_state_dir) / "paper_ledger.jsonl",
            mode=ExecutionMode.PAPER,
            run_id=run_id,
        )
        verification = ledger.verify()
        if not verification.valid:
            typer.echo(f"LEDGER VERIFY FAILED: {verification.error}", err=True)
            return 2

        state = _load_state(settings)
        report = render_weekly_report(state)
        typer.echo(report)

        if send_telegram:
            reporter = TelegramReporter(
                bot_token=(
                    settings.secrets.telegram_bot_token.get_secret_value()
                    if settings.secrets.telegram_bot_token
                    else None
                ),
                chat_id=settings.secrets.telegram_chat_id,
                enabled=settings.telegram_report_enabled,
            )
            sent = reporter.send(report, dry_run=False)
            typer.echo(f"telegram sent: {sent}")
        ledger.append({"type": "IMPROVE_RUN", "report_only": report_only})
        return 0


def main() -> None:
    raise SystemExit(app())


if __name__ == "__main__":
    main()

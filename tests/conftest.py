"""Shared test fixtures.

Two protections live here:

1. **Path sandbox (autouse).** ``scripts/paper_trader.py`` and friends bind
   ``Path("data/results")`` constants at import time. Any test that exercises
   the trading paths without an explicit ``isolated_paths`` fixture would
   otherwise write opens/closes into the PRODUCTION trade log. (This actually
   happened: 203 synthetic entries landed in ``paper_trades.jsonl`` and had to
   be quarantined.) The autouse fixture below redirects every data-path
   constant to a per-test sandbox BEFORE the test runs, so isolation is the
   default, not opt-in. Tests that need bespoke paths (``isolated_paths``)
   override these same names later and win.

   ``scripts/forward_run.py`` mutes ``paper_trader.log_trade`` at import time
   so forward replays never write to the production trade log. That global
   side effect is snapshotted/restored here too: the pristine module state is
   captured at conftest import time (before any test module is collected) and
   restored after every test, so import order can never change test outcomes.

2. **Session tripwire.** Before the session starts we fingerprint the
   production data files; after it ends we re-check. If the suite — despite
   the sandbox — touched production data, the session fails loudly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import scripts.paper_trader as pt

_PRISTINE_STATE = dict(vars(pt))

# Every production data path the bot modules bind at import time.
_SANDBOXED_PATHS = {
    "LOG_DIR": ".",
    "TRADE_LOG": "paper_trades.jsonl",
    "STATE_FILE": "paper_state.json",
    "SUMMARY_FILE": "paper_summary.json",
    "RUN_LOG": "run_history.jsonl",
    "OPTIMIZED_PARAMS_FILE": "bot_strategy_params.json",
    "LOCK_FILE": "bot.lock",
}

# Production files whose integrity the tripwire guards.
_PRODUCTION_FILES = [
    Path("data/results/paper_trades.jsonl"),
    Path("data/results/paper_state.json"),
    Path("data/results/run_history.jsonl"),
]


def _fingerprint(path: Path) -> tuple | None:
    if not path.exists():
        return None
    return (str(path), path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture(autouse=True)
def _isolate_paper_trader_module(monkeypatch, tmp_path):
    # Redirect every data-path constant into this test's sandbox.
    sandbox = tmp_path / "results"
    sandbox.mkdir(parents=True, exist_ok=True)
    for name, fname in _SANDBOXED_PATHS.items():
        if hasattr(pt, name):
            monkeypatch.setattr(pt, name, sandbox / fname if fname != "." else sandbox)
    yield
    # Restore pristine module state (undoes e.g. forward_run's log_trade mute).
    vars(pt).clear()
    vars(pt).update(_PRISTINE_STATE)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_sessionstart(session):
    session._prod_before = {p: _fingerprint(p) for p in _PRODUCTION_FILES}
    result = yield
    after = {p: _fingerprint(p) for p in _PRODUCTION_FILES}
    changed = [
        f"{p}: {session._prod_before.get(p)} -> {after.get(p)}"
        for p in _PRODUCTION_FILES
        if session._prod_before.get(p) != after.get(p)
    ]
    if changed:
        raise AssertionError(
            "PRODUCTION DATA CHANGED DURING TEST RUN — the sandbox leaked:\n  "
            + "\n  ".join(changed)
        )
    return result

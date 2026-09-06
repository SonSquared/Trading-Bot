"""Shared test fixtures.

``scripts/forward_run.py`` mutes ``paper_trader.log_trade`` at import time
so forward replays never write to the production trade log. That is the
correct production behavior, but it is a global side effect: once any test
module imports ``forward_run`` (at collection time), every later test sees
a muted trade log.

The pristine module state is therefore snapshotted HERE, at conftest
import time — pytest loads conftest before any test module, so the
snapshot predates all collection-time imports. An autouse fixture then
restores the module state after every test, so import order can never
change test outcomes.
"""

from __future__ import annotations

import pytest

import scripts.paper_trader as pt

_PRISTINE_STATE = dict(vars(pt))


@pytest.fixture(autouse=True)
def _isolate_paper_trader_module():
    yield
    vars(pt).clear()
    vars(pt).update(_PRISTINE_STATE)

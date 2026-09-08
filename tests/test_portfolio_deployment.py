"""Portfolio-deployment integrity tests.

The Actions runner must trade exactly the portfolio the walk-forward league
promoted. This pins two layers of that guarantee:

1. The committed bot_strategy_params.json loads cleanly and its strategies
   all exist in the registry (the loader would silently skip invalid ones).
2. The hardcoded fallback portfolio in paper_trader.py mirrors the committed
   file exactly — so if the params file is ever lost or corrupt, the fallback
   can never silently resurrect a strategy the league rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import scripts.paper_trader as pt
from trading_system.strategies import STRATEGY_REGISTRY


DEPLOYED_PARAMS = Path("data/results/bot_strategy_params.json")


def _required_keys() -> list[str]:
    return ["strategy", "pair", "timeframe", "weight", "params"]


class TestDeployedPortfolio:
    def test_params_file_exists_and_is_committed(self):
        assert DEPLOYED_PARAMS.exists(), (
            "the deployed portfolio file is missing — the cloud runner would "
            "fall back to hardcoded defaults"
        )
        # Tracked in git (the whole point of the fix): check-ignore must not
        # flag it. If gitignored, every Actions run silently used the stale
        # fallback portfolio.
        import subprocess
        result = subprocess.run(
            ["git", "check-ignore", str(DEPLOYED_PARAMS)],
            capture_output=True, text=True)
        assert result.returncode != 0, (
            "bot_strategy_params.json is gitignored — Actions cannot see the "
            "promoted portfolio and silently trades the stale fallback"
        )

    def test_params_file_is_valid_and_registry_complete(self):
        params = json.loads(DEPLOYED_PARAMS.read_text())
        assert params, "deployed portfolio must not be empty"
        for name, cfg in params.items():
            for key in _required_keys():
                assert key in cfg, f"{name}: missing '{key}'"
            assert cfg["strategy"] in STRATEGY_REGISTRY, (
                f"{name}: strategy '{cfg['strategy']}' not in registry — the "
                f"loader would silently skip this config"
            )
            assert 0 < cfg["weight"] <= 1.0, f"{name}: weight out of range"

    def test_weights_sum_to_one(self):
        params = json.loads(DEPLOYED_PARAMS.read_text())
        total = sum(cfg["weight"] for cfg in params.values())
        assert abs(total - 1.0) < 0.01, (
            f"portfolio weights sum to {total:.3f}, expected ~1.0"
        )

    def test_fallback_portfolio_mirrors_deployed(self):
        deployed = json.loads(DEPLOYED_PARAMS.read_text())
        assert pt.STRATEGIES == deployed, (
            "hardcoded fallback portfolio diverged from the committed "
            "deployed portfolio — a fallback that resurrects rejected "
            "strategies is worse than no fallback"
        )

    def test_loader_prefers_committed_file(self, monkeypatch, tmp_path):
        # Simulate the real runner: OPTIMIZED_PARAMS_FILE redirected into the
        # sandbox but content identical to the committed file (conftest
        # sandboxes the path; we restore the real file's content for this
        # test because the assertion is about the loading path itself).
        monkeypatch.setattr(
            pt, "OPTIMIZED_PARAMS_FILE", DEPLOYED_PARAMS, raising=False)
        loaded = pt.load_active_strategies()
        deployed = json.loads(DEPLOYED_PARAMS.read_text())
        assert loaded == deployed
        assert pt.ACTIVE_STRATEGIES == deployed

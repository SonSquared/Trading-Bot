"""Regression test: when MULTIPLE pairs promote in one league run, each
promotion must build on the previous one instead of overwriting it.

This reproduces the real production incident: ETH and BTC both promoted,
but the BTC deploy call rebuilt from the original incumbent, discarding
the ETH promotion and dropping the deployed winner entirely.
"""
import json

import pytest

import scripts.strategy_league as league


@pytest.fixture
def league_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(league, "PARAMS_FILE", tmp_path / "bot_strategy_params.json")
    monkeypatch.setattr(league, "PARAMS_BACKUP", tmp_path / "backup.json")
    return tmp_path


def _cfg(strategy, pair, weight):
    return {
        "strategy": strategy, "pair": pair, "timeframe": "4h",
        "weight": weight, "params": {"period": 10},
    }


class TestChainedMultiPairPromotion:
    def test_second_promotion_does_not_erase_first(self, league_paths):
        """ETH promotes to Donchian, then BTC promotes to Keltner.
        The final config must contain BOTH winners plus the untouched
        RSI slot, matching what the fixed deploy_winner semantics promise."""
        incumbent = {
            "Bollinger_Reversion_ETH_USDT_USDT": _cfg("Bollinger_Reversion", "ETH_USDT_USDT", 0.4),
            "Bollinger_Reversion_BTC_USDT_USDT": _cfg("Bollinger_Reversion", "BTC_USDT_USDT", 0.35),
            "RSI_Reversion_BTC_USDT_USDT": _cfg("RSI_Reversion", "BTC_USDT_USDT", 0.25),
        }

        eth_round = {
            "pair": "ETH_USDT_USDT",
            "winner": {"strategy": "Donchian_Breakout",
                       "params": {"channel_period": 40, "exit_period": 10}},
        }
        btc_round = {
            "pair": "BTC_USDT_USDT",
            "winner": {"strategy": "Keltner_Breakout",
                       "params": {"ema_period": 20, "multiplier": 2.5}},
        }

        # Chain exactly as main() now does.
        working = league.deploy_winner("ETH_USDT_USDT", eth_round, incumbent)
        working = league.deploy_winner("BTC_USDT_USDT", btc_round, working)

        # Both winners present.
        assert "Donchian_Breakout_ETH_USDT_USDT" in working
        assert "Keltner_Breakout_BTC_USDT_USDT" in working
        # Losers gone, unevaluated RSI slot preserved.
        assert "Bollinger_Reversion_ETH_USDT_USDT" not in working
        assert "Bollinger_Reversion_BTC_USDT_USDT" not in working
        assert "RSI_Reversion_BTC_USDT_USDT" in working
        # Weights inherited from the slots they replaced.
        assert working["Donchian_Breakout_ETH_USDT_USDT"]["weight"] == 0.4
        assert working["Keltner_Breakout_BTC_USDT_USDT"]["weight"] == 0.35
        assert working["RSI_Reversion_BTC_USDT_USDT"]["weight"] == 0.25

    def test_production_incident_replay(self, league_paths):
        """Replay the exact sequence that ran in production: both pairs
        promote, deploy once at the end. With the old chained bug the
        written file would contain only the BTC winner."""
        incumbent = {
            "Bollinger_Reversion_ETH_USDT_USDT": _cfg("Bollinger_Reversion", "ETH_USDT_USDT", 0.4),
            "Bollinger_Reversion_BTC_USDT_USDT": _cfg("Bollinger_Reversion", "BTC_USDT_USDT", 0.35),
            "RSI_Reversion_BTC_USDT_USDT": _cfg("RSI_Reversion", "BTC_USDT_USDT", 0.25),
        }
        league.PARAMS_FILE.write_text(
            json.dumps(incumbent, indent=2), encoding="utf-8")

        eth_round = {"pair": "ETH_USDT_USDT",
                     "winner": {"strategy": "Donchian_Breakout",
                                "params": {"channel_period": 40}}}
        btc_round = {"pair": "BTC_USDT_USDT",
                     "winner": {"strategy": "Donchian_Breakout",
                                "params": {"channel_period": 40}}}

        working = None
        for pair, rnd in (("ETH_USDT_USDT", eth_round),
                          ("BTC_USDT_USDT", btc_round)):
            if rnd is not None:
                base = working if working is not None else incumbent
                working = league.deploy_winner(pair, rnd, base)
        league.apply_config(working, dry_run=False)

        written = json.loads(league.PARAMS_FILE.read_text(encoding="utf-8"))
        strategies_by_pair = {
            cfg["pair"]: cfg["strategy"] for cfg in written.values()}
        assert strategies_by_pair["ETH_USDT_USDT"] == "Donchian_Breakout"
        assert strategies_by_pair["BTC_USDT_USDT"] == "Donchian_Breakout"
        # Three slots survive: both winners + the unevaluated RSI slot.
        assert len(written) == 3
        assert league.PARAMS_BACKUP.exists()

"""Task 1 (plan MD): foundation — immutable models and safe configuration.

Every rule the plan calls "global constraint" that can be enforced at the
schema layer is pinned here:
- live mode requires explicit enablement AND isolated state dirs,
- 50x leverage is a hard schema ceiling (not an operating target),
- per-trade risk is capped,
- OrderIntent is immutable and side-constrained,
- secrets never survive repr()/str() (they must not leak into logs).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from crypto_system.config import Settings
from crypto_system.models import ExecutionMode, OrderIntent, RiskLimits


def test_live_requires_explicit_enablement_and_isolated_state(tmp_path):
    with pytest.raises(ValidationError):
        Settings(mode="live", live_enabled=False, paper_state_dir=tmp_path)


def test_max_leverage_cannot_exceed_fifty():
    with pytest.raises(ValidationError):
        RiskLimits(max_leverage=50.01)


def test_max_leverage_below_zero_rejected():
    with pytest.raises(ValidationError):
        RiskLimits(max_leverage=0.0)


def test_max_leverage_default_is_three_and_cap_is_fifty():
    limits = RiskLimits()
    assert limits.max_leverage == 3
    assert RiskLimits(max_leverage=50).max_leverage == 50


def test_max_risk_per_trade_default_and_cap():
    assert RiskLimits().max_risk_per_trade == 0.0025
    with pytest.raises(ValidationError):
        RiskLimits(max_risk_per_trade=0.006)


def test_settings_defaults_to_paper_and_live_disabled():
    settings = Settings.load(env={})
    assert settings.mode == "paper"
    assert settings.live_enabled is False
    assert ExecutionMode.PAPER.value == "paper"


def test_live_mode_with_isolated_state_and_enablement_is_valid(tmp_path):
    settings = Settings(
        mode="live",
        live_enabled=True,
        paper_state_dir=tmp_path / "paper",
        live_state_dir=tmp_path / "live",
    )
    assert settings.mode == "live"


def test_live_enabled_flag_requires_live_mode():
    with pytest.raises(ValidationError):
        Settings(live_enabled=True)


def test_order_intent_is_immutable_and_side_constrained():
    intent = OrderIntent(
        symbol="BTCUSDT", side="long", notional=100.0, strategy="baseline_trend"
    )
    with pytest.raises(ValidationError):
        intent.side = "short"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        OrderIntent(symbol="BTCUSDT", side="sideways", notional=1.0, strategy="x")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        OrderIntent(symbol="BTCUSDT", side="long", notional=0, strategy="x")


def test_risk_limits_are_immutable():
    limits = RiskLimits()
    with pytest.raises(ValidationError):
        limits.max_leverage = 10  # type: ignore[misc]


def test_settings_hides_secrets_in_repr():
    settings = Settings.load(
        env={
            "CS_MODE": "paper",
            "CS_TELEGRAM_BOT_TOKEN": "secret-token-value",
            "CS_BINANCE_API_SECRET": "secret-key-value",
        }
    )
    rendered = f"{settings!r} {settings.model_dump()}"
    assert "secret-token-value" not in rendered
    assert "secret-key-value" not in rendered


def test_settings_parse_live_enabled_from_env(tmp_path):
    settings = Settings.load(
        env={
            "CS_MODE": "live",
            "CS_LIVE_ENABLED": "true",
            "CS_PAPER_STATE_DIR": str(tmp_path / "p"),
            "CS_LIVE_STATE_DIR": str(tmp_path / "l"),
        }
    )
    assert settings.mode == "live" and settings.live_enabled is True


def test_settings_reject_unknown_env_settings():
    with pytest.raises(ValidationError):
        Settings.load(env={"CS_MODE": "paper", "CS_FLY_MODE": "enabled"})

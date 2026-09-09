"""Safe configuration (plan MD Task 1).

Rules enforced here, fail-closed:
- Mode is paper unless explicitly set; live mode requires BOTH
  ``live_enabled=True`` AND a live_state_dir distinct from paper_state_dir.
- live_enabled without live mode is a config error (a stray flag must never
  silently enable anything later).
- Secrets come only from the environment (or an injected mapping for tests);
  they are SecretStr so they never leak through repr/str/dump.
- Extra env keys are rejected: a typo'd variable fails loudly instead of
  silently using a default (the same lesson as the old bot's cache amnesia).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from crypto_system.models import ExecutionMode

_ENV_PREFIX = "CS_"


class Secrets(BaseModel):
    """Credential container. Values are SecretStr: never logged, never dumped."""

    model_config = ConfigDict(frozen=True)

    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None
    binance_api_key: SecretStr | None = None
    binance_api_secret: SecretStr | None = None


class Settings(BaseModel):
    """Validated, secret-safe application settings."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

    mode: ExecutionMode = ExecutionMode.PAPER
    live_enabled: bool = False

    paper_state_dir: Path = Path("data/cs_state/paper")
    live_state_dir: Path = Path("data/cs_state/live")

    # Data + execution
    exchange: str = "binance-usdm"
    quote_asset: str = "USDT"
    initial_cash: float = 10_000.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    funding_interval_hours: int = 8

    # Research league
    promotion_cadence: str = "monthly"  # the sole auto-promotion cadence

    # Reporting
    telegram_report_enabled: bool = True

    secrets: Secrets = Field(default_factory=Secrets, exclude=True, repr=False)

    @model_validator(mode="after")
    def _live_mode_rules(self) -> "Settings":
        if self.mode is ExecutionMode.LIVE:
            if not self.live_enabled:
                raise ValueError(
                    "live mode requires live_enabled=True (explicit enablement)"
                )
            if self.live_state_dir == self.paper_state_dir:
                raise ValueError(
                    "live_state_dir must differ from paper_state_dir "
                    "(paper and live state are isolated)"
                )
        elif self.live_enabled:
            raise ValueError(
                "live_enabled=True requires mode='live' — never pre-arm live"
            )
        return self

    @classmethod
    def load(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Load settings from an optional env mapping, then real environment.

        Test-injected mappings take precedence; production reads real env and
        an optional ``.env`` file (never committed).
        """
        if env is not None:
            merged: dict[str, Any] = dict(env)
        else:
            merged = {k: v for k, v in os.environ.items() if k.startswith(_ENV_PREFIX)}
            dot_env = Path(".env")
            if dot_env.exists():
                for line in dot_env.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    merged.setdefault(key.strip(), value.strip())
        # Only keys with the CS_ prefix are configuration; ignore the rest.
        scoped: dict[str, Any] = {
            k.removeprefix(_ENV_PREFIX).lower(): v
            for k, v in merged.items()
            if k.startswith(_ENV_PREFIX)
        }
        if "mode" in scoped:
            mode_raw = scoped.pop("mode")
            mode: ExecutionMode | None = ExecutionMode(mode_raw)
        else:
            mode = None
        secrets = Secrets(
            telegram_bot_token=scoped.pop("telegram_bot_token", None),
            telegram_chat_id=scoped.pop("telegram_chat_id", None),
            binance_api_key=scoped.pop("binance_api_key", None),
            binance_api_secret=scoped.pop("binance_api_secret", None),
        )
        kwargs: dict[str, Any] = dict(scoped)
        if mode is not None:
            kwargs["mode"] = mode
        return cls(secrets=secrets, **kwargs)

    @property
    def is_live(self) -> bool:
        return self.mode is ExecutionMode.LIVE

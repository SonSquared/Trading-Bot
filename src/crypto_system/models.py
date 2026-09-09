"""Immutable domain models for the crypto_system platform.

Design rules (plan MD, global constraints):
- Every model is frozen (immutable) — nothing mutates a decision after the fact.
- Risk caps are enforced at the schema layer, so an unsafe config cannot even
  be constructed, let alone deployed.
- Secrets never enter repr/str/serialization output.
- Paper and live state paths are separate fields; isolation is structural.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Side = Literal["long", "short"]
"""Direction of an order intent. Strategies cannot express anything else."""


class ExecutionMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utciso(dt: datetime | None = None) -> str:
    """Canonical ISO-8601 UTC string (millisecond precision, Z suffix)."""
    dt = dt or utcnow()
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def intent_digest(intent: OrderIntent) -> str:
    """SHA-256 over the canonical JSON of an intent's risk-relevant fields.

    Used by the approval queue: an approval is bound to THIS digest, so any
    modified intent cannot reuse an old approval.
    """
    payload = {
        "symbol": intent.symbol,
        "side": intent.side,
        "notional": str(Decimal(str(intent.notional))),
        "leverage": str(Decimal(str(intent.leverage))),
        "strategy": intent.strategy,
        "reduce_only": intent.reduce_only,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class RiskLimits(BaseModel):
    """Hard risk caps. 50x is a strict maximum, not an operating target."""

    model_config = ConfigDict(frozen=True)

    max_leverage: Annotated[float, Field(gt=0, le=50)] = 3
    max_risk_per_trade: Annotated[float, Field(gt=0, le=0.005)] = 0.0025
    max_gross_notional: Annotated[float, Field(gt=0)] = 500.0
    max_net_notional: Annotated[float, Field(gt=0)] = 250.0
    max_positions: Annotated[int, Field(gt=0, le=20)] = 4
    max_symbol_notional: Annotated[float, Field(gt=0)] = 200.0
    max_cluster_notional: Annotated[float, Field(gt=0)] = 350.0
    max_drawdown: Annotated[float, Field(gt=0, le=1)] = 0.10
    max_daily_loss: Annotated[float, Field(gt=0, le=0.1)] = 0.03
    margin_reserve: Annotated[float, Field(gt=0, le=0.5)] = 0.10
    min_stop_distance: Annotated[float, Field(gt=0, le=0.2)] = 0.01

    def projected_gross(self, existing_gross: float, added: float) -> float:
        return existing_gross + added


class OrderIntent(BaseModel):
    """An immutable *intent* emitted by a strategy.

    Strategies produce intents only — they cannot place, sign, or route
    orders. Only the risk governor authorizes an intent, and only the
    execution layer (or the approval service for live) acts on it.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1)
    side: Side
    notional: Annotated[float, Field(gt=0)]
    leverage: Annotated[float, Field(gt=0, le=50)] = 1.0
    strategy: str = Field(min_length=1)
    reduce_only: bool = False
    created_at: datetime = Field(default_factory=utcnow)

    def digest(self) -> str:
        return intent_digest(self)


class RiskDecision(BaseModel):
    """The governor's verdict on an intent. Immutable once issued."""

    model_config = ConfigDict(frozen=True)

    intent_digest: str
    accepted: bool
    reason: str = ""
    quantity: Annotated[float, Field(ge=0)] = 0.0
    stop_distance: Annotated[float, Field(gt=0)] = 0.01
    leverage: Annotated[float, Field(gt=0, le=50)] = 1.0
    checks: dict[str, str] = Field(default_factory=dict)
    decided_at: datetime = Field(default_factory=utcnow)

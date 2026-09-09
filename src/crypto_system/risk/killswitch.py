"""Latching kill switch (plan MD Task 5).

Halts latch: once tripped, no risk can be opened until a human acknowledges
with an operator identity. This covers loss/drawdown halts, data faults,
ledger faults, reconciliation mismatches, and order anomalies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crypto_system.models import utciso


@dataclass
class HaltEvent:
    reason_code: str
    detail: str
    at: str


@dataclass
class KillSwitch:
    """Fail-closed circuit breaker. Latched state survives until acked."""

    events: list[HaltEvent] = field(default_factory=list)
    acknowledgements: list[dict[str, str]] = field(default_factory=list)

    @property
    def latched(self) -> bool:
        return bool(self.events)

    @property
    def reasons(self) -> list[str]:
        return [f"{e.reason_code}: {e.detail}" for e in self.events]

    def trip(self, reason_code: str, detail: str) -> None:
        """Idempotent: tripping the same code twice keeps one event."""
        if any(e.reason_code == reason_code for e in self.events):
            return
        self.events.append(
            HaltEvent(reason_code=reason_code, detail=detail, at=utciso())
        )

    def acknowledge(self, reason_code: str, *, operator: str) -> None:
        """Clear a tripped halt. Requires a non-empty operator identity."""
        if not operator or not operator.strip():
            raise ValueError("operator identity required to acknowledge a halt")
        before = len(self.events)
        self.events = [e for e in self.events if e.reason_code != reason_code]
        if len(self.events) < before:
            self.acknowledgements.append(
                {
                    "reason_code": reason_code,
                    "operator": operator,
                    "at": utciso(),
                }
            )

    def can_open_risk(self) -> bool:
        return not self.latched

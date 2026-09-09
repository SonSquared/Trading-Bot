"""Human approval queue (plan MD Task 8).

Every live order requires an approval record that is:
- bound to an immutable intent digest (change the intent -> new approval),
- created with the exact risk decision to execute (no re-sizing later),
- expiring (TTL), single-use, and tied to a named human operator.

Approvals are the ONLY route from an intent to a live order, and the
queue never emails/pushes/urgently begs: it is a calm, auditable list.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from crypto_system.models import OrderIntent, RiskDecision, utcnow


class ApprovalExpired(RuntimeError):
    """The approval existed but its TTL elapsed."""


@dataclass(frozen=True)
class Approval:
    id: str
    intent: OrderIntent
    intent_digest: str
    decision: RiskDecision
    created_at: datetime
    ttl_seconds: float
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    consumed_at: Optional[datetime] = None

    @property
    def expired(self) -> bool:
        return utcnow() > self.created_at + timedelta(seconds=self.ttl_seconds)

    @property
    def usable(self) -> bool:
        return (
            self.approved_by is not None
            and self.consumed_at is None
            and not self.expired
        )


@dataclass
class ApprovalQueue:
    items: list[Approval] = field(default_factory=list)

    def create(
        self,
        intent: OrderIntent,
        decision: RiskDecision,
        *,
        operator_pending: bool = True,
        ttl_seconds: float = 900.0,
    ) -> Approval:
        if decision.intent_digest != intent.digest():
            raise ValueError("decision does not match the intent being approved")
        approval = Approval(
            id=f"APR-{uuid.uuid4().hex[:10]}",
            intent=intent,
            intent_digest=intent.digest(),
            decision=decision,
            created_at=utcnow(),
            ttl_seconds=ttl_seconds,
        )
        self.items.append(approval)
        return approval

    def approve(self, approval_id: str, operator: str, intent_digest: str) -> bool:
        """A human with an identity approves THIS digest. Returns success."""
        if not operator or not operator.strip():
            return False
        approval = self._find(approval_id)
        if approval is None or approval.intent_digest != intent_digest:
            return False
        if approval.expired or approval.approved_by is not None:
            return False
        self.items = [
            (
                Approval(
                    id=a.id,
                    intent=a.intent,
                    intent_digest=a.intent_digest,
                    decision=a.decision,
                    created_at=a.created_at,
                    ttl_seconds=a.ttl_seconds,
                    approved_by=operator,
                    approved_at=utcnow(),
                    consumed_at=None,
                )
                if a.id == approval_id
                else a
            )
            for a in self.items
        ]
        return True

    def get_usable(self, approval_id: str, intent_digest: str) -> Approval:
        approval = self._find(approval_id)
        if approval is None:
            raise KeyError(f"unknown approval {approval_id}")
        if approval.consumed_at is not None:
            raise RuntimeError("approval already used/consumed")
        if approval.expired:
            raise ApprovalExpired(f"approval {approval_id} expired")
        if approval.approved_by is None:
            raise RuntimeError("approval not yet approved by an operator")
        if approval.intent_digest != intent_digest:
            raise RuntimeError("approval digest does not match the intent")
        return approval

    def consume(self, approval_id: str) -> None:
        self.items = [
            (
                Approval(
                    id=a.id,
                    intent=a.intent,
                    intent_digest=a.intent_digest,
                    decision=a.decision,
                    created_at=a.created_at,
                    ttl_seconds=a.ttl_seconds,
                    approved_by=a.approved_by,
                    approved_at=a.approved_at,
                    consumed_at=utcnow(),
                )
                if a.id == approval_id
                else a
            )
            for a in self.items
        ]

    def pending(self) -> list[Approval]:
        return [a for a in self.items if a.approved_by is None and not a.expired]

    def _find(self, approval_id: str) -> Optional[Approval]:
        return next((a for a in self.items if a.id == approval_id), None)

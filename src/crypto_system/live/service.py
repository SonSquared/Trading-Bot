"""Live execution service (plan MD Task 8).

Guard order (checked in this exact sequence — the cheapest and most
systemic guards first):

1. mode gate: service cannot exist unless mode=live AND live_enabled
   (enforced in the constructor),
2. killswitch: a latched halt blocks everything,
3. ledger: a failed verification blocks everything,
4. approval: must exist, match the intent digest, be approved by a named
   operator, unexpired, and unused,
5. reconciliation: local vs exchange positions must agree,
6. submit with a client order id; every outcome is ledgered.

On a timeout AFTER submission the order MAY be live exchange-side, so the
service latches an order_ambiguous halt and NEVER retries — recovery
happens through reconciliation plus human acknowledgement.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol

from crypto_system.audit.ledger import Ledger
from crypto_system.config import Settings
from crypto_system.execution.reconcile import reconcile_positions
from crypto_system.live.approval import ApprovalQueue
from crypto_system.models import ExecutionMode, OrderIntent
from crypto_system.risk.killswitch import KillSwitch


class TransportTimeout(RuntimeError):
    """The submission outcome is unknown — treat as ambiguous."""


class LiveTransport(Protocol):
    def submit(self, intent: OrderIntent, client_order_id: str) -> dict[str, Any]:
        ...

    def fetch_positions(self) -> dict[str, dict[str, float]]:
        ...


class LiveExecutionService:
    def __init__(
        self,
        *,
        settings: Settings,
        ledger: Ledger,
        transport: LiveTransport,
        killswitch: KillSwitch,
        approvals: ApprovalQueue | None = None,
    ) -> None:
        if settings.mode is not ExecutionMode.LIVE or not settings.live_enabled:
            raise RuntimeError(
                "LiveExecutionService requires mode='live' with live_enabled=True; "
                "it must be unconstructible in paper deployments"
            )
        self.settings = settings
        self.ledger = ledger
        self.transport = transport
        self.killswitch = killswitch
        self.approvals = approvals or ApprovalQueue()

    # -- guards -------------------------------------------------------------

    def _system_healthy(self) -> None:
        if self.killswitch.latched:
            raise RuntimeError(f"halt latched: {'; '.join(self.killswitch.reasons)}")
        verification = self.ledger.verify()
        if not verification.valid:
            self.killswitch.trip("ledger_fault", verification.error)
            raise RuntimeError(f"ledger verification failed: {verification.error}")

    def _pre_submit_reconciliation(self) -> None:
        # The service holds no open local book yet; whatever the exchange
        # reports must be empty, or a human reconciles first.
        local: dict[str, dict[str, float | str]] = {}
        reconciliation = reconcile_positions(local, self.transport.fetch_positions())
        if not reconciliation.ok:
            self.killswitch.trip("reconciliation_mismatch", reconciliation.detail)
            raise RuntimeError(
                f"pre-submit reconciliation failed: {reconciliation.detail}"
            )

    # -- execution ----------------------------------------------------------

    def execute(self, approval_id: str) -> dict[str, Any]:
        self._system_healthy()
        approval = self.approvals.get_usable(approval_id, self._digest_of(approval_id))
        intent: OrderIntent = approval.intent
        self._pre_submit_reconciliation()

        client_order_id = f"CS-{uuid.uuid4().hex[:16]}"
        self.ledger.append(
            {
                "type": "ORDER_SUBMIT",
                "live": True,
                "client_order_id": client_order_id,
                "digest": intent.digest(),
                "symbol": intent.symbol,
                "side": intent.side,
                "qty": approval.decision.quantity,
            }
        )
        try:
            ack = self.transport.submit(intent, client_order_id)
        except TransportTimeout:
            # The order MAY exist exchange-side. Never retry blindly.
            self.killswitch.trip(
                "order_ambiguous",
                f"timeout after submit for {client_order_id}; "
                f"reconcile and acknowledge before any new order",
            )
            self.ledger.append(
                {
                    "type": "ORDER_AMBIGUOUS",
                    "live": True,
                    "client_order_id": client_order_id,
                }
            )
            raise
        self.ledger.append(
            {
                "type": "ORDER_ACK",
                "live": True,
                "client_order_id": client_order_id,
                "exchange_order_id": str(ack.get("order_id", "")),
                "status": str(ack.get("status", "")),
            }
        )
        self.approvals.consume(approval_id)
        return ack

    # -- helpers ------------------------------------------------------------

    def _digest_of(self, approval_id: str) -> str:
        record = next(
            (a for a in self.approvals.items if a.id == approval_id), None
        )
        if record is None:
            raise RuntimeError(f"no approval record {approval_id}")
        return record.intent_digest

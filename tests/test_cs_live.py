"""Task 8 (plan MD): human-approved, reconciled Binance live adapter.

Pinned here (all transport-mocked, zero network):
- approvals bind to an immutable intent digest: a changed intent cannot
  reuse an old approval, approvals expire, approvals are single-use,
- the live service refuses to act without explicit live enablement, with a
  latched killswitch, on a failed ledger verify, or without an approved
  unexpired approval,
- a timeout AFTER submission latches an order-ambiguous halt and NEVER
  retries (submit_count stays 1),
- reconciliation detects local/exchange position divergence and latches a
  halt on mismatch.
"""

from __future__ import annotations

import pytest

from crypto_system.audit.ledger import Ledger
from crypto_system.config import Settings
from crypto_system.execution.reconcile import reconcile_positions
from crypto_system.live.approval import ApprovalExpired, ApprovalQueue
from crypto_system.live.service import LiveExecutionService, TransportTimeout
from crypto_system.models import ExecutionMode, OrderIntent, RiskDecision


def _intent(notional: float = 1000.0) -> OrderIntent:
    return OrderIntent(
        symbol="BTCUSDT", side="long", notional=notional, strategy="trend"
    )


def _decision(intent: OrderIntent) -> RiskDecision:
    return RiskDecision(
        intent_digest=intent.digest(), accepted=True, quantity=0.01, reason="ok"
    )


def _live_settings(tmp_path) -> Settings:
    return Settings(
        mode=ExecutionMode.LIVE,
        live_enabled=True,
        paper_state_dir=tmp_path / "paper",
        live_state_dir=tmp_path / "live",
    )


def _paper_settings() -> Settings:
    return Settings()  # mode=paper, live_enabled=False


class FakeTransport:
    """Mock exchange transport. Never touches a network."""

    def __init__(self) -> None:
        self.submit_count = 0
        self.raise_timeout_after_send = False
        self.positions: dict[str, dict[str, float]] = {}
        self.last_order: dict | None = None

    def submit(self, intent: OrderIntent, client_order_id: str) -> dict:
        self.submit_count += 1
        if self.raise_timeout_after_send:
            # The order WAS sent (count incremented) but the ack was lost.
            raise TransportTimeout("ack lost after submission")
        self.last_order = {"intent": intent, "client_order_id": client_order_id}
        return {
            "order_id": "EXCH-1",
            "client_order_id": client_order_id,
            "status": "FILLED",
            "qty": intent.reduce_only and 0 or 0.01,
        }

    def fetch_positions(self) -> dict[str, dict[str, float]]:
        return self.positions


def _service(tmp_path, *, settings: Settings | None = None):
    settings = settings or _live_settings(tmp_path)
    ledger = Ledger(tmp_path / "live" / "live.jsonl", mode=ExecutionMode.LIVE)
    queue = ApprovalQueue()
    transport = FakeTransport()
    from crypto_system.risk.killswitch import KillSwitch

    service = LiveExecutionService(
        settings=settings, ledger=ledger, transport=transport,
        killswitch=KillSwitch(), approvals=queue,
    )
    return service, queue, transport, ledger


class TestApprovalQueue:
    def test_changed_intent_cannot_use_old_approval(self):
        queue = ApprovalQueue()
        intent_a = _intent(notional=1000.0)
        approval = queue.create(intent_a, _decision(intent_a), operator_pending=True)
        intent_b = _intent(notional=2000.0)
        assert not queue.approve(approval.id, "operator-1", intent_b.digest())

    def test_matching_intent_approves(self):
        queue = ApprovalQueue()
        intent = _intent()
        approval = queue.create(intent, _decision(intent), operator_pending=True)
        assert queue.approve(approval.id, "operator-1", intent.digest())

    def test_approval_expires(self):
        queue = ApprovalQueue()
        intent = _intent()
        approval = queue.create(
            intent, _decision(intent), operator_pending=True,
            ttl_seconds=-1,  # already expired
        )
        assert not queue.approve(approval.id, "operator-1", intent.digest())
        with pytest.raises(ApprovalExpired):
            queue.get_usable(approval.id, intent.digest())

    def test_approval_is_single_use(self):
        queue = ApprovalQueue()
        intent = _intent()
        approval = queue.create(intent, _decision(intent), operator_pending=True)
        assert queue.approve(approval.id, "op-1", intent.digest())
        first = queue.get_usable(approval.id, intent.digest())
        queue.consume(first.id)
        with pytest.raises(Exception, match="used|consumed"):
            queue.get_usable(approval.id, intent.digest())

    def test_approve_requires_operator_identity(self):
        queue = ApprovalQueue()
        intent = _intent()
        approval = queue.create(intent, _decision(intent), operator_pending=True)
        assert not queue.approve(approval.id, "", intent.digest())


class TestLiveService:
    def test_refuses_in_paper_mode(self, tmp_path):
        # The service is UNCONSTRUCTIBLE in paper mode — the strongest form
        # of "no live order can originate from a paper deployment".
        with pytest.raises(RuntimeError, match="live"):
            _service(tmp_path, settings=_paper_settings())

    def test_happy_path_mocked(self, tmp_path):
        service, queue, transport, ledger = _service(tmp_path)
        intent = _intent()
        approval = queue.create(intent, _decision(intent), operator_pending=True)
        queue.approve(approval.id, "op-1", intent.digest())
        ack = service.execute(approval.id)
        assert ack["client_order_id"].startswith("CS-")
        assert transport.submit_count == 1
        kinds = [e["payload"]["type"] for e in ledger.entries()]
        assert "ORDER_SUBMIT" in kinds and "ORDER_ACK" in kinds

    def test_unapproved_order_rejected(self, tmp_path):
        service, queue, transport, _ = _service(tmp_path)
        intent = _intent()
        approval = queue.create(intent, _decision(intent), operator_pending=True)
        # NOT approved
        with pytest.raises(RuntimeError, match="approv"):
            service.execute(approval.id)
        assert transport.submit_count == 0

    def test_timeout_after_send_latches_halt_without_retry(self, tmp_path):
        service, queue, transport, _ = _service(tmp_path)
        intent = _intent()
        approval = queue.create(intent, _decision(intent), operator_pending=True)
        queue.approve(approval.id, "op-1", intent.digest())
        transport.raise_timeout_after_send = True
        with pytest.raises(TransportTimeout):
            service.execute(approval.id)
        assert not service.killswitch.can_open_risk()
        assert transport.submit_count == 1  # never retried

    def test_latched_killswitch_blocks_execution(self, tmp_path):
        service, queue, transport, _ = _service(tmp_path)
        service.killswitch.trip("order_ambiguous", "prior anomaly")
        intent = _intent()
        approval = queue.create(intent, _decision(intent), operator_pending=True)
        queue.approve(approval.id, "op-1", intent.digest())
        with pytest.raises(RuntimeError, match="halt"):
            service.execute(approval.id)
        assert transport.submit_count == 0

    def test_ledger_verify_failure_blocks_execution(self, tmp_path):
        service, queue, transport, ledger = _service(tmp_path)
        intent = _intent()
        approval = queue.create(intent, _decision(intent), operator_pending=True)
        queue.approve(approval.id, "op-1", intent.digest())
        ledger.tamper_for_test(0, {"tampered": True}) if ledger.entries() else None
        # Corrupt by writing garbage directly
        ledger.path.write_text("garbage\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="ledger"):
            service.execute(approval.id)
        assert transport.submit_count == 0


class TestReconcile:
    def test_matching_positions_pass(self):
        local = {"BTCUSDT": {"qty": 0.5, "side": "long"}}
        result = reconcile_positions(local, {"BTCUSDT": {"qty": 0.5, "side": "long"}})
        assert result.ok

    def test_divergence_detected(self):
        local = {"BTCUSDT": {"qty": 0.5, "side": "long"}}
        result = reconcile_positions(local, {"BTCUSDT": {"qty": 0.4, "side": "long"}})
        assert not result.ok
        assert "BTCUSDT" in result.detail

    def test_missing_exchange_position_detected(self):
        result = reconcile_positions({"ETHUSDT": {"qty": 2.0, "side": "short"}}, {})
        assert not result.ok

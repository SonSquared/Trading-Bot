"""Task 2 (plan MD): auditable event ledger with hash chaining and recovery.

Pinned here:
- appending mutates only the tail; any historical tampering breaks verify(),
- the hash chain uses canonical JSON + SHA-256 over (prev_hash + payload),
- replay() rebuilds positions/equity from records alone (state is a cache),
- secrets are redacted at append time (belt and braces with SecretStr),
- live records can never be written to the paper ledger path and vice versa.
"""

from __future__ import annotations

import json

import pytest

from crypto_system.audit.ledger import Ledger, LedgerCorrupted, redact
from crypto_system.models import ExecutionMode


def _entries_from(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestHashChain:
    def test_append_writes_jsonl_with_chain(self, tmp_path):
        ledger = Ledger(tmp_path / "paper.jsonl", mode=ExecutionMode.PAPER)
        ledger.append({"type": "fill", "symbol": "BTCUSDT", "qty": 0.1})
        entries = _entries_from(tmp_path / "paper.jsonl")
        assert len(entries) == 1
        assert entries[0]["payload"]["symbol"] == "BTCUSDT"
        assert entries[0]["seq"] == 0
        assert len(entries[0]["hash"]) == 64 and entries[0]["prev_hash"] == "0" * 64

    def test_modified_historical_event_fails_verification(self, tmp_path):
        ledger = Ledger(tmp_path / "paper.jsonl", mode=ExecutionMode.PAPER)
        ledger.append({"type": "fill", "symbol": "BTCUSDT", "qty": 0.1})
        ledger.append({"type": "fill", "symbol": "BTCUSDT", "qty": 0.2})
        ledger.tamper_for_test(0, {"qty": 0.2})
        assert not ledger.verify().valid

    def test_verify_detects_deletion_and_reordering(self, tmp_path):
        ledger = Ledger(tmp_path / "paper.jsonl", mode=ExecutionMode.PAPER)
        for i in range(4):
            ledger.append({"type": "fill", "qty": float(i)})
        path = tmp_path / "paper.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        # Deletion of a MIDDLE record (a truncated tail is a crash artifact,
        # not tampering — hash chains cannot see a removed last line):
        path.write_text("\n".join(lines[:1] + lines[2:]) + "\n", encoding="utf-8")
        assert not ledger.verify().valid
        # Reordering
        path.write_text(
            "\n".join([lines[0], lines[2], lines[1], lines[3]]) + "\n",
            encoding="utf-8",
        )
        assert not ledger.verify().valid

    def test_verify_detects_truncated_tail(self, tmp_path):
        ledger = Ledger(tmp_path / "paper.jsonl", mode=ExecutionMode.PAPER)
        ledger.append({"a": 1})
        ledger.append({"b": 2})
        path = tmp_path / "paper.jsonl"
        # Simulate a crash mid-write: last line without newline and invalid JSON.
        raw = path.read_bytes()
        path.write_bytes(raw.rsplit(b"\n", 1)[0] + b"\n" + b'{"payload": {"b": 2, "t')
        result = ledger.verify()
        assert result.valid is False
        # Truncation to the last valid entries is allowed as recovery.
        assert ledger.recover_to_consistent_tail() == 2

    def test_secret_redaction_at_append(self, tmp_path):
        ledger = Ledger(tmp_path / "paper.jsonl", mode=ExecutionMode.PAPER)
        ledger.append(
            {
                "type": "config",
                "api_key": "AKIA-EXAMPLE-SECRET",
                "nested": {"telegram_bot_token": "super-secret-token"},
            }
        )
        raw = (tmp_path / "paper.jsonl").read_text(encoding="utf-8")
        assert "AKIA-EXAMPLE-SECRET" not in raw
        assert "super-secret-token" not in raw
        assert "[REDACTED]" in raw

    def test_redact_helper_maps_key_names(self):
        assert redact({"api_key": "x"})["api_key"] == "[REDACTED]"
        assert redact({"password": "x"})["password"] == "[REDACTED]"
        assert redact({"normal": "x"})["normal"] == "x"


class TestReplay:
    def test_replay_rebuilds_positions_and_equity(self, tmp_path):
        ledger = Ledger(tmp_path / "paper.jsonl", mode=ExecutionMode.PAPER)
        ledger.append({"type": "OPEN", "symbol": "BTCUSDT", "side": "long",
                       "qty": 1.0, "price": 100.0, "fee": 0.05, "cash_after": 9999.95})
        ledger.append({"type": "CLOSE", "symbol": "BTCUSDT", "side": "long",
                       "qty": 1.0, "price": 110.0, "fee": 0.05, "pnl": 9.95,
                       "cash_after": 10009.90})
        state = ledger.replay()
        assert state["equity_cash"] == pytest.approx(10009.90)
        assert state["positions"] == {}

    def test_replay_tracks_open_position(self, tmp_path):
        ledger = Ledger(tmp_path / "paper.jsonl", mode=ExecutionMode.PAPER)
        ledger.append({"type": "OPEN", "symbol": "ETHUSDT", "side": "short",
                       "qty": 2.0, "price": 50.0, "fee": 0.01, "cash_after": 9999.99})
        state = ledger.replay()
        assert state["positions"]["ETHUSDT"]["side"] == "short"
        assert state["positions"]["ETHUSDT"]["qty"] == 2.0

    def test_replay_fails_on_ledger_mismatch(self, tmp_path):
        ledger = Ledger(tmp_path / "paper.jsonl", mode=ExecutionMode.PAPER)
        ledger.append({"type": "OPEN", "symbol": "X", "side": "long",
                       "qty": 1.0, "price": 10.0, "fee": 0.0, "cash_after": 9.0})
        # A CLOSE for a symbol never opened: replay must fail closed.
        ledger.append({"type": "CLOSE", "symbol": "X", "side": "long",
                       "qty": 1.0, "price": 12.0, "fee": 0.0, "pnl": 2.0,
                       "cash_after": 11.0})
        ledger.append({"type": "CLOSE", "symbol": "GHOST", "side": "long",
                       "qty": 1.0, "price": 5.0, "fee": 0.0, "pnl": -5.0,
                       "cash_after": 6.0})
        with pytest.raises(LedgerCorrupted):
            ledger.replay()

    def test_replay_detects_cash_mismatch(self, tmp_path):
        ledger = Ledger(tmp_path / "paper.jsonl", mode=ExecutionMode.PAPER)
        ledger.append({"type": "OPEN", "symbol": "X", "side": "long",
                       "qty": 1.0, "price": 10.0, "fee": 0.1, "cash_after": 500.0})
        with pytest.raises(LedgerCorrupted):
            ledger.replay(initial_cash=1000.0)


class TestPaperLiveIsolation:
    def test_live_records_rejected_on_paper_ledger(self, tmp_path):
        ledger = Ledger(tmp_path / "paper.jsonl", mode=ExecutionMode.PAPER)
        with pytest.raises(ValueError, match="live"):
            ledger.append({"type": "ORDER_SUBMIT", "live": True}, source=ExecutionMode.LIVE)

    def test_live_ledger_requires_live_mode_construction(self, tmp_path):
        paper = Ledger(tmp_path / "p.jsonl", mode=ExecutionMode.PAPER)
        live = Ledger(tmp_path / "l.jsonl", mode=ExecutionMode.LIVE)
        paper.append({"type": "OPEN"})
        live.append({"type": "ORDER_SUBMIT", "live": True})
        assert "ORDER_SUBMIT" not in (tmp_path / "p.jsonl").read_text(encoding="utf-8")

    def test_state_dir_rejects_paper_path_for_live(self, tmp_path):
        from crypto_system.audit.state import StateStore

        paper_dir = tmp_path / "paper"
        paper_dir.mkdir()
        with pytest.raises(ValueError, match="isolat"):
            StateStore(mode=ExecutionMode.LIVE, live_dir=paper_dir, paper_dir=paper_dir)

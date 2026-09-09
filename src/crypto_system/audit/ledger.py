"""Hash-chained append-only JSONL ledger (plan MD Task 2).

Properties:
- Each record = {seq, ts, prev_hash, hash, payload}; hash = SHA-256 over the
  canonical JSON of (prev_hash + payload). Any historical edit, deletion, or
  reordering breaks verification.
- Payloads are canonically serialized with sorted keys; secrets are redacted
  by key name at append time.
- Every record carries run/config/data references when provided.
- Replay rebuilds positions and cash from records alone: the ledger is the
  source of truth, any state file is a cache.
- Live records can never enter a paper-mode ledger (fail closed).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crypto_system.models import ExecutionMode, utciso

_GENESIS = "0" * 64

_SECRET_KEY_PARTS = (
    "api_key", "api_secret", "secret", "token", "password", "private_key",
    "telegram_bot_token", "binance_api_key", "binance_api_secret",
)


def redact(obj: Any) -> Any:
    """Recursively redact values whose key looks like a secret."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if any(part in key.lower() for part in _SECRET_KEY_PARTS):
                out[key] = "[REDACTED]"
            else:
                out[key] = redact(value)
        return out
    if isinstance(obj, list):
        return [redact(item) for item in obj]
    return obj


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _entry_hash(prev_hash: str, payload: Any) -> str:
    return hashlib.sha256(f"{prev_hash}{_canonical(payload)}".encode()).hexdigest()


@dataclass(frozen=True)
class VerifyResult:
    valid: bool
    entries: int
    error: str = ""


class LedgerCorrupted(RuntimeError):
    """Raised when replay encounters a ledger inconsistent with itself."""


class Ledger:
    """Append-only, hash-chained JSONL event ledger."""

    def __init__(
        self,
        path: Path | str,
        mode: ExecutionMode,
        *,
        run_id: str = "",
        config_hash: str = "",
        data_version: str = "",
    ) -> None:
        self.path = Path(path)
        self.mode = mode
        self._run_id = run_id
        self._config_hash = config_hash
        self._data_version = data_version
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tail_hash = self._scan_tail_hash()

    # -- internals ----------------------------------------------------------

    def _scan_tail_hash(self) -> str:
        if not self.path.exists():
            return _GENESIS
        last_hash = _GENESIS
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    last_hash = record["hash"]
                except (json.JSONDecodeError, KeyError):
                    break
        return last_hash

    def _refs(self) -> dict[str, str]:
        return {
            "run_id": self._run_id,
            "config_hash": self._config_hash,
            "data_version": self._data_version,
        }

    # -- public API ---------------------------------------------------------

    def append(
        self, payload: dict[str, Any], *, source: ExecutionMode | None = None
    ) -> dict[str, Any]:
        source = source or self.mode
        if source is not self.mode:
            raise ValueError(
                f"refusing to write a {source.value} record to a "
                f"{self.mode.value} ledger — paper and live never share a path"
            )
        record: dict[str, Any] = {
            "seq": self._scan_tail_seq() + 1 if self.path.exists() else 0,
            "ts": utciso(),
            "prev_hash": self._tail_hash,
            "hash": _entry_hash(self._tail_hash, payload),
            "payload": redact(payload),
            **self._refs(),
        }
        line = _canonical(record) + "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        self._tail_hash = record["hash"]
        return record

    def _scan_tail_seq(self) -> int:
        last_seq = -1
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    last_seq = json.loads(line)["seq"]
                except (json.JSONDecodeError, KeyError):
                    break
        return last_seq

    def verify(self) -> VerifyResult:
        if not self.path.exists():
            return VerifyResult(True, 0)
        prev = _GENESIS
        count = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    return VerifyResult(False, count, f"line {count}: invalid JSON: {exc}")
                if record.get("prev_hash") != prev:
                    return VerifyResult(False, count, f"line {count}: chain break")
                expected = _entry_hash(record.get("prev_hash", ""), record.get("payload"))
                if record.get("hash") != expected:
                    return VerifyResult(False, count, f"line {count}: hash mismatch")
                if record.get("seq") != count:
                    return VerifyResult(False, count, f"line {count}: sequence gap")
                prev = record["hash"]
                count += 1
        return VerifyResult(True, count)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        break
        return out

    def recover_to_consistent_tail(self) -> int:
        """Truncate a corrupted tail (crash mid-write) to the last valid entry."""
        result = self.verify()
        if result.valid:
            return result.entries
        good: list[str] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                candidate = good + [line]
                probe = "\n".join(s.strip() for s in candidate if s.strip())
                if not self._verify_lines(probe):
                    break
                good.append(line)
        with self.path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write("".join(good))
        self._tail_hash = self._scan_tail_hash()
        return len([g for g in good if g.strip()])

    def _verify_lines(self, text: str) -> bool:
        prev = _GENESIS
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return False
            if record.get("prev_hash") != prev or record.get("seq") != i:
                return False
            if record.get("hash") != _entry_hash(
                record.get("prev_hash", ""), record.get("payload")
            ):
                return False
            prev = record["hash"]
        return True

    # -- replay -------------------------------------------------------------

    def replay(self, *, initial_cash: float = 10_000.0) -> dict[str, Any]:
        """Rebuild positions and cash exclusively from ledger records."""
        cash = initial_cash
        positions: dict[str, dict[str, Any]] = {}
        for entry in self.entries():
            payload = entry.get("payload", {})
            kind = payload.get("type")
            if kind == "OPEN":
                # Perp convention: notional is margin-backed; opening costs
                # only the fee. P&L settles at CLOSE.
                symbol = payload["symbol"]
                if symbol in positions:
                    raise LedgerCorrupted(f"OPEN for already-open position {symbol}")
                positions[symbol] = {
                    "side": payload["side"],
                    "qty": payload["qty"],
                    "entry_price": payload["price"],
                }
                cash -= float(payload.get("fee", 0.0))
                if "cash_after" in payload and abs(payload["cash_after"] - cash) > 1e-6:
                    raise LedgerCorrupted(
                        f"cash mismatch at seq {entry['seq']}: "
                        f"recorded {payload['cash_after']} != reconstructed {cash}"
                    )
            elif kind == "CLOSE":
                symbol = payload["symbol"]
                if symbol not in positions:
                    raise LedgerCorrupted(f"CLOSE for unknown position {symbol}")
                pos = positions.pop(symbol)
                qty = payload["qty"]
                if abs(qty - pos["qty"]) > 1e-12:
                    raise LedgerCorrupted(f"CLOSE qty mismatch for {symbol}")
                if payload["side"] != pos["side"]:
                    raise LedgerCorrupted(f"CLOSE side mismatch for {symbol}")
                price = payload["price"]
                direction = 1.0 if pos["side"] == "long" else -1.0
                pnl = (
                    (price - pos["entry_price"]) * qty * direction
                    - float(payload.get("fee", 0.0))
                )
                if "pnl" in payload and abs(payload["pnl"] - pnl) > 1e-6:
                    raise LedgerCorrupted(
                        f"pnl mismatch at seq {entry['seq']}: "
                        f"recorded {payload['pnl']} != reconstructed {pnl}"
                    )
                cash += pnl
                if "cash_after" in payload and abs(payload["cash_after"] - cash) > 1e-6:
                    raise LedgerCorrupted(
                        f"cash mismatch at seq {entry['seq']}: "
                        f"recorded {payload['cash_after']} != reconstructed {cash}"
                    )
            elif kind in ("FUND", "FEE"):
                amount = float(payload.get("amount", 0.0))
                cash -= amount
                if "cash_after" in payload and abs(payload["cash_after"] - cash) > 1e-6:
                    raise LedgerCorrupted(f"cash mismatch at seq {entry['seq']}")
        return {"equity_cash": cash, "positions": positions}

    # -- test support -------------------------------------------------------

    def tamper_for_test(self, index: int, payload_patch: dict[str, Any]) -> None:
        """Rewrite a historical payload WITHOUT re-chaining (test helper)."""
        entries = self.entries()
        if index < 0 or index >= len(entries):
            raise IndexError(index)
        entries[index]["payload"].update(payload_patch)
        with self.path.open("w", encoding="utf-8", newline="\n") as fh:
            for entry in entries:
                fh.write(_canonical(entry) + "\n")

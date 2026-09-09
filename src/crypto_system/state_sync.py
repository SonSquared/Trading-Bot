"""Shared state synchronization (plan MD Task 7).

- ``StateLock``: file-based lease lock with a TTL. A live holder blocks any
  second acquire (fail closed); an expired lease may be taken over (a dead
  runner must not deadlock the system forever).
- ``new_run_id``: unique run identifiers stamped into ledger records.
- ``version_check``: optimistic concurrency for state files — a writer
  whose base version no longer matches the on-disk version fails closed.
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class LockHeld(RuntimeError):
    """Another live holder owns the lease."""


def new_run_id() -> str:
    """Sortable, unique run id: UTC timestamp + random suffix."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:  # pragma: no cover
        return "unknown-host"


class StateLock:
    """Lease lock: exclusive while held, self-expiring after the TTL."""

    def __init__(
        self,
        path: Path | str,
        *,
        owner: str,
        ttl_seconds: float = 300.0,
    ) -> None:
        self.path = Path(path)
        self.owner = owner
        self.ttl_seconds = ttl_seconds

    # -- helpers ------------------------------------------------------------

    def _read_lease(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
            return data
        except json.JSONDecodeError:
            return None

    def _lease_expired(self, lease: dict[str, Any]) -> bool:
        acquired = datetime.fromisoformat(lease["acquired_at"])
        return datetime.now(timezone.utc) >= acquired + timedelta(
            seconds=float(lease["ttl_seconds"])
        )

    # -- API ----------------------------------------------------------------

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lease = self._read_lease()
        if lease is not None and not self._lease_expired(lease):
            if lease.get("owner") == self.owner:
                return  # re-entrant
            raise LockHeld(
                f"state lock held by {lease.get('owner')} "
                f"(host {lease.get('host')}) since {lease.get('acquired_at')}"
            )
        payload = {
            "owner": self.owner,
            "host": _hostname(),
            "pid": os.getpid(),
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "ttl_seconds": self.ttl_seconds,
        }
        # Atomic create-only open (O_EXCL is atomic on POSIX and Windows).
        # Attempt 0: normal take. Attempt 1: expired-lease takeover.
        for attempt in range(2):
            if attempt == 1:
                self.path.unlink(missing_ok=True)
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=1)
                return
            except FileExistsError:
                lease = self._read_lease()
                if lease is not None and not self._lease_expired(lease):
                    raise LockHeld(
                        f"state lock held by {lease.get('owner')} (raced)"
                    ) from None
                # lease expired: loop once more for the takeover
        raise LockHeld("could not acquire state lock after takeover attempt")

    def release(self) -> None:
        lease = self._read_lease()
        if lease is not None and lease.get("owner") == self.owner:
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> "StateLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def version_check(path: Path | str, expected_version: int) -> int:
    """Fail closed if the on-disk version differs from the caller's base.

    Returns the new version to write. The state file carries
    ``{"version": n, ...}``; a mismatch means another writer won the race.
    """
    state_path = Path(path)
    if state_path.exists():
        current: int = int(
            json.loads(state_path.read_text(encoding="utf-8")).get("version", 0)
        )
    else:
        current = 0
    if current != expected_version:
        raise RuntimeError(
            f"state version conflict: expected {expected_version}, on disk {current}"
        )
    return current + 1

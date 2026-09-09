"""Isolated state stores for paper and live modes (plan MD Task 2).

The state file is a *cache* of the ledger. Recovery always replays the
ledger; the store exists so the hot loop does not re-read the whole file
every cycle. Paper and live stores are path-isolated by construction and
refuse to be constructed pointing at the same directory.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from crypto_system.audit.ledger import Ledger, LedgerCorrupted
from crypto_system.models import ExecutionMode


class StateStore:
    """Atomic, mode-isolated state cache over a ledger."""

    def __init__(
        self,
        *,
        mode: ExecutionMode,
        paper_dir: Path | str,
        live_dir: Path | str | None = None,
    ) -> None:
        paper = Path(paper_dir)
        live = Path(live_dir) if live_dir is not None else None
        if mode is ExecutionMode.LIVE:
            if live is None or live == paper:
                raise ValueError(
                    "live state dir must be provided and isolated from paper "
                    "(paper and live state never share a write path)"
                )
            self.dir = live
        else:
            self.dir = paper
        self.mode = mode
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{mode.value}_state.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        return data

    def save(self, state: dict[str, Any]) -> None:
        """Atomic write: temp file + os.replace, fsynced."""
        fd, tmp_name = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(state, fh, sort_keys=True, indent=1)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def recover(self, ledger: Ledger, *, initial_cash: float = 10_000.0) -> dict[str, Any]:
        """Rebuild state from the ledger (the only trusted source)."""
        result = ledger.verify()
        if not result.valid:
            raise LedgerCorrupted(f"ledger failed verification: {result.error}")
        state = ledger.replay(initial_cash=initial_cash)
        state["recovered_at_seq"] = result.entries - 1
        self.save(state)
        return state

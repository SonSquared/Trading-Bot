"""Reconciliation (plan MD Task 8): local state vs exchange truth.

Reconcile before submitting, after acking, and during recovery. Any
divergence is a halt condition, never an auto-correction: the machine does
not guess which side is right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ReconcileResult:
    ok: bool
    detail: str


def reconcile_positions(
    local: Mapping[str, Mapping[str, float | str]],
    exchange: Mapping[str, Mapping[str, float | str]],
    *,
    qty_tolerance: float = 1e-9,
) -> ReconcileResult:
    """Compare position maps; report every difference precisely."""
    problems: list[str] = []
    for symbol in sorted(set(local) | set(exchange)):
        local_pos = local.get(symbol)
        exchange_pos = exchange.get(symbol)
        if local_pos is None and exchange_pos is not None:
            problems.append(
                f"{symbol}: exchange-only position (qty={exchange_pos.get('qty')})"
            )
            continue
        if exchange_pos is None and local_pos is not None:
            problems.append(
                f"{symbol}: local-only position (qty={local_pos.get('qty')})"
            )
            continue
        if local_pos is None or exchange_pos is None:  # handled above
            continue
        if (
            abs(float(local_pos["qty"]) - float(exchange_pos["qty"]))
            > qty_tolerance
        ):
            problems.append(
                f"{symbol}: qty divergence local={local_pos['qty']} "
                f"exchange={exchange_pos['qty']}"
            )
        if str(local_pos.get("side")) != str(exchange_pos.get("side")):
            problems.append(
                f"{symbol}: side divergence local={local_pos.get('side')} "
                f"exchange={exchange_pos.get('side')}"
            )
    if problems:
        return ReconcileResult(False, "; ".join(problems))
    return ReconcileResult(True, "positions match")

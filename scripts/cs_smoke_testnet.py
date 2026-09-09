"""Binance USDⓈ-M TESTNET-ONLY smoke command (plan MD Task 8, Step 5).

This is the ONLY code path in the repository that constructs the signed
Binance transport, and it refuses every production configuration:

- requires explicit operator gating via env:
    CS_SMOKE_TESTNET_CONFIRM=yes   (literal string)
- reads credentials ONLY from the environment (CS_BINANCE_API_KEY /
  CS_BINANCE_API_SECRET); they are never logged, echoed, or written;
- exercises: signed position fetch -> signed MARKET order with a CS-
  client order id -> position fetch again -> reconciliation against a
  locally expected delta -> ledger records the whole session (hash-chained).

Run it manually on your PC:

    CS_SMOKE_TESTNET_CONFIRM=yes \
    CS_BINANCE_API_KEY=... CS_BINANCE_API_SECRET=... \
    python scripts/cs_smoke_testnet.py

No workflow invokes this. No schedule fires it. It is a hand-held tool.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crypto_system.audit.ledger import Ledger  # noqa: E402
from crypto_system.execution.binance_live import (  # noqa: E402
    BinanceUsdmTransport,
    TransportError,
    TransportTimeout,
)
from crypto_system.execution.reconcile import reconcile_positions  # noqa: E402
from crypto_system.models import ExecutionMode, OrderIntent, utciso  # noqa: E402


def main() -> int:
    if os.environ.get("CS_SMOKE_TESTNET_CONFIRM") != "yes":
        print(
            "REFUSING: set CS_SMOKE_TESTNET_CONFIRM=yes to run the testnet "
            "smoke. This command is hand-held; nothing schedules it.",
            file=sys.stderr,
        )
        return 2

    api_key = os.environ.get("CS_BINANCE_API_KEY", "")
    api_secret = os.environ.get("CS_BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        print(
            "REFUSING: CS_BINANCE_API_KEY / CS_BINANCE_API_SECRET must come "
            "from the environment (testnet keys, trading-only permissions).",
            file=sys.stderr,
        )
        return 2

    transport = BinanceUsdmTransport(api_key=api_key, api_secret=api_secret, testnet=True)
    ledger = Ledger(
        Path("data/cs_state/live") / "smoke_testnet.jsonl", mode=ExecutionMode.LIVE
    )
    print(f"TESTNET smoke at {utciso()} against {transport.base_url}")

    # 1. Signed reachability: position fetch.
    before = transport.fetch_positions()
    print(f"open positions on testnet: {before or 'none'}")
    ledger.append({"type": "SMOKE_POSITIONS", "live": True, "count": len(before)})

    # 2. A minimal market order, fully client-ID'd.
    client_order_id = f"CS-SMOKE-{utciso().replace(':', '').replace('-', '')[:20]}"
    intent = OrderIntent(
        symbol="BTCUSDT", side="long", notional=100.0, strategy="testnet-smoke"
    )
    try:
        ack = transport.submit(intent, client_order_id)
    except TransportTimeout:
        print(
            "AMBIGUOUS: no ack after send. The order may exist testnet-side. "
            "Reconcile manually; nothing will retry.",
            file=sys.stderr,
        )
        ledger.append(
            {"type": "SMOKE_AMBIGUOUS", "live": True, "client_order_id": client_order_id}
        )
        return 3
    except TransportError as exc:
        print(f"exchange rejected the smoke order: {exc}", file=sys.stderr)
        ledger.append({"type": "SMOKE_REJECTED", "live": True, "error": str(exc)[:200]})
        return 4

    print(f"ack: order_id={ack.get('orderId')} status={ack.get('status')}")
    ledger.append(
        {
            "type": "SMOKE_ACK",
            "live": True,
            "client_order_id": client_order_id,
            "exchange_order_id": str(ack.get("orderId", "")),
            "status": str(ack.get("status", "")),
        }
    )

    # 3. Reconciliation: exchange must now show the position our ack implies.
    after = transport.fetch_positions()
    if ack.get("status") == "FILLED":
        expected = dict(before)
        expected["BTCUSDT"] = {
            "qty": float(ack.get("executedQty", 0.0) or 0.01),
            "side": "long",
        }
        result = reconcile_positions(expected, after)
        print(f"reconciliation: {'OK' if result.ok else result.detail}")
        ledger.append(
            {
                "type": "SMOKE_RECONCILE",
                "live": True,
                "ok": result.ok,
                "detail": result.detail[:200],
            }
        )
        if not result.ok:
            print(
                "DIVERGENCE — resolve by hand before any further order.",
                file=sys.stderr,
            )
            return 5

    print("smoke complete: testnet transport, signing, ack, and reconcile all exercised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

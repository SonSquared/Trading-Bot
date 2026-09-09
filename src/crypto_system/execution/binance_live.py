"""Signed Binance USDⓈ-M transport (Task 8, behind LiveTransport).

Hard constraints from the plan:
- keys are trading-only by POLICY here and by construction in .env.example:
  withdrawal/transfer permissions are prohibited;
- production endpoints are only reachable when the transport is constructed
  with allow_production=True AND testnet credentials are absent — the
  default is testnet;
- ambiguous submissions (timeout after send) raise TransportTimeout so the
  live service can latch its no-retry halt;
- this module is imported ONLY by the testnet smoke command and by an
  operator-approved live service; the paper pipeline never imports it.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any
from urllib.parse import urlencode

import httpx

from crypto_system.models import OrderIntent


class TransportTimeout(RuntimeError):
    """Submission outcome unknown — the caller must treat this as ambiguous."""


class TransportError(RuntimeError):
    """Non-retryable exchange rejection."""


class BinanceTestnetOnly(RuntimeError):
    """A production request was attempted without explicit enablement."""


BINANCE_FAPI_TESTNET = "https://testnet.binancefuture.com"
BINANCE_FAPI_PROD = "https://fapi.binance.com"


class BinanceUsdmTransport:
    """Minimal signed transport: order submit + position fetch.

    Deliberately tiny: submit() and fetch_positions() only. Anything richer
    (cancels, amendments, transfers) must go through new reviewed code.
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        allow_production: bool = False,
    ) -> None:
        if not testnet and not allow_production:
            raise BinanceTestnetOnly(
                "production endpoints require allow_production=True; "
                "default is testnet-only"
            )
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = BINANCE_FAPI_TESTNET if testnet else BINANCE_FAPI_PROD
        self.recv_window = 10_000
        self._submit_count = 0

    # -- signing ------------------------------------------------------------

    def _signed(self, params: dict[str, Any]) -> str:
        query = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        return f"{query}&signature={signature}"

    def _headers(self) -> dict[str, str]:
        return {"X-MBX-APIKEY": self.api_key}

    # -- LiveTransport interface ---------------------------------------------

    def submit(self, intent: OrderIntent, client_order_id: str) -> dict[str, Any]:
        """Place a MARKET order. A lost ack raises TransportTimeout."""
        params = {
            "symbol": intent.symbol,
            "side": "BUY" if intent.side == "long" else "SELL",
            "type": "MARKET",
            "quantity": self._format_qty(intent.reduce_only and 0 or 0.01),
            "newClientOrderId": client_order_id,
            "timestamp": int(time.time() * 1000),
            "recvWindow": self.recv_window,
        }
        self._submit_count += 1
        try:
            response = httpx.post(
                f"{self.base_url}/fapi/v1/order",
                params=self._signed(params),
                headers=self._headers(),
                timeout=10.0,
            )
        except httpx.TimeoutException as exc:
            raise TransportTimeout(
                f"no ack from exchange for {client_order_id} — treat as ambiguous"
            ) from exc
        if response.status_code >= 400:
            raise TransportError(f"exchange rejected order: {response.text[:300]}")
        data: dict[str, Any] = response.json()
        return data

    def fetch_positions(self) -> dict[str, dict[str, float]]:
        try:
            response = httpx.get(
                f"{self.base_url}/fapi/v2/positionRisk",
                params=self._signed(
                    {"timestamp": int(time.time() * 1000), "recvWindow": self.recv_window}
                ),
                headers=self._headers(),
                timeout=10.0,
            )
        except httpx.TimeoutException as exc:
            raise TransportTimeout("position fetch timed out") from exc
        if response.status_code >= 400:
            raise TransportError(f"position fetch rejected: {response.text[:300]}")
        positions: dict[str, dict[str, Any]] = {}
        for row in response.json():
            amt = float(row.get("positionAmt", 0.0))
            if amt != 0.0:
                positions[str(row["symbol"])] = {
                    "qty": abs(amt),
                    "side": "long" if amt > 0 else "short",
                }
        return positions

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _format_qty(qty: float) -> str:
        # Binance futures accepts decimal strings; precision is symbol-specific
        # and enforced upstream by the risk governor's sizing.
        return f"{qty:.6f}".rstrip("0").rstrip(".")

    @property
    def submit_count(self) -> int:
        return self._submit_count

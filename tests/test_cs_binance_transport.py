"""Task 8 leftover: signed Binance transport — mocked, no network.

Pinned here:
- production endpoints are unreachable without allow_production=True,
- default (and every operator path) is TESTNET-ONLY,
- orders are signed and side-mapped correctly,
- a lost ack (timeout after send) surfaces as TransportTimeout so the live
  service latches its no-retry halt — never a blind retry,
- the paper pipeline never imports this module.
"""

from __future__ import annotations

import hashlib
import hmac
from unittest import mock
from urllib.parse import parse_qs

import pytest

from crypto_system.execution.binance_live import (
    BINANCE_FAPI_PROD,
    BINANCE_FAPI_TESTNET,
    BinanceTestnetOnly,
    BinanceUsdmTransport,
    TransportError,
    TransportTimeout,
)
from crypto_system.models import OrderIntent


def _intent(side: str = "long") -> OrderIntent:
    return OrderIntent(symbol="BTCUSDT", side=side, notional=100.0, strategy="t")


class TestEndpointGate:
    def test_production_requires_explicit_allowance(self):
        with pytest.raises(BinanceTestnetOnly):
            BinanceUsdmTransport(api_key="k", api_secret="s", testnet=False)

    def test_default_and_explicit_testnet_urls(self):
        t = BinanceUsdmTransport(api_key="k", api_secret="s")
        assert t.base_url == BINANCE_FAPI_TESTNET
        t2 = BinanceUsdmTransport(api_key="k", api_secret="s", testnet=True)
        assert t2.base_url == BINANCE_FAPI_TESTNET

    def test_production_with_allowance_points_to_prod(self):
        t = BinanceUsdmTransport(
            api_key="k", api_secret="s", testnet=False, allow_production=True
        )
        assert t.base_url == BINANCE_FAPI_PROD


class TestSigningAndSubmission:
    def test_request_is_signed_and_side_mapped(self):
        t = BinanceUsdmTransport(api_key="testkey", api_secret="testsecret")
        captured: dict = {}

        class Resp:
            status_code = 200

            def json(self):
                return {"orderId": 1, "status": "FILLED"}

        def fake_post(url, params=None, headers=None, timeout=None):
            captured["url"] = url
            captured["params"] = params
            captured["headers"] = headers
            return Resp()

        with mock.patch("crypto_system.execution.binance_live.httpx.post", fake_post):
            ack = t.submit(_intent("long"), "CS-abc")

        assert ack["status"] == "FILLED"
        assert captured["url"].startswith(BINANCE_FAPI_TESTNET)
        assert captured["headers"]["X-MBX-APIKEY"] == "testkey"

        # The transport hands httpx the signed query as a string param.
        query = captured["params"]
        parsed = parse_qs(query, keep_blank_values=True)
        assert parsed["symbol"] == ["BTCUSDT"]
        assert parsed["side"] == ["BUY"]
        assert parsed["newClientOrderId"] == ["CS-abc"]

        # HMAC-SHA256 signature verifies against the secret
        sig = parsed["signature"][0]
        payload = "&".join(
            f"{k}={v[0]}" for k, v in parsed.items() if k != "signature"
        )
        expected = hmac.new(b"testsecret", payload.encode(), hashlib.sha256).hexdigest()
        assert sig == expected

    def test_short_side_maps_to_sell(self):
        t = BinanceUsdmTransport(api_key="k", api_secret="s")
        captured: dict = {}

        class Resp:
            status_code = 200

            def json(self):
                return {"orderId": 2}

        with mock.patch(
            "crypto_system.execution.binance_live.httpx.post",
            lambda url, params=None, headers=None, timeout=None: (
                captured.update({"params": params}), Resp()
            )[-1],
        ):
            t.submit(_intent("short"), "CS-x")
        assert parse_qs(captured["params"], keep_blank_values=True)["side"] == ["SELL"]

    def test_timeout_after_send_raises_transport_timeout(self):
        t = BinanceUsdmTransport(api_key="k", api_secret="s")

        def fake_post(url, params=None, headers=None, timeout=None):
            raise __import__("httpx").TimeoutException("ack lost")

        with mock.patch("crypto_system.execution.binance_live.httpx.post", fake_post):
            with pytest.raises(TransportTimeout):
                t.submit(_intent(), "CS-x")
        assert t.submit_count == 1  # the send HAPPENED; caller must not retry

    def test_exchange_rejection_raises_transport_error(self):
        t = BinanceUsdmTransport(api_key="k", api_secret="s")

        class Resp:
            status_code = 400

            def text(self):
                return ""

            def __init__(self):
                self.text = '{"code":-2019,"msg":"Margin is insufficient."}'

        with mock.patch(
            "crypto_system.execution.binance_live.httpx.post",
            lambda url, params=None, headers=None, timeout=None: Resp(),
        ):
            with pytest.raises(TransportError):
                t.submit(_intent(), "CS-x")


class TestIsolation:
    def test_paper_pipeline_never_imports_transport(self):
        from pathlib import Path

        offenders: list[str] = []
        for path in Path("src/crypto_system").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "binance_live" in text and path.name != "__init__.py":
                if "execution" not in str(path.parent).replace("\\\\", "/"):
                    offenders.append(str(path))
        assert not offenders, f"binance_live imported outside execution layer: {offenders}"

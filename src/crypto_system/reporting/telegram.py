"""Telegram reporting (plan MD Task 7): strictly report-only.

The Telegram layer can NEVER place, modify, or approve an order — it only
sends rendered text. ``send`` is a no-op unless explicitly enabled AND not
in dry-run, so tests and quick mode never touch the network.
"""

from __future__ import annotations

from typing import Any

from crypto_system.audit.ledger import redact


class TelegramReporter:
    def __init__(
        self,
        *,
        bot_token: str | None = None,
        chat_id: str | None = None,
        enabled: bool = False,
    ) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bool(bot_token) and bool(chat_id)

    def send(self, text: str, *, dry_run: bool = True) -> bool:
        """Send text. Returns True when a send actually happened.

        Default is dry_run=True: nothing leaves the machine unless the
        caller explicitly opts in and credentials exist.
        """
        if dry_run or not self.enabled:
            return False
        import httpx  # local import: network is opt-in

        payload: dict[str, Any] = redact({"text": text})
        response = httpx.post(
            f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
            json={"chat_id": self.chat_id, "text": payload["text"]},
            timeout=10.0,
        )
        return response.status_code == 200

"""Ownership of the account's scale: which number is the account's origin,
and whether existing history was recorded at that scale.

History: the agent read ``config["paper"]["starting_equity"]`` — a key no
config ever had — and fell back to a hardcoded 10000. The configured
``bot.paper_starting_equity: 97`` was therefore dead, and the paper account ran
a $10,000 book while the user's real balance was $97. Nothing failed, which is
exactly why it survived several audits.

So the number has ONE owner and no fallback: ``configs/ai_bot.yaml`` ->
``bot.paper_starting_equity``. A config without it raises instead of silently
inventing an account size, because a silently wrong origin means every figure
the bot reports — risk per trade, drawdown, P&L — describes the wrong account.
"""

from __future__ import annotations

from typing import Any

# The account size a fresh config should start from: the user's real capital.
DEFAULT_STARTING_EQUITY = 97.0

# Dotted path, so the error message names the exact key to add.
CONFIG_KEY = "bot.paper_starting_equity"

# Two origins within this relative distance are the same account.
SAME_SCALE_TOLERANCE = 0.01


def resolve_starting_equity(config: dict[str, Any] | None) -> tuple[float, str]:
    """Read the paper account's starting equity from the config.

    Returns ``(equity, source)``. Raises when the key is absent or unusable —
    the config is the single owner of this number, so there is no default to
    fall back to.
    """
    bot_cfg = (config or {}).get("bot") or {}
    raw = bot_cfg.get("paper_starting_equity")
    if raw is None:
        raise ValueError(
            f"Missing {CONFIG_KEY} in the AI bot config — the account's "
            "starting equity must come from the config (no fallback). "
            f"For a ${DEFAULT_STARTING_EQUITY:,.0f} account set "
            f"{CONFIG_KEY}: {DEFAULT_STARTING_EQUITY:g}"
        )
    try:
        equity = float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{CONFIG_KEY} must be a number, got {raw!r}"
        ) from e
    if equity <= 0:
        raise ValueError(
            f"{CONFIG_KEY} must be positive, got {equity!r}"
        )
    return equity, CONFIG_KEY


def same_scale(
    ledger_start_equity: Any,
    configured_equity: Any,
    rel_tolerance: float = SAME_SCALE_TOLERANCE,
) -> bool:
    """True when a ledger origin and the configured origin are the same size."""
    try:
        origin = float(ledger_start_equity)
        configured = float(configured_equity)
    except (TypeError, ValueError):
        return True  # unknown origin: never accuse a ledger we cannot read
    if origin <= 0 or configured <= 0:
        return True
    return abs(origin - configured) / configured <= rel_tolerance


def legacy_scale_notice(
    ledger_start_equity: Any, configured_equity: Any
) -> str | None:
    """A one-line warning when history pre-dates a change of account size.

    Non-None means the ledger was seeded at a different scale than the config
    now names, so its P&L, win rate and drawdown describe a different account.
    The history is never rewritten — it is labelled.
    """
    if same_scale(ledger_start_equity, configured_equity):
        return None
    try:
        origin = float(ledger_start_equity)
        configured = float(configured_equity)
    except (TypeError, ValueError):
        return None
    return (
        f"LEGACY SCALE: history was seeded at ${origin:,.2f} but the "
        f"configured account is ${configured:,.2f} — these figures are "
        f"${origin:,.0f}-scale, not ${configured:,.0f}-scale."
    )

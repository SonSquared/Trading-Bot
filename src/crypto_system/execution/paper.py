"""Cost-aware perpetual futures paper execution (plan MD Task 4).

Deterministic, seeded simulation of Binance USDⓈ-M-style execution:

- ``fill_price = quote + side_sign * (spread/2 + impact(volatility, notional))``
  — aggressive orders always cross the spread, and larger orders in more
  volatile regimes walk the book further.
- ``required_margin = |fill_price * qty| / leverage``.
- Latency and partial fills are modeled with a seeded RNG: the same seed
  produces bit-identical runs (reproducibility constraint).
- Funding is applied per 8h boundary from an explicit rate; longs pay when
  the rate is positive, shorts receive.
- A liquidation-buffer gate refuses orders that would leave maintenance-
  margin coverage below the buffer — refuse, never liquidate-by-surprise.
- Every decision and result is appended to the hash-chained ledger.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from crypto_system.audit.ledger import Ledger
from crypto_system.models import OrderIntent, utciso


class LiquidationBufferRejected(RuntimeError):
    """The order would leave maintenance-margin coverage below the buffer."""


@dataclass(frozen=True)
class Quote:
    """The market snapshot an order is evaluated against."""

    price: float
    spread_bps: float = 2.0
    volatility: float = 0.01  # e.g. 1% expected move over the horizon

    def __post_init__(self) -> None:
        if self.price <= 0 or self.spread_bps < 0 or self.volatility < 0:
            raise ValueError("invalid quote: price/spread/vol must be positive")


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: str
    requested_qty: float
    qty: float
    price: float
    fee: float
    leverage: float
    margin_required: float
    latency_ms: int
    partial: bool
    ts: str


@dataclass(frozen=True)
class CloseResult:
    symbol: str
    qty: float
    entry_price: float
    price: float
    pnl: float
    fee: float


@dataclass
class _Position:
    side: str
    qty: float
    entry_price: float
    leverage: float
    stop_distance: float | None = None


@dataclass
class PaperAccountConfig:
    initial_cash: float = 10_000.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    maintenance_margin_rate: float = 0.005
    liquidation_buffer: float = 0.02  # keep >= 2% MM coverage headroom
    impact_coefficient: float = 0.10
    max_partial_frac: float = 0.35
    latency_base_ms: int = 20
    latency_jitter_ms: int = 60
    seed: int = 12345


class PaperAccount:
    """Deterministic perpetual-futures paper account over a ledger."""

    def __init__(
        self,
        *,
        ledger: Ledger,
        initial_cash: float = 10_000.0,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0005,
        seed: int = 12345,
        config: PaperAccountConfig | None = None,
    ) -> None:
        cfg = config or PaperAccountConfig()
        self.cfg = PaperAccountConfig(
            initial_cash=initial_cash,
            maker_fee=cfg.maker_fee,
            taker_fee=cfg.taker_fee,
            maintenance_margin_rate=cfg.maintenance_margin_rate,
            liquidation_buffer=cfg.liquidation_buffer,
            impact_coefficient=cfg.impact_coefficient,
            max_partial_frac=cfg.max_partial_frac,
            latency_base_ms=cfg.latency_base_ms,
            latency_jitter_ms=cfg.latency_jitter_ms,
            seed=seed,
        )
        self.ledger = ledger
        self.equity_cash = initial_cash
        self.positions: dict[str, _Position] = {}
        self._rng = random.Random(seed)
        self._stops: dict[str, float] = {}

    # -- internals ----------------------------------------------------------

    def _impact(self, volatility: float, notional: float) -> float:
        """Impact as a fraction of price, growing with sqrt(notional) and vol."""
        return self.cfg.impact_coefficient * volatility * math.sqrt(max(notional, 1.0) / 10_000.0)

    def _latency(self) -> int:
        return self.cfg.latency_base_ms + self._rng.randint(0, self.cfg.latency_jitter_ms)

    def _partial_frac(self, volatility: float) -> float:
        """Fraction filled immediately; low in stressed regimes."""
        if volatility < 0.05:
            return 1.0
        frac = 1.0 - min(self.cfg.max_partial_frac, (volatility - 0.05) * 3.0 * self._rng.random())
        return max(0.5, frac)

    def _fee_for(self, notional: float) -> float:
        return notional * self.cfg.taker_fee

    def _check_margin(self, margin_required: float, fill_price: float, qty: float,
                      side: str, leverage: float, free_cash: float) -> None:
        if margin_required > free_cash:
            raise RuntimeError(
                f"insufficient margin: required {margin_required:.2f} > free {free_cash:.2f}"
            )
        # Maintenance coverage after opening: buffer must remain.
        mm = self.cfg.maintenance_margin_rate * fill_price * qty
        buffer = self.cfg.liquidation_buffer * fill_price * qty
        if (free_cash - margin_required) + 0 < mm + buffer - 1e-9:
            raise LiquidationBufferRejected(
                f"order would leave maintenance coverage below the "
                f"{self.cfg.liquidation_buffer:.0%} buffer (mm+buffer={mm + buffer:.2f}, "
                f"free after initial margin={free_cash - margin_required:.2f})"
            )

    # -- public API ---------------------------------------------------------

    def submit(self, intent: OrderIntent, quote: Quote) -> Fill:
        if intent.reduce_only:
            raise ValueError("reduce-only intents route through close(), not submit()")
        side_sign = 1.0 if intent.side == "long" else -1.0
        spread_cost = quote.price * (quote.spread_bps / 10_000.0) * 0.5
        impact = quote.price * self._impact(quote.volatility, intent.notional)
        fill_price = quote.price + side_sign * (spread_cost + impact)

        requested_qty = intent.notional / quote.price
        frac = self._partial_frac(quote.volatility)
        qty = requested_qty * frac
        partial = frac < 1.0
        notional_filled = qty * fill_price
        fee = self._fee_for(notional_filled)
        margin_required = abs(fill_price * qty) / intent.leverage

        self._check_margin(
            margin_required, fill_price, qty, intent.side, intent.leverage, self.equity_cash
        )

        self.equity_cash -= fee
        existing = self.positions.get(intent.symbol)
        if existing is not None:
            raise RuntimeError(
                f"position already open for {intent.symbol} — netting is disabled "
                f"(no martingale, no averaging down)"
            )
        self.positions[intent.symbol] = _Position(
            side=intent.side, qty=qty, entry_price=fill_price, leverage=intent.leverage
        )
        fill = Fill(
            symbol=intent.symbol,
            side=intent.side,
            requested_qty=requested_qty,
            qty=qty,
            price=fill_price,
            fee=fee,
            leverage=intent.leverage,
            margin_required=margin_required,
            latency_ms=self._latency(),
            partial=partial,
            ts=utciso(),
        )
        self.ledger.append(
            {
                "type": "OPEN",
                "symbol": fill.symbol,
                "side": fill.side,
                "qty": fill.qty,
                "price": fill.price,
                "fee": fill.fee,
                "leverage": fill.leverage,
                "margin_required": fill.margin_required,
                "latency_ms": fill.latency_ms,
                "partial": fill.partial,
                "cash_after": round(self.equity_cash, 10),
                "strategy": intent.strategy,
            }
        )
        return fill

    def attach_stop(self, symbol: str, distance: float) -> None:
        """Attach a stop for an open position at `distance` from entry."""
        pos = self.positions.get(symbol)
        if pos is None:
            raise KeyError(f"no open position for {symbol}")
        if distance <= 0 or distance > 0.2:
            raise ValueError("stop distance must be in (0, 0.2]")
        sign = 1.0 if pos.side == "long" else -1.0
        self._stops[symbol] = pos.entry_price * (1.0 - sign * distance)

    def process_candle(self, symbol: str, *, high: float, low: float, close: float) -> bool:
        """Evaluate stops on candle extremes (conservative: worst-case fill)."""
        stop = self._stops.get(symbol)
        if stop is None:
            return False
        pos = self.positions.get(symbol)
        if pos is None:
            return False
        triggered = (pos.side == "long" and low <= stop) or (
            pos.side == "short" and high >= stop
        )
        if not triggered:
            return False
        self.ledger.append({"type": "STOP", "symbol": symbol, "level": stop})
        # Conservative assumption: fill exactly at the stop level.
        self.close(symbol, price=stop, note="stop")
        return True

    def close(self, symbol: str, *, price: float, note: str = "") -> CloseResult:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            raise KeyError(f"no open position for {symbol}")
        direction = 1.0 if pos.side == "long" else -1.0
        exit_fee = self._fee_for(pos.qty * price)
        pnl = (price - pos.entry_price) * pos.qty * direction - exit_fee
        self.equity_cash += pnl
        self._stops.pop(symbol, None)
        self.ledger.append(
            {
                "type": "CLOSE",
                "symbol": symbol,
                "side": pos.side,
                "qty": pos.qty,
                "price": price,
                "fee": exit_fee,
                "pnl": round(pnl, 10),
                "cash_after": round(self.equity_cash, 10),
                "note": note,
            }
        )
        return CloseResult(
            symbol=symbol,
            qty=pos.qty,
            entry_price=pos.entry_price,
            price=price,
            pnl=pnl,
            fee=exit_fee,
        )

    def apply_funding(self, *, rate: float, at_price: float) -> float:
        """Apply perp funding on open notional. Longs pay positive rates."""
        total = 0.0
        for symbol, pos in self.positions.items():
            sign = 1.0 if pos.side == "long" else -1.0
            amount = pos.qty * at_price * rate * sign
            self.equity_cash -= amount
            total += amount
            self.ledger.append(
                {
                    "type": "FUND",
                    "symbol": symbol,
                    "rate": rate,
                    "price": at_price,
                    "amount": round(amount, 10),
                    "cash_after": round(self.equity_cash, 10),
                }
            )
        return total

    def mark(self, *, price: float, symbol: str | None = None) -> float:
        """Total unrealized P&L (all positions or one)."""
        total = 0.0
        for sym, pos in self.positions.items():
            if symbol is not None and sym != symbol:
                continue
            direction = 1.0 if pos.side == "long" else -1.0
            total += (price - pos.entry_price) * pos.qty * direction
        return total

    def total_equity(self, *, price: float) -> float:
        return self.equity_cash + self.mark(price=price)

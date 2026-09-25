"""Pluggable decision sources: the one part of a replay that is *not* the bot.

Everything else in a replay is production code (AIAgent, PaperLedger,
_validate_decision, venue limits). This module is the seam where "what to trade"
is supplied, so the mechanics can be measured with the judgement held
deliberately apart from it.

Every source declares its own provenance and it is printed with its results:

    label   null            no edge by construction; measures costs
            mechanical      a simple declared rule; measures a *rule*
            recorded-LLM    replays decisions the model actually made
            test            fixtures

    fitted  True only if the source's numbers were chosen by looking at results.
            Nothing shipped here is fitted; if you tune one, say so here and the
            report will carry it.

A source sees exactly what a wakeup saw: the same closed-candle indicator dict
the prompt is built from, the account state, and the rules. It cannot see the
future because it is never given it — the slot context is assembled from the
historical window that ends at the cursor.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol

from trading_system.bot.ai_engine import TradeAction, TradingDecision

# Provenance labels.
LABEL_NULL = "null"
LABEL_MECHANICAL = "mechanical"
LABEL_RECORDED_LLM = "recorded-LLM"
LABEL_TEST = "test"

LABEL_CAVEATS: dict[str, str] = {
    LABEL_NULL: (
        "no edge by construction — any non-zero result is the cost structure "
        "(fees, modelled slippage), not a strategy"
    ),
    LABEL_MECHANICAL: (
        "a declared rule, not the AI's rule — says something about the mechanics "
        "and about that rule, and nothing about the model's judgement"
    ),
    LABEL_RECORDED_LLM: (
        "real recorded model decisions replayed under the mechanics; no "
        "look-ahead, but the sample is only what was logged"
    ),
    LABEL_TEST: "synthetic fixture; not evidence about anything",
}


@dataclass
class SlotContext:
    """Everything a decision source may look at, for one simulated wakeup."""

    now: datetime
    timeframe: str
    strategy: dict
    equity: float
    positions: list[dict]
    prices: dict[str, float]
    indicators: dict[str, dict]
    snapshot: dict


def open_side(slot: SlotContext, pair: str) -> str | None:
    """'long'/'short' when ``pair`` has a position, else None."""
    for pos in slot.positions:
        if pos.get("pair") == pair:
            return str(pos.get("side") or "")
    return None


def affordable_size_pct(slot: SlotContext, stop_pct: float) -> float:
    """Largest position the RULES would accept, from slot-visible information.

    Deliberately the same arithmetic the limits use and never larger, so a
    rejection in the journal means the mechanics refused something the rules
    themselves allowed — not that the source asked for an absurd number.
    """
    rules = slot.strategy.get("rules", {}) or {}
    size_cap = float(rules.get("max_position_size_pct", 30.0))
    heat_cap = float(rules.get("max_portfolio_heat_pct", 30.0))
    risk_cap = float(rules.get("max_risk_per_trade_pct", 2.0))
    heat_used = float(slot.snapshot.get("portfolio_heat_pct") or 0.0)
    room = max(0.0, heat_cap - heat_used)
    risk_room = risk_cap / (stop_pct / 100.0) if stop_pct > 0 else 0.0
    return max(0.0, min(size_cap, heat_cap, room, risk_room))


class DecisionSource(Protocol):
    """What ``AIAgent`` expects of its ``engine``."""

    name: str
    label: str
    fitted: bool
    description: str

    def reset(self) -> None: ...

    def set_context(self, slot: SlotContext) -> None: ...

    def params_dict(self) -> dict: ...

    def decide(
        self,
        market_context: str,
        portfolio_context: str,
        strategy_context: str,
        risk_context: str,
    ) -> TradingDecision: ...


class BaseSource:
    """Shared plumbing: declarations, trades, and the slot context."""

    name = "base"
    label = LABEL_TEST
    fitted = False
    description = ""
    params: dict[str, Any] = {}

    def __init__(self) -> None:
        self.slot: SlotContext | None = None
        self.calls = 0

    # -- declarations -------------------------------------------------------

    def params_dict(self) -> dict:
        return dict(self.params)

    def provenance(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "fitted": self.fitted,
            "description": self.description,
            "params": self.params_dict(),
            "caveat": LABEL_CAVEATS.get(self.label, ""),
        }

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Start a fresh run. Default: nothing to reset."""

    def set_context(self, slot: SlotContext) -> None:
        self.slot = slot

    # -- building a decision ------------------------------------------------

    def _context(self) -> SlotContext:
        if self.slot is None:
            raise RuntimeError(
                f"{self.name}: decide() called before set_context() — a source "
                "must be given the slot it is deciding for"
            )
        return self.slot

    def _entry(
        self, pair: str, side: str, size_pct: float, stop_pct: float,
        tp_pct: float, confidence: float, why: str,
    ) -> TradeAction:
        return TradeAction(
            pair=pair, side=side, size_pct=round(size_pct, 4),
            stop_loss_pct=stop_pct, take_profit_pct=tp_pct,
            confidence=confidence, reasoning=why,
        )

    def _close(self, pair: str, why: str, confidence: float = 70.0) -> TradeAction:
        return TradeAction(
            pair=pair, side="close", size_pct=0.0, stop_loss_pct=0.0,
            take_profit_pct=0.0, confidence=confidence, reasoning=why,
        )

    def _decide_with(self, actions: list[TradeAction], why: str) -> TradingDecision:
        return TradingDecision(
            actions=actions, market_outlook="n/a (mechanical)",
            risk_assessment=why, reasoning=why, model=self.name,
        )

    def decide(self, market_context: str, portfolio_context: str,
               strategy_context: str, risk_context: str) -> TradingDecision:
        self.calls += 1
        return self._decide_with([], "base source does nothing")

    # -- what the sources share --------------------------------------------

    def _geometry(self) -> tuple[float, float]:
        return float(self.params["stop_pct"]), float(self.params["tp_pct"])

    def _pairs(self) -> list[str]:
        return [p for p, ind in self._context().indicators.items() if ind.get("ok")]

    def _confidence(self) -> float:
        return float(self.params.get("confidence", 70.0))


# ---------------------------------------------------------------------------
# The null hypothesis
# ---------------------------------------------------------------------------

class NullSource(BaseSource):
    """Never trades. The zero line: it proves the harness charges nothing."""

    name = "null"
    label = LABEL_NULL
    description = "never trades; the zero-fee, zero-trade baseline"
    params: dict[str, Any] = {}


class RandomSource(BaseSource):
    """Seeded coin-flip entries with the strategy's stop/target geometry.

    The most useful baseline available, because it cannot be right for a
    reason: with a driftless price and no costs its expectancy is zero, so
    whatever comes out is the account's *cost* structure expressed in dollars.
    Direction is a seeded coin flip; holds run to the stop or the target.
    """

    name = "random"
    label = LABEL_NULL
    description = (
        "seeded 50/50 long/short entries at the strategy's stop/target "
        "geometry, held to stop or target — the cost baseline"
    )

    def __init__(self, seed: int = 7, trade_probability: float = 0.25,
                 stop_pct: float = 5.0, tp_pct: float = 7.5,
                 confidence: float = 70.0):
        super().__init__()
        self.seed = seed
        self.params = {
            "seed": seed,
            "trade_probability": trade_probability,
            "stop_pct": stop_pct,
            "tp_pct": tp_pct,
            "confidence": confidence,
        }
        self._rng = random.Random(seed)

    def reset(self) -> None:
        self._rng = random.Random(self.seed)

    def decide(self, market_context: str, portfolio_context: str,
               strategy_context: str, risk_context: str) -> TradingDecision:
        self.calls += 1
        slot = self._context()
        stop_pct, tp_pct = self._geometry()
        size_pct = affordable_size_pct(slot, stop_pct)
        if size_pct <= 0:
            return self._decide_with([], "no room under the exposure cap")
        actions: list[TradeAction] = []
        for pair in self._pairs():
            if open_side(slot, pair):
                continue
            if self._rng.random() >= float(self.params["trade_probability"]):
                continue
            side = "long" if self._rng.random() < 0.5 else "short"
            actions.append(self._entry(
                pair, side, size_pct, stop_pct, tp_pct, self._confidence(),
                f"random {side} (seed {self.seed})",
            ))
        return self._decide_with(actions, "random entries, held to stop or target")


# ---------------------------------------------------------------------------
# Declared mechanical rules (no parameters were chosen from these results)
# ---------------------------------------------------------------------------

class TrendSource(BaseSource):
    """Momentum: enter with a confirmed EMA stack in a trending market.

    The rule, the lookbacks and the ADX floor are the conventional defaults the
    prompt itself describes. Nothing here was selected by looking at the
    results — and the report says so, because a rule that *had* been selected
    this way would not be evidence of anything.
    """

    name = "trend"
    label = LABEL_MECHANICAL
    description = (
        "declared momentum: enter long/short when EMA 9/21/50 are stacked and "
        "ADX >= 20, exit when the stack flips; 5% stop / 7.5% target"
    )

    def __init__(self, stop_pct: float = 5.0, tp_pct: float = 7.5,
                 adx_min: float = 20.0, confidence: float = 70.0):
        super().__init__()
        self.params = {
            "stop_pct": stop_pct, "tp_pct": tp_pct,
            "adx_min": adx_min, "confidence": confidence,
            "entry": "EMA9>EMA21>EMA50 and ADX>=adx_min",
            "exit": "EMA stack flips",
        }

    def decide(self, market_context: str, portfolio_context: str,
               strategy_context: str, risk_context: str) -> TradingDecision:
        self.calls += 1
        slot = self._context()
        stop_pct, tp_pct = self._geometry()
        adx_min = float(self.params["adx_min"])
        size_pct = affordable_size_pct(slot, stop_pct)
        actions: list[TradeAction] = []
        for pair in self._pairs():
            ind = slot.indicators[pair]
            e9, e21, e50 = ind.get("ema_9"), ind.get("ema_21"), ind.get("ema_50")
            if None in (e9, e21, e50):
                continue
            up = e9 > e21 > e50
            down = e9 < e21 < e50
            held = open_side(slot, pair)
            if held:
                if (held == "long" and down) or (held == "short" and up):
                    actions.append(self._close(pair, "EMA stack flipped"))
                continue
            if size_pct <= 0:
                continue
            adx = ind.get("adx")
            if adx is None or adx < adx_min:
                continue
            if up:
                actions.append(self._entry(
                    pair, "long", size_pct, stop_pct, tp_pct, self._confidence(),
                    f"EMA9>EMA21>EMA50 with ADX {adx:.1f}",
                ))
            elif down:
                actions.append(self._entry(
                    pair, "short", size_pct, stop_pct, tp_pct, self._confidence(),
                    f"EMA9<EMA21<EMA50 with ADX {adx:.1f}",
                ))
        return self._decide_with(actions, "momentum entries / flip exits")


class RsiReversionSource(BaseSource):
    """Mean reversion: buy oversold, sell overbought, exit at the midline.

    The declared counterpart to :class:`TrendSource` — an opposite bias, the
    same stated thresholds (30/70/50), the same geometry, nothing fitted.
    """

    name = "rsi-reversion"
    label = LABEL_MECHANICAL
    description = (
        "declared mean reversion: buy RSI<=30, sell RSI>=70, exit when RSI "
        "crosses 50; 5% stop / 7.5% target"
    )

    def __init__(self, stop_pct: float = 5.0, tp_pct: float = 7.5,
                 oversold: float = 30.0, overbought: float = 70.0,
                 midline: float = 50.0, confidence: float = 70.0):
        super().__init__()
        self.params = {
            "stop_pct": stop_pct, "tp_pct": tp_pct,
            "oversold": oversold, "overbought": overbought,
            "midline": midline, "confidence": confidence,
            "entry": "RSI<=oversold long / RSI>=overbought short",
            "exit": "RSI crosses midline",
        }

    def decide(self, market_context: str, portfolio_context: str,
               strategy_context: str, risk_context: str) -> TradingDecision:
        self.calls += 1
        slot = self._context()
        stop_pct, tp_pct = self._geometry()
        oversold = float(self.params["oversold"])
        overbought = float(self.params["overbought"])
        midline = float(self.params["midline"])
        size_pct = affordable_size_pct(slot, stop_pct)
        actions: list[TradeAction] = []
        for pair in self._pairs():
            rsi = slot.indicators[pair].get("rsi_14")
            if rsi is None:
                continue
            held = open_side(slot, pair)
            if held:
                if (held == "long" and rsi >= midline) or (
                    held == "short" and rsi <= midline
                ):
                    actions.append(self._close(pair, f"RSI back to {midline:g}"))
                continue
            if size_pct <= 0:
                continue
            if rsi <= oversold:
                actions.append(self._entry(
                    pair, "long", size_pct, stop_pct, tp_pct, self._confidence(),
                    f"RSI {rsi:.1f} oversold",
                ))
            elif rsi >= overbought:
                actions.append(self._entry(
                    pair, "short", size_pct, stop_pct, tp_pct, self._confidence(),
                    f"RSI {rsi:.1f} overbought",
                ))
        return self._decide_with(actions, "mean-reversion entries / midline exits")


# ---------------------------------------------------------------------------
# The model's own calls, replayed
# ---------------------------------------------------------------------------

class JournalSource(BaseSource):
    """Replays decisions the model ACTUALLY made — the only honest LLM replay.

    Each record is ``{"timestamp": iso, "actions": [{pair, side, size_pct,
    stop_loss_pct, take_profit_pct, confidence, reasoning}]}``. A slot with no
    record inside ``tolerance_minutes`` produces no trade (the model did not
    call then), nothing is interpolated and nothing is re-asked. There is no
    look-ahead because the actions were chosen before the outcome was known.

    A record with the key present but an empty list is a real decision to do
    nothing and counts as replayable; a record with the key ABSENT was never
    recorded in a replayable form (older journals stored only the *count* of
    requested actions), and is counted separately so "no trades" is never
    silently read as "the model sat flat".

    What it cannot fix: how many calls were recorded, and whether the model saw
    the same inputs this replay does.
    """

    name = "journal"
    label = LABEL_RECORDED_LLM
    description = (
        "replays recorded model decisions from a JSONL log at the slot they "
        "were made; unmatched slots do nothing"
    )

    ACTION_KEYS = ("actions", "actions_proposed")

    def __init__(self, records: list[dict], tolerance_minutes: float = 20.0):
        super().__init__()
        self.records = sorted(records, key=lambda r: str(r.get("timestamp", "")))
        self.tolerance_minutes = tolerance_minutes
        self.params = {
            "records": len(self.records),
            "tolerance_minutes": tolerance_minutes,
        }
        self.matched_slots = 0
        self.unmatched_slots = 0
        self.replayable_slots = 0
        self.records_without_actions = 0

    def reset(self) -> None:
        self.matched_slots = 0
        self.unmatched_slots = 0
        self.replayable_slots = 0
        self.records_without_actions = 0

    def _match(self, now: datetime) -> dict | None:
        best, best_gap = None, None
        for rec in self.records:
            ts = _parse_ts(rec.get("timestamp"))
            if ts is None:
                continue
            gap = abs((now - ts).total_seconds()) / 60.0
            if gap <= self.tolerance_minutes and (best_gap is None or gap < best_gap):
                best, best_gap = rec, gap
        return best

    def decide(self, market_context: str, portfolio_context: str,
               strategy_context: str, risk_context: str) -> TradingDecision:
        self.calls += 1
        slot = self._context()
        rec = self._match(slot.now)
        if rec is None:
            self.unmatched_slots += 1
            return self._decide_with([], "no recorded decision at this slot")
        self.matched_slots += 1
        # "actions" is this source's own format; "actions_proposed" is what the
        # agent's journal writes, so a state-branch journal is replayable as-is.
        if any(key in rec for key in self.ACTION_KEYS):
            self.replayable_slots += 1
        else:
            # e.g. a pre-2026-09 journal row, which stored `actions_requested: 1`
            # and no readable action. Counted, not guessed at.
            self.records_without_actions += 1
        actions = []
        for raw in (
            rec.get("actions") or rec.get("actions_proposed") or []
        ):
            if not isinstance(raw, dict):
                continue
            try:
                actions.append(TradeAction(
                    pair=str(raw.get("pair", "")),
                    side=str(raw.get("side", "")),
                    size_pct=float(raw.get("size_pct", 0) or 0),
                    stop_loss_pct=float(raw.get("stop_loss_pct", 0) or 0),
                    take_profit_pct=float(raw.get("take_profit_pct", 0) or 0),
                    confidence=float(raw.get("confidence", 0) or 0),
                    reasoning=str(raw.get("reasoning", "recorded")),
                ))
            except (TypeError, ValueError):
                continue
        return TradingDecision(
            actions=actions,
            market_outlook=str(rec.get("market_outlook", "recorded")),
            risk_assessment=str(rec.get("risk_assessment", "recorded")),
            reasoning=str(rec.get("reasoning", "recorded decision")),
            model=str(rec.get("model", "recorded")),
        )


def _parse_ts(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def load_journal_records(path: str) -> list[dict]:
    """Read a decisions JSONL. Accepts an assistant/journal-shaped file."""
    import json
    from pathlib import Path

    records: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("timestamp"):
            records.append(row)
    if not records:
        raise ValueError(
            f"{path} has no usable decision records. Expected one JSON object "
            'per line with a "timestamp" and "actions".'
        )
    return records


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class ScriptedSource(BaseSource):
    """Returns whatever a test hands it. Not evidence about anything."""

    name = "scripted"
    label = LABEL_TEST
    description = "test fixture: returns a supplied decision"

    def __init__(self, decisions: list[TradingDecision] | None = None,
                 builder: Callable[[SlotContext], TradingDecision] | None = None):
        super().__init__()
        self.decisions = list(decisions or [])
        self.builder = builder
        self.params = {"provided": len(self.decisions), "builder": bool(builder)}

    def decide(self, market_context: str, portfolio_context: str,
               strategy_context: str, risk_context: str) -> TradingDecision:
        self.calls += 1
        if self.builder is not None:
            return self.builder(self._context())
        if self.decisions:
            return self.decisions.pop(0)
        return self._decide_with([], "scripted: nothing left")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_source(name: str, **params: Any) -> BaseSource:
    """Instantiate a source by name (the CLI's ``--source``)."""
    builders: dict[str, Callable[..., BaseSource]] = {
        "null": NullSource,
        "random": RandomSource,
        "trend": TrendSource,
        "rsi-reversion": RsiReversionSource,
    }
    if name == "journal":
        records = params.pop("records", None)
        return JournalSource(records=records or [], **params)
    if name == "scripted":
        return ScriptedSource(**params)
    if name not in builders:
        raise ValueError(
            f"Unknown source {name!r}. Known: "
            f"{', '.join(sorted(list(builders) + ['journal', 'scripted']))}"
        )
    return builders[name](**params)


def source_catalog() -> list[dict]:
    """Provenance for every source, for the report and the CLI help."""
    out = []
    for name in ("null", "random", "trend", "rsi-reversion"):
        src = build_source(name)
        out.append(src.provenance())
    out.append({
        "name": "journal", "label": LABEL_RECORDED_LLM, "fitted": False,
        "description": JournalSource.description, "params": {},
        "caveat": LABEL_CAVEATS[LABEL_RECORDED_LLM],
    })
    return out

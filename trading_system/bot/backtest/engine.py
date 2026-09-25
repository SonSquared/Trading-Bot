"""Drive the REAL ``AIAgent`` wakeup loop over a historical timeline.

The central design choice: this module does not re-implement the strategy. It
constructs a production ``AIAgent`` in paper mode with a simulated clock and a
historical exchange, then calls the production ``run_wakeup()`` once per
schedule slot. Validation, sizing, venue filters, fee accounting, stop/target
geometry, the drawdown halt and the journal are therefore the *same code* the
bot runs — a replay cannot quietly drift from production, because a change to
production changes the replay.

What the engine adds on top (and labels as such):

  * the timeline — which slots exist, and what "now" is for each one
  * window orchestration — one continuous run plus independent fresh-$97 runs
    per walk-forward fold and per declared market regime
  * an optional intrabar trigger model, because the bot's paper mode only
    checks stops at slot prices while a real exchange fills them intrabar

An indicator memo is installed for the duration of a suite. It is behaviour-
preserving (``compute_indicators`` is a pure function of its frame) and it
matters: the first pass costs ~15 ms per pair per slot, and without the memo
every source and every window would pay it again.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import structlog

from trading_system.bot.account import resolve_starting_equity
from trading_system.bot.ai_agent import MIN_TRADE_USD, AIAgent, PaperLedger
from trading_system.bot.backtest.historical import (
    HistoricalExchange,
    load_funding,
    load_ohlcv,
    timeframe_interval,
)
from trading_system.bot.backtest.sources import (
    BaseSource,
    SlotContext,
    build_source,
    load_journal_records,
)
from trading_system.bot.backtest.synthetic import build_synthetic_frames
from trading_system.bot.venue_limits import (
    max_notional_for_risk,
    min_equity_required,
    min_legal_notional,
)

logger = structlog.get_logger(__name__)

# Candles a slot needs before the strategy's own indicators are fully warm
# (market_data.MIN_CANDLES is 50; recent_high_7d needs 168). Starting earlier
# would journal "market data unavailable" errors that say nothing about the
# strategy.
WARMUP_CANDLES = 180

TRIGGER_SLOT = "slot"
TRIGGER_INTRABAR = "intrabar"

TRIGGER_MODEL_CAVEATS = {
    TRIGGER_SLOT: (
        "the level decides WHETHER to exit and the price the wakeup actually "
        "observes is the fill — the bot's real paper behaviour, so a gap "
        "through the stop costs more than the stop distance"
    ),
    TRIGGER_INTRABAR: (
        "a resting stop/target fills at its trigger level intrabar, like a real "
        "exchange order. NOT the bot's paper behaviour: a labelled sensitivity "
        "(when one bar touches both levels the stop is assumed to fill first; "
        "real fills would slip past the level, which this does not model)"
    ),
}

# Declared up front, calendar-year, chosen to cover distinct market states
# rather than to flatter anything.
REGIMES: tuple[tuple[str, str, str], ...] = (
    ("2022 bear", "2022-01-01", "2022-12-31"),
    ("2023 recovery", "2023-01-01", "2023-12-31"),
    ("2024 bull", "2024-01-01", "2024-12-31"),
    ("2025 mixed", "2025-01-01", "2025-12-31"),
    ("2026 YTD", "2026-01-01", "2026-12-31"),
)


# ---------------------------------------------------------------------------
# Indicator memo
# ---------------------------------------------------------------------------

_ORIGINAL_COMPUTE = None


def _memo_key(df: pd.DataFrame) -> tuple:
    return (
        len(df),
        str(df.index[-1]),
        float(df["close"].iloc[-1]),
        float(df["volume"].iloc[-1]),
    )


@contextmanager
def indicator_cache(max_entries: int = 100_000) -> Iterator[dict]:
    """Share one ``compute_indicators`` result per candle window, then restore.

    Pure-function memo: the same frame gives the same dict, so this changes
    speed and nothing else. It is installed as a module attribute so both
    ``fetch_market_context`` (inside the agent) and the engine's own calls hit
    the same cache.
    """
    global _ORIGINAL_COMPUTE
    import trading_system.bot.market_data as md

    if _ORIGINAL_COMPUTE is None:
        _ORIGINAL_COMPUTE = md.compute_indicators
    original = _ORIGINAL_COMPUTE
    cache: dict[tuple, dict] = {}

    def memo(df):
        if df is None or getattr(df, "empty", True):
            return original(df)
        try:
            key = _memo_key(df)
        except (KeyError, IndexError, TypeError, ValueError):
            return original(df)
        hit = cache.get(key)
        if hit is None:
            hit = original(df)
            if len(cache) >= max_entries:
                cache.clear()
            cache[key] = hit
        return hit

    md.compute_indicators = memo
    try:
        yield cache
    finally:
        md.compute_indicators = original


def cached_indicators(df: pd.DataFrame) -> dict:
    """``compute_indicators``, through the memo when one is installed."""
    import trading_system.bot.market_data as md

    return md.compute_indicators(df)


# ---------------------------------------------------------------------------
# Time and schedule
# ---------------------------------------------------------------------------

class SimClock:
    """A clock the engine moves by hand; the agent treats it as ``now``."""

    def __init__(self) -> None:
        self.now: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


def load_slots(config_path: str | Path) -> list[tuple[int, int]]:
    """Schedule slots (UTC hour, minute) — parsed by the PRODUCTION gate.

    Reusing the gate's own parser is deliberate: the replay cannot disagree
    with the cloud about when a wakeup happens.
    """
    from scripts.ai_bot_gate import _load_slots

    return _load_slots(Path(config_path))


def iter_slots(start: datetime, end: datetime,
               slots: list[tuple[int, int]]) -> Iterator[datetime]:
    """Every schedule slot in ``[start, end)``, UTC, in order."""
    day = start.astimezone(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    while day < end:
        for hour, minute in slots:
            ts = day.replace(hour=hour, minute=minute)
            if start <= ts < end:
                yield ts
        day += timedelta(days=1)


# ---------------------------------------------------------------------------
# Config / results
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    data_dir: Path = Path("data/raw")
    config_path: Path = Path("configs/ai_bot.yaml")
    pairs: list[str] = field(default_factory=list)      # [] = from the config
    timeframe: str = "1h"
    start: str | None = None
    end: str | None = None
    days: int | None = None    # trim the span to the last N days of the data
    folds: int = 4
    source: str = "random"
    source_params: dict[str, Any] = field(default_factory=dict)
    trigger_model: str = TRIGGER_SLOT
    regimes: bool = True
    folds_enabled: bool = True
    forming_candle: bool = False
    journal_path: Path | None = None
    keep_workdir: bool = False
    synthetic: bool = False
    synthetic_days: int = 400
    synthetic_seed: int = 11
    synthetic_start: str = "2022-01-01"
    workdir: Path | None = None      # inspect (and keep) the replay's data dirs

    def as_dict(self) -> dict:
        out = {}
        for key in ("data_dir", "config_path", "pairs", "timeframe", "start",
                    "end", "days", "folds", "source", "source_params",
                    "trigger_model",
                    "regimes", "folds_enabled", "forming_candle",
                    "journal_path", "keep_workdir", "synthetic",
                    "synthetic_days", "synthetic_seed", "synthetic_start",
                    "workdir"):
            value = getattr(self, key)
            out[key] = str(value) if isinstance(value, Path) else value
        return out


@dataclass
class WindowResult:
    """One replay over one time span, on one fresh (or continuous) account."""

    label: str
    kind: str                  # continuous | fold | regime
    start: str
    end: str
    slots: int
    start_equity: float
    end_equity: float
    return_pct: float
    max_drawdown_pct: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    fees: float
    net_pnl: float
    gross_pnl: float
    opens: int
    avg_risk_usd: float
    expectancy_usd: float
    stopped_out: int
    took_profit: int
    closed_by_ai: int
    open_positions_at_end: int
    heat_pct_peak: float
    ohlcv_calls: int = 0
    candles_served: int = 0
    distinct_windows: int = 0
    look_ahead_violations: int = 0
    order_path_attempts: int = 0
    alerts_recorded: int = 0
    venues: dict[str, dict] = field(default_factory=dict)
    venue_reach: dict[str, dict] = field(default_factory=dict)
    rejections: dict[str, int] = field(default_factory=dict)
    rejection_examples: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    closed_trades: list[dict] = field(default_factory=list)

    def as_dict(self, with_curve: bool = False, with_samples: bool = False) -> dict:
        out = {
            "label": self.label, "kind": self.kind,
            "start": self.start, "end": self.end, "slots": self.slots,
            "start_equity": round(self.start_equity, 2),
            "end_equity": round(self.end_equity, 2),
            "return_pct": round(self.return_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "trades": self.trades, "wins": self.wins, "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "fees": round(self.fees, 4),
            "net_pnl": round(self.net_pnl, 4),
            "gross_pnl": round(self.gross_pnl, 4),
            "opens": self.opens,
            "avg_risk_usd": round(self.avg_risk_usd, 4),
            "expectancy_usd": round(self.expectancy_usd, 6),
            "stopped_out": self.stopped_out,
            "took_profit": self.took_profit,
            "closed_by_ai": self.closed_by_ai,
            "open_positions_at_end": self.open_positions_at_end,
            "heat_pct_peak": round(self.heat_pct_peak, 4),
            "ohlcv_calls": self.ohlcv_calls,
            "candles_served": self.candles_served,
            "distinct_windows": self.distinct_windows,
            "look_ahead_violations": self.look_ahead_violations,
            "order_path_attempts": self.order_path_attempts,
            "alerts_recorded": self.alerts_recorded,
            "venues": self.venues,
            "venue_reach": self.venue_reach,
            "rejections": dict(sorted(self.rejections.items())),
            "rejection_examples": self.rejection_examples,
            "errors": self.errors[:20],
        }
        if with_curve:
            out["equity_curve"] = self.equity_curve
        if with_samples:
            out["samples"] = self.samples
            out["closed_trades"] = self.closed_trades
        return out


@dataclass
class BacktestResult:
    """One source's evidence, across every window it was run over."""

    source: dict
    trigger_model: str
    config: dict
    windows: list[WindowResult]
    trades: list[dict]
    evidence: dict
    limitations: list[str] = field(default_factory=list)

    def window(self, label: str) -> WindowResult | None:
        for w in self.windows:
            if w.label == label:
                return w
        return None

    def as_dict(self, with_curve: bool = False) -> dict:
        return {
            "source": self.source,
            "trigger_model": self.trigger_model,
            "trigger_model_note": TRIGGER_MODEL_CAVEATS.get(self.trigger_model, ""),
            "config": self.config,
            "windows": [w.as_dict(with_curve=with_curve) for w in self.windows],
            "trades": self.trades,
            "evidence": self.evidence,
            "limitations": self.limitations,
        }

    def to_json(self, with_curve: bool = False, indent: int = 2) -> str:
        return json.dumps(self.as_dict(with_curve=with_curve), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------

class RecordingNotifier:
    """Records what the operator would have been told; sends nothing."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def _record(self, kind: str, **payload: Any) -> bool:
        self.messages.append({"kind": kind, **payload})
        return True

    def notify_trade_open(self, **kwargs) -> bool:
        return self._record("trade_open", **kwargs)

    def notify_trade_close(self, **kwargs) -> bool:
        return self._record("trade_close", **kwargs)

    def notify_error(self, error: str, context: str = "") -> bool:
        return self._record("error", error=error, context=context)

    def notify_emergency_stop(self, reason: str) -> bool:
        return self._record("emergency_stop", reason=reason)

    def notify_sltp_trigger(self, **kwargs) -> bool:
        return self._record("sltp_trigger", **kwargs)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

_REJECTION_MARKERS: tuple[tuple[str, str], ...] = (
    ("venue minimum", "venue: below the pair's MIN_NOTIONAL"),
    ("lot", "venue: quantity below the lot step"),
    ("portfolio heat", "exposure cap"),
    ("implied risk", "risk cap (2% of equity)"),
    ("outside (0,", "size cap"),
    ("size ", "size cap"),
    ("confidence", "confidence below the floor"),
    ("risk/reward", "risk/reward below 1.5"),
    ("already open", "position already open for that pair"),
    ("no open position", "close requested with no position"),
    ("trading halted", "drawdown halt"),
    ("not in tradable list", "pair not whitelisted"),
)


def classify_rejection(reason: str) -> str:
    lowered = reason.lower()
    for marker, bucket in _REJECTION_MARKERS:
        if marker in lowered:
            return bucket
    return "other"


def _max_drawdown_pct(curve: list[float]) -> float:
    peak, worst = 0.0, 0.0
    for value in curve:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak * 100.0)
    return worst


def _equity(ledger: PaperLedger, prices: dict[str, float]) -> float:
    return ledger.cash + ledger.unrealized(prices)


def _venue_reach(limits: dict, strategy: dict, equity: float) -> dict[str, dict]:
    """Per pair: can this account place a legal order at all, and up to how much?

    The same arithmetic the agent uses to refuse an entry (``venue_limits``
    helpers, the ``MIN_TRADE_USD`` floor), quoted at the *widest* stop the rules
    allow so the ceiling is the most generous one the account can ever reach.
    The report turns this into plain sentences rather than asserting them.
    """
    rules = strategy.get("rules", {}) or {}
    risk_pct = float(rules.get("max_risk_per_trade_pct", 2.0))
    stop_pct = float(rules.get("stop_loss_max_pct", 5.0))
    cap_pct = min(
        float(rules.get("max_position_size_pct", 100.0)),
        float(rules.get("max_portfolio_heat_pct", 100.0)),
    )
    reach: dict[str, dict] = {}
    for pair, lim in sorted(limits.items()):
        floor = min_legal_notional(lim, MIN_TRADE_USD)
        ceiling = min(
            max_notional_for_risk(equity, risk_pct, stop_pct),
            equity * cap_pct / 100.0,
        )
        reach[pair] = {
            "floor_usd": round(floor, 2),
            "ceiling_usd": round(ceiling, 2),
            "tradable": floor <= ceiling,
            "min_equity_usd": round(
                min_equity_required(floor, risk_pct, stop_pct, cap_pct), 2
            ),
        }
    return reach


# ---------------------------------------------------------------------------
# The replay
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict:
    import yaml

    if not Path(path).exists():
        raise FileNotFoundError(f"no AI bot config at {path}")
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _warm_start(frames: dict[str, pd.DataFrame], timeframe: str) -> pd.Timestamp:
    """First instant at which every pair has a fully warm indicator window."""
    interval = timeframe_interval(timeframe)
    return max(df.index[0] + interval * WARMUP_CANDLES for df in frames.values())


def _default_journal_path(cfg: BacktestConfig) -> Path:
    bot_cfg = _load_config(cfg.config_path).get("bot", {}) or {}
    return Path(bot_cfg.get("data_dir", "data/ai_bot")) / "decisions.jsonl"


def run_backtest(cfg: BacktestConfig,
                 frames: dict[str, pd.DataFrame] | None = None,
                 funding: dict[str, pd.Series] | None = None,
                 source: BaseSource | None = None) -> BacktestResult:
    """Replay ``cfg.source`` over the configured span, folds and regimes."""
    raw_config = _load_config(cfg.config_path)
    bot_cfg = raw_config.get("bot", {}) or {}
    pairs = list(cfg.pairs or bot_cfg.get("pairs") or [])
    if not pairs:
        raise ValueError(f"no pairs configured (bot.pairs in {cfg.config_path})")
    equity = resolve_starting_equity(raw_config)[0]
    timeframe = cfg.timeframe or bot_cfg.get("timeframe", "1h")

    if frames is None or funding is None:
        loaded, loaded_funding, _, _, data_source = resolve_inputs(cfg, pairs, timeframe)
        frames = frames if frames is not None else loaded
        funding = funding if funding is not None else loaded_funding
    else:
        data_source = (
            "synthetic (generated, NOT market data)" if cfg.synthetic
            else str(cfg.data_dir)
        )

    if source is None:
        params = dict(cfg.source_params)
        if cfg.source == "journal":
            params["records"] = load_journal_records(
                cfg.journal_path or _default_journal_path(cfg)
            )
        source = build_source(cfg.source, **params)

    slots = load_slots(cfg.config_path)
    if not slots:
        raise ValueError("the config's schedule has no slots")

    warm = _warm_start(frames, timeframe)
    data_start = min(df.index[0] for df in frames.values())
    data_end = max(df.index[-1] + timeframe_interval(timeframe)
                   for df in frames.values())
    start = max(warm, _as_utc(cfg.start)) if cfg.start else warm
    end = min(data_end, _as_utc(cfg.end)) if cfg.end else data_end
    if cfg.days and cfg.days > 0:
        # "the last N days" of the span that was asked for. Applied after the
        # warm-up floor, so a short window can never start before the indicators
        # are warm, and it lands in the evidence as the replayed range.
        start = max(start, end - timedelta(days=int(cfg.days)))

    spans: list[tuple[str, str, pd.Timestamp, pd.Timestamp]] = [
        ("all data (continuous)", "continuous", start, end)
    ]
    if cfg.folds_enabled and cfg.folds > 0:
        spans += _fold_spans(start, end, cfg.folds)
    if cfg.regimes:
        spans += _regime_spans(start, end)

    if cfg.workdir is not None:
        base_workdir = Path(cfg.workdir)
        base_workdir.mkdir(parents=True, exist_ok=True)
    else:
        base_workdir = Path(tempfile.mkdtemp(prefix="ai_backtest_"))
    windows: list[WindowResult] = []
    all_trades: list[dict] = []
    venues: dict[str, dict] = {}
    violations = order_attempts = 0
    try:
        for label, kind, span_start, span_end in spans:
            if span_end <= span_start:
                continue
            window = _run_window(
                cfg=cfg, raw_config=raw_config, pairs=pairs, frames=frames,
                funding=funding, timeframe=timeframe, start=span_start,
                end=span_end, label=label, kind=kind, equity=equity,
                slots=slots, source=source, base_workdir=base_workdir,
            )
            windows.append(window)
            if kind == "continuous":
                # The action log of the continuous run only: folds and regimes
                # overlap it, so concatenating them would duplicate entries.
                all_trades = window.samples
            venues = venues or window.venues
            violations += window.look_ahead_violations
            order_attempts += window.order_path_attempts
    finally:
        if not cfg.keep_workdir and cfg.workdir is None:
            shutil.rmtree(base_workdir, ignore_errors=True)

    evidence = {
        "data_source": data_source,
        "data_dir": "synthetic" if cfg.synthetic else str(cfg.data_dir),
        "pairs": pairs,
        "data_range": {"start": str(data_start), "end": str(data_end)},
        "effective_range": {"start": str(start), "end": str(end)},
        "warmup_candles": WARMUP_CANDLES,
        "timeframe": timeframe,
        "slots_per_day": len(slots),
        "slot_hours_utc": [f"{h:02d}:{m:02d}" for h, m in slots],
        "schedule_source": str(cfg.config_path),
        "source_calls": getattr(source, "calls", None),
        "journal_matched_slots": getattr(source, "matched_slots", None),
        "journal_unmatched_slots": getattr(source, "unmatched_slots", None),
        "journal_replayable_slots": getattr(source, "replayable_slots", None),
        "journal_records_without_actions": getattr(
            source, "records_without_actions", None
        ),
        "funding_data_pairs": sorted(funding or {}),
        "continuous_slots": sum(w.slots for w in windows if w.kind == "continuous"),
        # Computed from the same helpers the agent sizes with, so the report's
        # "BTC cannot be traded at $97" is arithmetic, not an assertion.
        "venue_reach": next(
            (w.venue_reach for w in windows if w.kind == "continuous"), {}
        ),
        # Guards, not assumptions: the counter would be non-zero here if a
        # replayed wakeup had ever been served a candle after its cursor, or if
        # anything had reached an order method.
        "order_path_attempts": order_attempts,
        "look_ahead_violations": violations,
        "workdir_kept": cfg.keep_workdir,
    }

    return BacktestResult(
        source=source.provenance(),
        trigger_model=cfg.trigger_model,
        config={
            "equity_from_config": equity,
            "config_path": str(cfg.config_path),
            "risk_rules": raw_config.get("risk", {}) or {},
            "bot": {
                "pairs": pairs, "timeframe": timeframe,
                "paper_starting_equity": bot_cfg.get("paper_starting_equity"),
            },
            "venues": venues,
        },
        windows=windows,
        trades=all_trades,
        evidence=evidence,
        limitations=_limitations(cfg),
    )


def resolve_inputs(cfg: BacktestConfig, pairs: list[str], timeframe: str):
    """Candles (and funding) for a run, real or synthetic."""
    if cfg.synthetic:
        return (
            build_synthetic_frames(
                pairs, timeframe, start=cfg.synthetic_start,
                days=cfg.synthetic_days, seed=cfg.synthetic_seed,
            ),
            {}, pairs, timeframe, "synthetic (generated, NOT market data)",
        )
    frames = {p: load_ohlcv(cfg.data_dir, p, timeframe) for p in pairs}
    funding = {p: s for p in pairs
               if (s := load_funding(cfg.data_dir, p)) is not None}
    return frames, funding, pairs, timeframe, str(cfg.data_dir)


def _as_utc(value: str | datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _fold_spans(start: pd.Timestamp, end: pd.Timestamp,
                folds: int) -> list[tuple[str, str, pd.Timestamp, pd.Timestamp]]:
    """Contiguous, non-overlapping, equal-duration walk-forward windows."""
    total = end - start
    step = total / folds
    spans = []
    for i in range(folds):
        a = start + step * i
        b = end if i == folds - 1 else start + step * (i + 1)
        spans.append((f"fold {i + 1}/{folds}", "fold", a, b))
    return spans


def _regime_spans(start: pd.Timestamp, end: pd.Timestamp):
    spans = []
    for label, first, last in REGIMES:
        a = max(start, _as_utc(first))
        b = min(end, _as_utc(last) + pd.Timedelta(days=1))
        if b > a:
            spans.append((label, "regime", a, b))
    return spans


def _limitations(cfg: BacktestConfig) -> list[str]:
    out = [
        "The LLM's judgement is NOT tested. A mechanical rule (or a coin flip) "
        "stands in for the model, and the model cannot be backtested in "
        "principle without look-ahead: its output depends on everything it "
        "believes about the world at decision time, not only on these candles.",
        "One timeframe, one decision per slot: no news, no order-book depth, no "
        "liquidity limit on size.",
        "By default the replay decides on CLOSED candles only, so the slot price "
        "is the newest closed candle's close. Production also sees the candle "
        "still forming and a live ticker: the replay is a slightly different "
        "input (and the less optimistic one), not a look-ahead. The "
        "--forming-candle flag models the production fetch instead.",
        "Paper-mode accounting throughout — the bot's own: taker fees only "
        "(0.05% per side), no slippage, no spread, no partial fills.",
        "In the slot trigger model a stop or target exits at the price the "
        "wakeup observes, so a gap through the level costs more than the level "
        "itself — the paper ledger's real behaviour, kept rather than smoothed.",
        "The paper ledger does not charge funding. Real perp positions pay or "
        "receive it every 8 hours; it is omitted here and quantified in the "
        "report rather than ignored.",
        "Compounding runs on a tiny base: a handful of trades at $97 dominates "
        "the result, and the number of independent bets is small.",
    ]
    if cfg.source == "random":
        out.append(
            "This source is a coin flip. Its result measures the account's cost "
            "structure, not a strategy."
        )
    return out


# ---------------------------------------------------------------------------
# One window
# ---------------------------------------------------------------------------

def _run_window(
    cfg: BacktestConfig, raw_config: dict, pairs: list[str],
    frames: dict[str, pd.DataFrame], funding: dict[str, pd.Series],
    timeframe: str, start: pd.Timestamp, end: pd.Timestamp,
    label: str, kind: str, equity: float, slots: list[tuple[int, int]],
    source: BaseSource, base_workdir: Path,
) -> WindowResult:
    workdir = base_workdir / f"{kind}_{label}".replace(" ", "_").replace("/", "-")
    workdir.mkdir(parents=True, exist_ok=True)

    clock = SimClock()
    exchange = HistoricalExchange(
        frames, timeframe=timeframe, funding=funding,
        forming_candle=cfg.forming_candle,
    )
    ledger = PaperLedger(
        workdir / "paper_ledger.json", starting_cash=equity, clock=clock,
    )
    notifier = RecordingNotifier()

    window_config = json.loads(json.dumps(raw_config, default=str))
    window_config.setdefault("bot", {})["mode"] = "paper"
    # Fresh window -> fresh history: nothing may carry a peak equity (and so a
    # drawdown halt) in from another window.
    agent = AIAgent(
        exchange=exchange,
        data_dir=workdir,
        config=window_config,
        mode="paper",
        engine=source,
        ledger=ledger,
        notifier=notifier,
        clock=clock,
    )
    source.reset()
    strategy = agent._read_strategy()
    venues = {p: lim.as_dict() for p, lim in sorted(agent._limits.items())}
    logger.info(
        "backtest_window_start", window=label, kind=kind, start=str(start),
        end=str(end), pairs=pairs, venues=sorted(venues),
    )

    slot_list = list(iter_slots(start.to_pydatetime(), end.to_pydatetime(), slots))
    curve: list[float] = []
    heat_peak = 0.0
    rejections: dict[str, int] = {}
    rejection_examples: dict[str, str] = {}
    errors: list[str] = []
    samples: list[dict] = []
    stopped = took_profit = closed_by_ai = 0
    risk_sum, risk_n = 0.0, 0
    consecutive_errors = 0

    for i, slot_dt in enumerate(slot_list):
        clock.now = slot_dt
        exchange.set_cursor(slot_dt)
        prices = {p: exchange.last_price(p) for p in pairs}
        indicators = {
            p: cached_indicators(exchange.get_ohlcv(p, timeframe, limit=200))
            for p in pairs
        }
        positions = ledger.open_positions_list()
        snapshot = agent._risk_snapshot(
            strategy, agent._read_progress(), _equity(ledger, prices), positions
        )
        source.set_context(SlotContext(
            now=slot_dt, timeframe=timeframe, strategy=strategy,
            equity=snapshot["equity"], positions=positions,
            prices=prices, indicators=indicators, snapshot=snapshot,
        ))

        result = agent.run_wakeup()

        if result.get("status") == "error":
            msg = "; ".join(str(e) for e in result.get("errors", [])) or "unknown"
            errors.append(f"{slot_dt}: {msg}")
            consecutive_errors += 1
            if consecutive_errors >= 3 and not curve:
                raise RuntimeError(
                    f"every wakeup in '{label}' failed ({msg}). The replay is "
                    "not producing decisions — check the data and the warm-up "
                    "window before reading anything into a result."
                )
        else:
            consecutive_errors = 0

        for reason in result.get("rejections", []) or []:
            bucket = classify_rejection(str(reason))
            rejections[bucket] = rejections.get(bucket, 0) + 1
            rejection_examples.setdefault(bucket, str(reason))

        for trade in result.get("actions_taken", []) or []:
            samples.append(_trade_sample(
                trade, slot_dt, _equity(ledger, prices), snapshot["equity"],
            ))
            if trade.get("side") in ("long", "short"):
                size_usd = float(trade.get("size_usd", 0) or 0)
                stop_pct = float(trade.get("stop_loss_pct", 0) or 0)
                risk_sum += abs(size_usd) * abs(stop_pct) / 100.0
                risk_n += 1
            elif trade.get("side") == "close":
                closed_by_ai += 1

        for closed in result.get("closed_triggers", []) or []:
            reason = str(closed.get("reason", ""))
            if "stop" in reason:
                stopped += 1
            elif "profit" in reason:
                took_profit += 1

        if cfg.trigger_model == TRIGGER_INTRABAR:
            nxt = slot_list[i + 1] if i + 1 < len(slot_list) else end
            for rec in _intrabar_triggers(exchange, ledger, slot_dt, nxt):
                if "stop" in str(rec.get("reason", "")):
                    stopped += 1
                else:
                    took_profit += 1

        heat_peak = max(heat_peak, snapshot["portfolio_heat_pct"])
        curve.append(_equity(ledger, prices))
        if (i + 1) % 1000 == 0:
            logger.info(
                "backtest_progress", window=label, slots=i + 1,
                of=len(slot_list), equity=round(curve[-1], 2),
            )

    # Mark still-open positions to market for the closing figure.
    exchange.set_cursor(end)
    final_prices = {p: exchange.last_price(p) for p in pairs}
    end_equity = _equity(ledger, final_prices)
    if not curve:
        curve = [equity, end_equity]

    closed_trades = ledger.data.get("closed_trades", [])
    fees = sum(float(t.get("fees", 0) or 0) for t in closed_trades)
    gross = sum(float(t.get("gross_pnl", 0) or 0) for t in closed_trades)
    net_pnl = sum(float(t.get("net_pnl", 0) or 0) for t in closed_trades)
    wins = sum(1 for t in closed_trades if float(t.get("net_pnl", 0) or 0) > 0)
    losses = sum(1 for t in closed_trades if float(t.get("net_pnl", 0) or 0) < 0)

    return WindowResult(
        label=label, kind=kind, start=str(start), end=str(end),
        slots=len(slot_list), start_equity=equity, end_equity=end_equity,
        return_pct=(end_equity / equity - 1) * 100.0 if equity else 0.0,
        max_drawdown_pct=_max_drawdown_pct(curve),
        trades=len(closed_trades), wins=wins, losses=losses,
        win_rate=(wins / len(closed_trades) * 100.0) if closed_trades else 0.0,
        fees=fees, net_pnl=net_pnl, gross_pnl=gross, opens=risk_n,
        avg_risk_usd=(risk_sum / risk_n) if risk_n else 0.0,
        expectancy_usd=(net_pnl / len(closed_trades)) if closed_trades else 0.0,
        stopped_out=stopped, took_profit=took_profit, closed_by_ai=closed_by_ai,
        open_positions_at_end=len(ledger.positions),
        heat_pct_peak=heat_peak,
        closed_trades=[_closed_sample(t) for t in closed_trades],
        ohlcv_calls=exchange.ohlcv_calls,
        candles_served=exchange.candles_served,
        distinct_windows=exchange.distinct_windows,
        look_ahead_violations=exchange.look_ahead_violations,
        order_path_attempts=exchange.order_path_attempts,
        alerts_recorded=len(notifier.messages),
        venues=venues,
        venue_reach=_venue_reach(agent._limits, strategy, equity),
        rejections=rejections, rejection_examples=rejection_examples,
        errors=errors,
        equity_curve=[
            {"t": str(ts), "equity": round(v, 2)}
            for ts, v in zip(slot_list, curve)
        ],
        samples=samples,
    )


def _closed_sample(rec: dict) -> dict:
    return {
        "pair": rec.get("pair"),
        "position_side": rec.get("position_side"),
        "entry_price": rec.get("entry_price"),
        "exit_price": rec.get("exit_price"),
        "size_usd": rec.get("size_usd"),
        "gross_pnl": rec.get("gross_pnl"),
        "fees": rec.get("fees"),
        "net_pnl": rec.get("net_pnl"),
        "pnl_pct_net": rec.get("pnl_pct_net"),
        "reason": rec.get("reason"),
        "entry_time": rec.get("entry_time"),
        "close_time": rec.get("close_time"),
    }


def _trade_sample(trade: dict, slot_dt: datetime, equity_after: float,
                  equity_at_slot: float) -> dict:
    """One executed action, with the two equities a cap check needs.

    ``equity_at_slot`` is the account the entry was sized against (the same
    number the agent's validation used), so a test can verify the caps were
    respected rather than assume it.
    """
    return {
        "t": str(slot_dt),
        "pair": trade.get("pair"),
        "side": trade.get("side"),
        "price": trade.get("price"),
        "amount": trade.get("amount"),
        "size_usd": trade.get("size_usd"),
        "stop_loss_pct": trade.get("stop_loss_pct"),
        "take_profit_pct": trade.get("take_profit_pct"),
        "net_pnl": trade.get("net_pnl"),
        "fees": trade.get("fees"),
        "reason": trade.get("reason"),
        "equity_at_slot": round(equity_at_slot, 4),
        "equity_after": round(equity_after, 2),
    }


def _intrabar_triggers(exchange: HistoricalExchange, ledger: PaperLedger,
                       start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """Close positions whose stop/target a real order would have filled.

    Pessimistic on purpose: if one candle touches both levels, the stop is
    assumed to fill first.
    """
    closed: list[dict] = []
    for pair in list(ledger.positions.keys()):
        pos = ledger.positions.get(pair)
        if not pos:
            continue
        entry = float(pos["entry_price"])
        sl_pct = float(pos["stop_loss_pct"])
        tp_pct = float(pos["take_profit_pct"])
        is_long = pos["side"] == "buy"
        sl_px = entry * (1 - sl_pct / 100.0) if is_long else entry * (1 + sl_pct / 100.0)
        tp_px = entry * (1 + tp_pct / 100.0) if is_long else entry * (1 - tp_pct / 100.0)
        for _, row in exchange.raw_slice(pair, start, end).iterrows():
            low, high = float(row["low"]), float(row["high"])
            hit_sl = low <= sl_px if is_long else high >= sl_px
            hit_tp = high >= tp_px if is_long else low <= tp_px
            if hit_sl:
                rec = ledger.close_position(pair, sl_px, "stop_loss triggered")
                if rec:
                    closed.append(rec)
                break
            if hit_tp:
                rec = ledger.close_position(pair, tp_px, "take_profit triggered")
                if rec:
                    closed.append(rec)
                break
    return closed


def run_suite(configs: list[BacktestConfig]) -> list[BacktestResult]:
    """Run several sources over the same data, sharing one indicator memo."""
    frames = None
    funding = None
    results: list[BacktestResult] = []
    with indicator_cache():
        for cfg in configs:
            if frames is None:
                raw = _load_config(cfg.config_path)
                bot_cfg = raw.get("bot", {}) or {}
                pairs = list(cfg.pairs or bot_cfg.get("pairs") or [])
                timeframe = cfg.timeframe or bot_cfg.get("timeframe", "1h")
                frames, funding, _, _, _ = resolve_inputs(cfg, pairs, timeframe)
            results.append(run_backtest(cfg, frames=frames, funding=funding))
    return results

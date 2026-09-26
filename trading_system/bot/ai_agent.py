"""
AI Trading Agent — stateless, scheduled, with shared-file continuity.

Architecture (adapted from the Nate Herk / GPT-6 Astra plan, running on
gpt-4o instead of GPT-6 Astra):

Each "wakeup" is a completely fresh run with no in-memory history.
Continuity comes from persistent shared files in the data directory:

  - strategy.json    — the rules, constraints, and current strategy
  - progress.json    — handoff notes + equity tracking for the next wakeup
  - journal.jsonl    — append-only log of every wakeup (audit trail)
  - decisions.json   — latest raw AI decision for audit
  - paper_ledger.json— simulated fills/positions/equity (paper mode only)
  - trades.jsonl     — append-only executed-trade log

Flow for every wakeup:
  1. Read strategy document
  2. Read progress log (last handoff notes)
  3. Process paper SL/TP triggers against current prices
  4. Fetch live market data + build portfolio/risk/strategy context
  5. Call the AI engine with all context
  6. Validate the decision against hard risk rules
  7. Execute approved trades (closes first, then opens)
  8. Update progress log, journal, and trade log

Every failure path writes to the journal and returns an explicit error
status — nothing fails silently.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import structlog

from trading_system.bot.account import (
    DEFAULT_STARTING_EQUITY,
    legacy_scale_notice,
    resolve_starting_equity,
)
from trading_system.bot.ai_engine import AIEngine, TradeAction, TradingDecision
from trading_system.bot.exchange import ExchangeInterface
from trading_system.bot.market_data import fetch_market_context
from trading_system.bot.risk_manager import RiskManager
from trading_system.bot.telegram_notifier import TelegramNotifier
from trading_system.bot.venue_limits import (
    MarketLimits,
    floor_to_step,
    max_notional_for_risk,
    min_equity_required,
    min_legal_notional,
    order_problems,
    resolve_limits,
    round_trip_fee,
)

logger = structlog.get_logger(__name__)

VALID_SIDES = {"long", "short", "close"}
# How a "long"/"short" AI decision maps to an exchange order side.
ORDER_SIDE = {"long": "buy", "short": "sell"}
# Paper-trading taker fee per fill, in percent of notional. Matches the venue's
# public taker rate (Binance USDⓈ-M: 0.05%), so paper fees are not optimistic.
PAPER_TAKER_FEE_PCT = 0.05
# Floor for a paper fill. The VENUE minimum (see venue_limits.py) is usually
# far larger and always wins; this only stops pathological dust orders.
MIN_TRADE_USD = 5.0


# ---------------------------------------------------------------------------
# Default strategy document (the rules the AI follows)
# ---------------------------------------------------------------------------

DEFAULT_STRATEGY: dict[str, Any] = {
    "name": "AI Crypto Trader",
    "version": "1.1",
    "description": "LLM-driven crypto perpetual futures trading on Binance",
    "pairs": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
    "timeframe": "1h",
    "rules": {
        "max_risk_per_trade_pct": 2.0,
        "max_position_size_pct": 10.0,
        "max_portfolio_heat_pct": 30.0,
        "max_open_positions": 3,
        "max_drawdown_pct": 10.0,
        "stop_loss_max_pct": 5.0,
        "min_risk_reward_ratio": 1.5,
        "min_confidence_to_trade": 60,
    },
    "preferences": {
        "preferred_pairs": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
        "avoid_high_funding": True,
        "funding_threshold_pct": 0.05,
        "prefer_trend_following": True,
    },
}


def _utc_now_iso(now: datetime | None = None) -> str:
    """ISO-8601 UTC timestamp, or ``now`` formatted when one is supplied.

    The optional argument exists so a replay (the backtest harness) can write
    simulated timestamps while every other call site is unchanged.
    """
    return (now or datetime.now(timezone.utc)).isoformat()


def _parse_iso(value: Any) -> datetime | None:
    """A stored ISO timestamp as tz-aware UTC, or None when unusable."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _halt_summary(snapshot: dict) -> str:
    """One line for the AI's risk block: whether the halt binds, and why.

    A halt that is merely serving its cool-off is still a halt (closes only), so
    the AI must not be told "No" — but it is not a drawdown over the limit
    either, and saying so would misstate the account.
    """
    if not snapshot.get("trading_halted"):
        return "No"
    reason = snapshot.get("dd_halt_reason")
    if reason == "cooldown":
        return (f"YES — closes only, flat until {snapshot.get('dd_cooldown_until')} "
                "(cool-off after a drawdown breach)")
    if reason == "budget":
        return ("YES — closes only, drawdown still over the limit and the re-arm "
                "budget is spent")
    return "YES — closes only"


def _read_json(path: Path) -> Any | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Paper ledger — simulated fills, positions, SL/TP triggers, equity
# ---------------------------------------------------------------------------

class PaperLedger:
    """Simulated trading account for paper mode.

    Persists to JSON so state survives between wakeups (the agent itself is
    stateless; the ledger is the shared file that carries the account).
    """

    def __init__(self, path: Path, starting_cash: float = DEFAULT_STARTING_EQUITY,
                 taker_fee_pct: float = PAPER_TAKER_FEE_PCT,
                 clock: Callable[[], datetime] | None = None):
        self.path = path
        self.taker_fee_pct = taker_fee_pct
        # Trade timestamps come from here. Default is the real clock; a replay
        # injects simulated time so its records are reproducible.
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.data: dict[str, Any] = _read_json(path) if path.exists() else None
        if not isinstance(self.data, dict) or "cash" not in self.data:
            self.data = {
                "cash": float(starting_cash),
                "start_equity": float(starting_cash),
                "peak_equity": float(starting_cash),
                "positions": {},
                "closed_trades": [],
            }
            self._save()

    # -- persistence -------------------------------------------------------

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        os.replace(tmp, self.path)

    # -- account math ------------------------------------------------------

    @property
    def cash(self) -> float:
        return float(self.data["cash"])

    @property
    def positions(self) -> dict[str, dict]:
        return self.data["positions"]

    def unrealized(self, prices: dict[str, float]) -> float:
        total = 0.0
        for pair, pos in self.positions.items():
            price = prices.get(pair)
            if price is None:
                price = pos["entry_price"]
            direction = 1.0 if pos["side"] == "buy" else -1.0
            total += direction * (price - pos["entry_price"]) * pos["amount"]
        return total

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.unrealized(prices)

    def update_peak(self, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        if eq > self.data["peak_equity"]:
            self.data["peak_equity"] = eq
            self._save()
        return float(self.data["peak_equity"])

    def rearm_peak(self, equity: float) -> float:
        """Set the high-water mark to ``equity`` — the drawdown RE-ARM.

        ``update_peak`` only ever raises the mark. That is right for measuring
        drawdown and wrong for releasing a halt: a flat cash account below its
        peak can never climb back above the line on its own, so a halt whose
        only release condition is price is permanent. The caller decides when
        releasing is safe; this method moves the baseline exactly.
        """
        self.data["peak_equity"] = float(equity)
        self._save()
        return float(equity)

    # -- trading ------------------------------------------------------------

    def open_position(self, pair: str, order_side: str, price: float,
                      size_usd: float, stop_loss_pct: float,
                      take_profit_pct: float) -> dict:
        """Open a simulated position. order_side is 'buy' or 'sell'."""
        amount = size_usd / price
        fee = size_usd * self.taker_fee_pct / 100
        self.data["cash"] = self.cash - fee
        self.positions[pair] = {
            "side": order_side,
            "entry_price": float(price),
            "amount": float(amount),
            "size_usd": float(size_usd),
            "stop_loss_pct": float(stop_loss_pct),
            "take_profit_pct": float(take_profit_pct),
            "entry_time": _utc_now_iso(self._clock()),
            "high_pnl_pct": 0.0,
        }
        self._save()
        return dict(self.positions[pair])

    def close_position(self, pair: str, price: float, reason: str) -> dict | None:
        """Close a simulated position. Returns the closed-trade record."""
        pos = self.positions.get(pair)
        if not pos:
            return None
        direction = 1.0 if pos["side"] == "buy" else -1.0
        gross_pnl = direction * (price - pos["entry_price"]) * pos["amount"]
        exit_notional = pos["amount"] * price
        # The entry fee was already deducted from cash at open; only the exit
        # fee moves cash here. `net_pnl` reports the FULL round-trip cost so
        # the alert reconciles with the equity change (audited 2026-09-16:
        # the message said -$17.71 while equity moved -$18.21 — the missing
        # $0.50 was the entry fee, and every P&L total inherited the error).
        size_usd = float(pos.get("size_usd") or pos["amount"] * pos["entry_price"])
        entry_fee = size_usd * self.taker_fee_pct / 100
        exit_fee = exit_notional * self.taker_fee_pct / 100
        cash_delta = gross_pnl - exit_fee
        net_pnl = gross_pnl - entry_fee - exit_fee
        self.data["cash"] = self.cash + cash_delta
        del self.positions[pair]
        record = {
            "pair": pair,
            "side": "close",
            "position_side": pos["side"],
            "entry_price": pos["entry_price"],
            "exit_price": float(price),
            "amount": pos["amount"],
            "size_usd": round(size_usd, 4),
            "gross_pnl": round(gross_pnl, 4),
            "entry_fee": round(entry_fee, 4),
            "exit_fee": round(exit_fee, 4),
            "fees": round(entry_fee + exit_fee, 4),
            "net_pnl": round(net_pnl, 4),
            "cash_delta": round(cash_delta, 4),
            # Price move (unchanged convention, used by charts/backtests) vs
            # the round-trip net % that matches net_pnl and equity.
            "pnl_pct": round(
                direction * (price - pos["entry_price"]) / pos["entry_price"] * 100, 4
            ),
            "pnl_pct_net": round(net_pnl / size_usd * 100, 4) if size_usd else 0.0,
            "reason": reason,
            "entry_time": pos["entry_time"],
            "close_time": _utc_now_iso(self._clock()),
        }
        self.data["closed_trades"].append(record)
        # Keep the file bounded.
        self.data["closed_trades"] = self.data["closed_trades"][-500:]
        self._save()
        return record

    def check_triggers(self, prices: dict[str, float]) -> list[dict]:
        """Close any positions whose stop-loss or take-profit was hit.

        Returns the list of closed-trade records.
        """
        closed: list[dict] = []
        for pair in list(self.positions.keys()):
            pos = self.positions.get(pair)
            if not pos:
                continue
            price = prices.get(pair)
            if price is None or price <= 0:
                continue
            entry = pos["entry_price"]
            sl_pct, tp_pct = pos["stop_loss_pct"], pos["take_profit_pct"]
            if pos["side"] == "buy":
                hit_sl = price <= entry * (1 - sl_pct / 100)
                hit_tp = price >= entry * (1 + tp_pct / 100)
            else:
                hit_sl = price >= entry * (1 + sl_pct / 100)
                hit_tp = price <= entry * (1 - tp_pct / 100)
            if hit_sl:
                rec = self.close_position(pair, price, "stop_loss triggered")
                if rec:
                    closed.append(rec)
            elif hit_tp:
                rec = self.close_position(pair, price, "take_profit triggered")
                if rec:
                    closed.append(rec)
        return closed

    def consecutive_losses(self) -> int:
        count = 0
        for rec in reversed(self.data["closed_trades"]):
            if rec.get("net_pnl", 0) < 0:
                count += 1
            else:
                break
        return count

    def open_positions_list(self) -> list[dict]:
        """Positions in the same dict shape the exchange interface returns."""
        out = []
        for pair, pos in self.positions.items():
            out.append({
                "pair": pair,
                "side": "long" if pos["side"] == "buy" else "short",
                "size": pos["amount"],
                "notional": pos["size_usd"],
                "entry_price": pos["entry_price"],
                "unrealized_pnl": 0.0,  # filled in by the agent with live prices
                "leverage": 1.0,
            })
        return out


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AIAgent:
    """Stateless trading agent with shared-file continuity.

    Usage:
        agent = AIAgent(exchange, data_dir="data/ai_bot", config=cfg)
        result = agent.run_wakeup()   # one complete cycle

    ``cfg`` must carry ``bot.paper_starting_equity`` — the account's origin has
    exactly one owner and no fallback (see trading_system/bot/account.py).
    """

    def __init__(
        self,
        exchange: ExchangeInterface,
        data_dir: str | Path = "data/ai_bot",
        config: dict[str, Any] | None = None,
        strategy_file: str | None = None,
        ai_model: str | None = None,
        risk_manager: RiskManager | None = None,
        mode: str = "paper",  # "paper" | "live" | "dry_run"
        engine: AIEngine | None = None,       # injectable for tests
        ledger: PaperLedger | None = None,    # injectable for tests
        notifier: TelegramNotifier | None = None,
        clock: Callable[[], datetime] | None = None,   # injectable for replays
    ):
        if mode not in {"paper", "live", "dry_run"}:
            raise ValueError(
                f"Invalid mode {mode!r}: must be one of paper|live|dry_run"
            )
        self.exchange = exchange
        self.config = config or {}
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        # Single source of "now" for this agent. Default is the real clock; a
        # replay injects simulated time so wakeup ids, the daily rollover and
        # every written timestamp are reproducible instead of wall-clock.
        self._clock = clock or (lambda: datetime.now(timezone.utc))

        bot_cfg = self.config.get("bot", {})
        if strategy_file:
            self.strategy_path = Path(strategy_file)
        else:
            self.strategy_path = self.data_dir / "strategy.json"
        self.strategy_file_override = strategy_file

        self.pairs: list[str] = list(bot_cfg.get("pairs", DEFAULT_STRATEGY["pairs"]))
        self.timeframe: str = bot_cfg.get("timeframe", DEFAULT_STRATEGY["timeframe"])
        model = ai_model or self.config.get("ai", {}).get("model", "gemini-flash-latest")
        self.ai = engine or AIEngine(
            model=model,
            temperature=self.config.get("ai", {}).get("temperature", 0.3),
            max_tokens=self.config.get("ai", {}).get("max_tokens", 2000),
        )
        self.risk = risk_manager or self._build_risk_manager()

        # The account's origin comes from the config and nowhere else. There is
        # deliberately no fallback: a silent default is how a $10,000 book ran
        # for a $97 balance without anything failing.
        self.starting_equity, self.starting_equity_source = resolve_starting_equity(
            self.config
        )
        self.ledger = ledger or (
            PaperLedger(
                self.data_dir / "paper_ledger.json",
                starting_cash=self.starting_equity,
                clock=self._clock,
            )
            if mode == "paper" else None
        )
        # Venue order bounds (MIN_NOTIONAL / LOT_SIZE): from the exchange when
        # reachable, else the dated builtin table. An order the venue would
        # refuse must never be treated as fillable, in paper or live.
        self._limits: dict[str, MarketLimits] = {}
        self._venue_limits(self.pairs)

        # Live-execution guard state, reset at the start of every wakeup. A
        # protective-order failure is recorded here so it reaches the journal
        # and Telegram instead of only the log, and blocks further entries
        # until the next wakeup rather than firing more orders into a venue
        # that just refused a stop.
        self._execution_errors: list[str] = []
        self._entries_blocked_reason: str | None = None

        if notifier is not None:
            self.notifier = notifier
        else:
            tg = self.config.get("telegram", {})
            # AI_-prefixed names take precedence so the AI bot and the main
            # bot can use DIFFERENT Telegram bots without clashing; the
            # unprefixed names still work when no clash exists.
            self.notifier = TelegramNotifier(
                bot_token=(
                    os.environ.get("AI_TELEGRAM_BOT_TOKEN")
                    or os.environ.get("TELEGRAM_BOT_TOKEN", "")
                ),
                chat_id=(
                    os.environ.get("AI_TELEGRAM_CHAT_ID")
                    or os.environ.get("TELEGRAM_CHAT_ID", "")
                ),
                enabled=bool(tg.get("enabled", False)),
            )

        # File paths
        self.progress_path = self.data_dir / "progress.json"
        self.journal_path = self.data_dir / "journal.jsonl"
        self.decisions_path = self.data_dir / "decisions.json"
        self.trades_path = self.data_dir / "trades.jsonl"

    # ----- Clock -----

    def _now(self) -> datetime:
        """Current time for this agent (simulated during a replay)."""
        return self._clock()

    def _now_iso(self) -> str:
        return self._now().isoformat()

    def _build_risk_manager(self) -> RiskManager:
        rcfg = self.config.get("risk", {})
        return RiskManager(
            position_stop_loss_pct=rcfg.get("stop_loss_max_pct", 5.0),
            portfolio_max_dd_pct=rcfg.get("max_drawdown_pct", 10.0),
            max_open_positions=rcfg.get("max_open_positions", 3),
            max_portfolio_heat_pct=rcfg.get("max_portfolio_heat_pct", 30.0),
            # How long the drawdown halt keeps entries flat before it re-arms.
            # The same field the other bot's risk manager uses.
            dd_cooldown_hours=rcfg.get("dd_cooldown_hours", 24.0),
        )

    # ----- Venue limits -----

    def _venue_limits(self, pairs: list[str]) -> dict[str, MarketLimits]:
        """Venue order bounds per pair, looked up once per process.

        Live values are preferred; the dated builtin table covers the pairs it
        knows when the exchange is unreachable (the cloud runner is geo-blocked
        from Binance). A pair we know nothing about is left out entirely, and
        that gap is logged rather than treated as "unlimited".
        """
        for pair in pairs:
            if pair in self._limits:
                continue
            fetched = None
            getter = getattr(self.exchange, "get_market_limits", None)
            if getter is not None:
                try:
                    fetched = getter(pair)
                except Exception as e:
                    logger.warning(
                        "venue_limits_lookup_failed", pair=pair, error=str(e)[:150]
                    )
            resolved = resolve_limits(pair, fetched)
            if resolved is None:
                logger.warning("venue_limits_unknown", pair=pair)
                continue
            self._limits[pair] = resolved
        return {p: self._limits[p] for p in pairs if p in self._limits}

    def _venue_rejection(
        self, action: TradeAction, strategy: dict, snapshot: dict
    ) -> str | None:
        """Why the venue would refuse this entry, or None if it can accept it.

        The venue floor and the ceiling the account rules allow are compared
        directly. When the smallest order the exchange accepts is larger than
        the largest the rules permit, NO legal trade exists at this account
        size — we say that plainly instead of quietly never trading.
        """
        limits = self._limits.get(action.pair)
        if limits is None:
            return None  # unknown bounds for this pair (logged at lookup time)
        equity = float(snapshot.get("equity") or 0.0)
        if equity <= 0:
            return f"cannot size {action.pair}: equity unknown"
        rules = strategy.get("rules", {})
        risk_pct = float(rules.get("max_risk_per_trade_pct", 2.0))
        size_pct = float(rules.get("max_position_size_pct", 10.0))
        heat_pct = float(rules.get("max_portfolio_heat_pct", 30.0))
        # A single position is bounded by BOTH the per-trade size cap and the
        # total-exposure cap (it cannot be larger than everything allowed), so
        # the ceiling is the smaller of them. Quoting only the risk-derived
        # figure would overstate what can be placed and understate the equity
        # at which the pair becomes tradable.
        cap_pct = min(size_pct, heat_pct)
        floor = min_legal_notional(limits, MIN_TRADE_USD)
        ceiling = min(
            max_notional_for_risk(equity, risk_pct, action.stop_loss_pct),
            equity * cap_pct / 100.0,
        )
        if floor > ceiling:
            min_equity = min_equity_required(
                floor, risk_pct, action.stop_loss_pct, cap_pct
            )
            return (
                f"venue minimum ${floor:,.2f} for {action.pair} exceeds the "
                f"${ceiling:,.2f} a position may reach at ${equity:,.2f} equity "
                f"(risk {risk_pct:g}% of equity per trade, exposure cap "
                f"{cap_pct:g}% of equity; needs ~${min_equity:,.2f})"
            )
        requested = equity * (action.size_pct / 100.0)
        if requested < floor:
            return (
                f"requested ${requested:,.2f} is below the ${floor:,.2f} venue "
                f"minimum for {action.pair} "
                f"(ask for at least {floor / equity * 100:.1f}% of equity)"
            )
        return None

    # ----- Shared file I/O -----

    def _default_strategy(self) -> dict:
        """DEFAULT_STRATEGY deep-copied, with the config's hard rules applied."""
        doc = json.loads(json.dumps(DEFAULT_STRATEGY))
        doc["rules"].update(self.config.get("risk") or {})
        return doc

    def _with_config_rules(self, strategy: dict) -> dict:
        """Config ``risk:`` overrides the strategy document's rules.

        Both places owned these rules, so strategy.json silently shadowed the
        config: the cap in the operator's config was NOT the cap in force (a
        stale ``max_position_size_pct: 10`` there kept overriding a config of
        40). The config is the control panel for the HARD limits — the ones
        the AI cannot override — so it wins. Strategy-level fields (pairs,
        timeframe, preferences) still come from the strategy document, as does
        any rule the config does not set.
        """
        merged = dict(strategy)
        rules = dict(strategy.get("rules") or {})
        rules.update(self.config.get("risk") or {})
        merged["rules"] = rules
        return merged

    def _read_strategy(self) -> dict:
        """Read the strategy document. Creates the default if missing."""
        data = _read_json(self.strategy_path)
        if isinstance(data, dict) and data.get("pairs"):
            return self._with_config_rules(data)
        if not self.strategy_file_override:
            self._write_json(self.strategy_path, self._default_strategy())
            logger.info("default_strategy_written", path=str(self.strategy_path))
        return self._default_strategy()

    def _read_progress(self) -> dict:
        data = _read_json(self.progress_path)
        if isinstance(data, dict):
            return data
        return {
            "last_wakeup": None,
            "last_actions": [],
            "last_outlook": "unknown",
            "peak_equity": None,
            "day": None,
            "day_start_equity": None,
            "notes": "First wakeup — no prior history.",
        }

    def _write_progress(self, progress: dict) -> None:
        self._write_json(self.progress_path, progress)

    def _append_journal(self, entry: dict) -> None:
        with open(self.journal_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def _append_trade(self, trade: dict) -> None:
        with open(self.trades_path, "a") as f:
            f.write(json.dumps(trade, default=str) + "\n")

    def _write_json(self, path: Path, data: dict) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)

    # ----- Context building -----

    def _current_prices(self, pairs: list[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        for pair in pairs:
            try:
                ticker = self.exchange.get_ticker(pair)
                if ticker.get("last", 0) > 0:
                    prices[pair] = float(ticker["last"])
            except Exception as e:
                logger.warning("price_fetch_failed", pair=pair, error=str(e))
        return prices

    def _get_equity_and_positions(
        self, pairs: list[str]
    ) -> tuple[float, list[dict], dict[str, float]]:
        """Unified account view across paper and live modes."""
        prices = self._current_prices(pairs)

        if self.mode == "paper" and self.ledger is not None:
            # update_peak persists the high-water mark; the return is unused.
            self.ledger.update_peak(prices)
            equity = self.ledger.equity(prices)
            positions = self.ledger.open_positions_list()
            for pos in positions:
                price = prices.get(pos["pair"], pos["entry_price"])
                direction = 1.0 if pos["side"] == "long" else -1.0
                if pos["entry_price"]:
                    pos["unrealized_pnl"] = round(
                        direction * (price - pos["entry_price"]) * pos["size"], 2
                    )
                else:
                    pos["unrealized_pnl"] = 0.0
            return equity, positions, prices

        # live / dry_run: read from the exchange
        try:
            balance = self.exchange.get_balance()
            equity = float(balance.get("total", 0))
        except Exception as e:
            logger.error("balance_fetch_failed", error=str(e))
            equity = 0.0
        try:
            positions = self.exchange.get_positions()
        except Exception as e:
            logger.error("positions_fetch_failed", error=str(e))
            positions = []
        return equity, positions, prices

    def _build_portfolio_context(self, equity: float, positions: list[dict]) -> str:
        lines = [
            f"Total Equity: ${equity:,.2f}",
        ]
        if positions:
            lines.append(f"\nOpen Positions ({len(positions)}):")
            for pos in positions:
                side = "LONG" if pos.get("side") == "long" else "SHORT"
                lines.append(
                    f"  {pos.get('pair', '?')} {side} | Entry: ${pos.get('entry_price', 0):,.2f} | "
                    f"Size: ${abs(pos.get('notional', 0)):,.2f} | "
                    f"Unrealized P&L: ${pos.get('unrealized_pnl', 0):+,.2f}"
                )
        else:
            lines.append("\nNo open positions.")
        return "\n".join(lines)

    def _risk_snapshot(
        self, strategy: dict, progress: dict,
        equity: float, positions: list[dict],
        apply: bool = True,
    ) -> dict:
        """Real risk state — replaces the phantom risk.get_status() call.

        ``apply=False`` answers "what IS the state" with no side effects: no
        cooldown started, no baseline re-armed, nothing written to
        ``progress``, nothing logged. The backtest uses it to build a decision
        source's context, so the only wakeup that moves the halt is the
        production one.
        """
        rules = strategy.get("rules", {})
        max_dd = float(rules.get("max_drawdown_pct", 10.0))
        max_heat = float(rules.get("max_portfolio_heat_pct", 30.0))

        peak = progress.get("peak_equity")
        if self.mode == "paper" and self.ledger is not None:
            peak = self.ledger.data.get("peak_equity", peak)
        peak = float(peak) if peak else equity
        # The peak is a HIGH-WATER MARK in both modes, not a stored constant.
        # In paper the ledger raises it (update_peak) and this is a no-op; in
        # live there is no ledger, and the value in progress.json used to be
        # frozen at the first wakeup's equity — measured $97 -> $120 -> $150
        # with peak_equity pinned at 97.0. A halt measured from a stale baseline
        # is not protection, and a re-armed baseline that cannot rise is not a
        # new baseline, so raise it here where both modes pass through.
        peak = max(peak, equity)
        drawdown_pct = max(0.0, (peak - equity) / peak * 100) if peak > 0 else 0.0

        halt = self._drawdown_halt(
            rules, progress, equity, peak, drawdown_pct, apply=apply,
        )
        if halt["rearmed"]:
            # The halt just released by moving the baseline here, so the rest of
            # this wakeup measures drawdown from the new peak: zero.
            peak = equity
            drawdown_pct = 0.0

        heat_usd = sum(abs(float(p.get("notional", 0))) for p in positions)
        heat_pct = heat_usd / equity * 100 if equity > 0 else 0.0

        # Daily P&L: resets when the UTC date changes.
        today = self._now().strftime("%Y-%m-%d")
        day = progress.get("day")
        day_start = progress.get("day_start_equity")
        if day != today or day_start is None:
            day_start_equity = equity
        else:
            day_start_equity = float(day_start)
        daily_pnl = equity - day_start_equity

        consecutive_losses = (
            self.ledger.consecutive_losses()
            if self.mode == "paper" and self.ledger is not None
            else 0
        )

        return {
            "open_positions": len(positions),
            "equity": round(equity, 2),
            "portfolio_heat_pct": round(heat_pct, 2),
            "max_heat_pct": max_heat,
            "current_drawdown_pct": round(drawdown_pct, 2),
            "max_drawdown_pct": max_dd,
            "peak_equity": peak,
            "daily_pnl": round(daily_pnl, 2),
            "day_start_equity": day_start_equity,
            "consecutive_losses": consecutive_losses,
            "trading_halted": halt["halted"],
            "dd_halt_reason": halt["reason"],
            "dd_cooldown_until": halt["cooldown_until"],
            "dd_rearm_count": halt["rearm_count"],
            "dd_rearm_window_start": halt["window_start"],
            "dd_rearm_budget_left": halt["budget_left"],
            "dd_rearmed": halt["rearmed"],
        }

    def _drawdown_halt(
        self, rules: dict, progress: dict, equity: float, peak: float,
        drawdown_pct: float, apply: bool = True,
    ) -> dict:
        """The drawdown halt's state: engage, hold, release, or RE-ARM.

        The guard stops new entries once equity is ``max_drawdown_pct`` below
        the peak. Blocking on price alone is a one-way door — a flat account
        cannot trade, so its equity cannot climb back above the line, so it
        stays halted for good. The harness measured exactly that: the trend
        source made its last trade on 2022-02-04, then refused 12,588 entries
        for the remaining four and a half years.

        So the halt is a COOLDOWN with a defined release, and only two ways out:

          * it engages when the drawdown reaches the limit and starts a
            ``dd_cooldown_hours`` flat period;
          * that period must be SERVED. A wiggle back inside the limit does not
            shorten it — measured before this rule, the account traded again
            within hours of a breach, ten times in one window;
          * served AND back inside the limit: the halt releases by itself, and
            no re-arm is spent;
          * served and still below the line: the peak is RE-ARMED to the
            current equity, so protection resumes from a new baseline;
          * a losing regime must not be re-entered without limit, so at most
            ``dd_rearm_limit`` re-arms are allowed per ``dd_rearm_window_days``.
            Spending that budget HOLDS the halt for another flat period, it
            never ends it: the window rolls over, and a recovery inside the
            limit still releases, so this can never deadlock.

        The re-arm is deliberately BOOK-AGNOSTIC: it does not wait for the
        position that was open when the halt engaged to close. This halt blocks
        entries and never force-closes (a force-close is a separate risk
        decision, and this bot does not make it), so requiring a flat book would
        hand the length of the halt to however long that position happens to
        live. What bounds a re-arm with exposure still open is the heat cap,
        which counts the open position's notional against the same 30%: a
        full-size position (30% of equity) leaves no room for another entry at
        all, and a smaller one leaves only the remainder. (The sibling bot in
        ``scripts/paper_trader.py`` can require a flat book because its drawdown
        stop CLOSES the book first; this halt has no closing branch, so it has
        nothing to wait for.)

        The risk and exposure caps are not touched by any branch. What limits
        a halt that can release is the cooldown (a flat period SERVED before
        any release), the budget (at most ``limit`` fresh 10% drawdowns per
        window), and the unchanged per-trade caps underneath both.

        ``apply=False`` computes the same state without moving anything.
        """
        max_dd = float(rules.get("max_drawdown_pct", 10.0))
        cooldown_hours = float(
            rules.get("dd_cooldown_hours", self.risk.dd_cooldown_hours)
        )
        rearm_limit = max(0, int(rules.get("dd_rearm_limit", 2)))
        window_days = float(rules.get("dd_rearm_window_days", 30.0))
        now = self._now()

        cooldown_until = _parse_iso(progress.get("dd_cooldown_until"))
        window_start = _parse_iso(progress.get("dd_rearm_window_start"))
        count = int(progress.get("dd_rearm_count") or 0)

        # Roll the re-arm budget window before anything reads it.
        if (window_start is None
                or (now - window_start).total_seconds() >= window_days * 86400):
            window_start, count = now, 0

        breach = drawdown_pct >= max_dd
        cooling = cooldown_until is not None and now < cooldown_until

        # The order matters. ``cooling`` is tested first so a breach and a
        # recovery are both overridden by a cool-off still being served; the
        # breach test comes next so a calm account can never reach the re-arm
        # branch (measured: it burned a re-arm and reset the peak on every
        # ordinary wakeup).
        state = {
            "halted": False,
            "reason": None,
            "cooldown_until": None,
            "rearmed": False,
            "rearm_count": count,
            "budget_left": max(0, rearm_limit - count),
        }

        if cooling:
            # The cool-off is still being served, breach or not: the halt is a
            # flat PERIOD, not a flag that a price wiggle can clear. Letting a
            # recovery cancel it here is what made 10 engage/clear cycles in a
            # single window, with no flat time served after any of them.
            state.update(halted=True, reason="cooldown")
        elif not breach:
            # Inside the limit with nothing owed: released if an episode was
            # running, a no-op otherwise. No re-arm is ever spent here, so this
            # path is always available to a recovering account.
            if cooldown_until is not None and apply:
                logger.warning(
                    "drawdown_halt_released",
                    drawdown_pct=round(drawdown_pct, 2), limit_pct=max_dd,
                )
            cooldown_until = None
        elif cooldown_until is None:
            # Fresh halt: stop entries and start the cool-off clock.
            cooldown_until = now + timedelta(hours=cooldown_hours)
            state.update(halted=True, reason="limit")
            if apply:
                logger.warning(
                    "drawdown_halt_engaged", drawdown_pct=round(drawdown_pct, 2),
                    limit_pct=max_dd, cooldown_hours=cooldown_hours,
                    resumes_at=cooldown_until.isoformat(),
                )
        elif count >= rearm_limit:
            # Served but the window's re-arm budget is spent. Hold for another
            # flat period rather than re-deciding every slot, and log that once.
            # Bounded twice over: the window rolls over, and a recovery inside
            # the limit takes the branch above instead.
            cooldown_until = now + timedelta(hours=cooldown_hours)
            state.update(halted=True, reason="budget")
            if apply:
                logger.warning(
                    "drawdown_halt_held", drawdown_pct=round(drawdown_pct, 2),
                    rearms_used=count, rearm_limit=rearm_limit,
                    window_days=window_days,
                    reconsider_at=cooldown_until.isoformat(),
                )
        else:
            # RE-ARM: new baseline, trade again.
            if apply:
                self._rearm_peak(equity)
            count += 1
            cooldown_until = None
            state.update(
                rearmed=True, rearm_count=count,
                budget_left=max(0, rearm_limit - count),
            )
            if apply:
                logger.warning(
                    "drawdown_halt_rearmed", drawdown_pct=round(drawdown_pct, 2),
                    new_peak=round(equity, 2), rearms_used=count,
                    rearm_limit=rearm_limit, window_days=window_days,
                )

        state["cooldown_until"] = (
            cooldown_until.isoformat() if cooldown_until else None
        )
        # Hand the state back through the progress document, which is what the
        # next wakeup reads: a halt that is not persisted is not a halt.
        if apply:
            progress["dd_cooldown_until"] = state["cooldown_until"]
            progress["dd_rearm_count"] = count
            progress["dd_rearm_window_start"] = window_start.isoformat()
            if state["rearmed"]:
                # The re-armed baseline, so the handoff document agrees with the
                # ledger and the next wakeup measures from here.
                progress["peak_equity"] = equity
        state["window_start"] = window_start.isoformat()
        return state

    def _rearm_peak(self, equity: float) -> None:
        """Move the high-water mark to ``equity`` (paper ledger and progress)."""
        if self.mode == "paper" and self.ledger is not None:
            self.ledger.rearm_peak(equity)

    def _build_risk_context(self, strategy: dict, snapshot: dict, progress: dict) -> str:
        rules = strategy.get("rules", {})
        lines = [
            "Hard limits enforced by the system (the AI cannot override these):",
            f"  Max risk per trade: {rules.get('max_risk_per_trade_pct', 2)}% of equity",
            f"  Max single position size: {rules.get('max_position_size_pct', 10)}% of equity",
            f"  Max portfolio heat: {rules.get('max_portfolio_heat_pct', 30)}%",
            f"  Max open positions: {rules.get('max_open_positions', 3)}",
            f"  Max drawdown: {rules.get('max_drawdown_pct', 10)}% (entries stop at this "
            f"level and stay flat for {rules.get('dd_cooldown_hours', 24)}h; after that "
            "they resume from a re-armed baseline, or on their own if the account "
            "is back inside the limit — an open position does not delay the re-arm, "
            "but it keeps counting against the exposure cap)",
            f"  Drawdown re-arm budget: {rules.get('dd_rearm_limit', 2)} per "
            f"{rules.get('dd_rearm_window_days', 30)} days",
            f"  Max stop-loss distance: {rules.get('stop_loss_max_pct', 5)}%",
            f"  Min risk/reward ratio: {rules.get('min_risk_reward_ratio', 1.5)}",
            f"  Min confidence to trade: {rules.get('min_confidence_to_trade', 60)}",
            "",
            "--- Current Risk State ---",
            f"Open positions: {snapshot['open_positions']}",
            f"Portfolio heat: {snapshot['portfolio_heat_pct']:.1f}%"
            f" of {snapshot['max_heat_pct']}%",
            f"Current drawdown: {snapshot['current_drawdown_pct']:.1f}%"
            f" of {snapshot['max_drawdown_pct']}%",
            f"Peak equity: ${snapshot['peak_equity']:,.2f}",
            f"Daily P&L: ${snapshot['daily_pnl']:+,.2f}",
            f"Consecutive losses: {snapshot['consecutive_losses']}",
            "TRADING HALTED (drawdown limit): " + _halt_summary(snapshot),
        ]

        # The venue's floor is a hard rule the AI cannot see from the risk
        # numbers alone, so state it in the sizing terms the AI reasons in.
        if self._limits:
            equity = float(snapshot.get("equity") or 0.0)
            lines.append("")
            lines.append(
                "--- Venue Minimums (the exchange REJECTS any order below "
                "these — size above them) ---"
            )
            for pair in sorted(self._limits):
                floor = min_legal_notional(self._limits[pair], MIN_TRADE_USD)
                if equity > 0:
                    lines.append(
                        f"  {pair}: min order ${floor:,.2f} "
                        f"(= {floor / equity * 100:.1f}% of current equity)"
                    )
                else:
                    lines.append(f"  {pair}: min order ${floor:,.2f}")
            lines.append(
                "  Leverage changes margin only — risk is notional x stop "
                "distance and is unaffected by it."
            )

        if progress.get("last_wakeup"):
            lines.append("")
            lines.append("--- Previous Wakeup Handoff ---")
            lines.append(f"Last outlook: {progress.get('last_outlook', 'unknown')}")
            notes = progress.get("notes", "")
            if notes:
                lines.append(f"Notes: {notes}")

        return "\n".join(lines)

    def _build_strategy_context(self, strategy: dict) -> str:
        prefs = strategy.get("preferences", {})
        lines = [
            f"Strategy: {strategy.get('name', 'AI Crypto Trader')}",
            f"Tradable pairs (ONLY these may be traded): {', '.join(strategy.get('pairs', []))}",
            f"Timeframe: {strategy.get('timeframe', '1h')}",
            f"Preferred pairs: {', '.join(prefs.get('preferred_pairs', []))}",
            f"Avoid high funding: {prefs.get('avoid_high_funding', True)}",
            f"Funding threshold: {prefs.get('funding_threshold_pct', 0.05)}%",
            f"Style: "
            f"{'trend following preferred' if prefs.get('prefer_trend_following') else 'flexible'}",
        ]
        return "\n".join(lines)

    # ----- Decision validation -----

    def _validate_decision(
        self,
        decision: TradingDecision,
        strategy: dict,
        positions: list[dict],
        snapshot: dict,
    ) -> tuple[list[TradeAction], list[str]]:
        """Validate AI decisions against hard risk rules.

        Returns (approved_actions, rejection_reasons).
        """
        rules = strategy.get("rules", {})
        allowed_pairs = set(strategy.get("pairs", self.pairs))
        max_positions = int(rules.get("max_open_positions", 3))
        min_confidence = float(rules.get("min_confidence_to_trade", 60))
        min_rr = float(rules.get("min_risk_reward_ratio", 1.5))
        max_sl = float(rules.get("stop_loss_max_pct", 5.0))
        max_size = float(rules.get("max_position_size_pct", 10.0))
        max_risk = float(rules.get("max_risk_per_trade_pct", 2.0))
        max_heat = float(rules.get("max_portfolio_heat_pct", 30.0))

        approved: list[TradeAction] = []
        rejected: list[str] = []
        open_pairs = {p.get("pair") for p in positions}
        # Heat already committed: the open book plus anything approved earlier
        # in THIS wakeup. Evaluating the limit once per run let two same-wakeup
        # entries each pass it and together breach the cap — the defect the main
        # bot's replay caught (see tests/test_bot_robustness.py), and the AI
        # path had the same shape.
        heat_used_pct = float(snapshot.get("portfolio_heat_pct") or 0.0)

        for action in decision.actions:
            def reject(reason: str) -> None:
                rejected.append(f"{action.pair} {action.side}: {reason}")
                logger.info("action_rejected", pair=action.pair, side=action.side,
                            reason=reason, confidence=action.confidence)

            if action.side not in VALID_SIDES:
                reject(f"invalid side {action.side!r}")
                continue
            if action.pair not in allowed_pairs:
                reject(f"pair not in tradable list {sorted(allowed_pairs)}")
                continue

            if action.side == "close":
                if action.pair not in open_pairs:
                    reject("no open position to close")
                    continue
                approved.append(action)
                continue

            # --- New entries ---
            if snapshot["trading_halted"]:
                # Same prefix as before, so the halt keeps its own rejection
                # bucket; the suffix says WHEN entries come back and WHY, which
                # during a cool-off is not a drawdown over the limit — quoting
                # "dd >= limit" there would be a false statement in the journal.
                until = snapshot.get("dd_cooldown_until")
                tail = f" (flat until {until})" if until else ""
                dd = snapshot["current_drawdown_pct"]
                cap = snapshot["max_drawdown_pct"]
                why = {
                    "cooldown": f"cool-off being served (drawdown {dd}% vs "
                                f"{cap}% limit)",
                    "budget": f"drawdown {dd}% >= limit {cap}% and the re-arm "
                              "budget is spent",
                }.get(snapshot.get("dd_halt_reason"), f"drawdown {dd}% >= limit {cap}%")
                reject(f"trading halted: {why}{tail}")
                continue
            if len(positions) >= max_positions:
                reject(f"max open positions ({max_positions}) reached")
                continue
            if action.pair in open_pairs:
                reject("position already open for this pair")
                continue
            if action.confidence < min_confidence:
                reject(f"confidence {action.confidence} < {min_confidence}")
                continue
            if action.stop_loss_pct <= 0 or action.stop_loss_pct > max_sl:
                reject(f"stop-loss {action.stop_loss_pct}% outside (0, {max_sl}%]")
                continue
            rr = action.take_profit_pct / max(action.stop_loss_pct, 1e-9)
            if action.take_profit_pct <= 0 or rr < min_rr:
                reject(f"risk/reward {rr:.2f} < {min_rr}")
                continue
            if action.size_pct <= 0 or action.size_pct > max_size:
                reject(f"size {action.size_pct}% outside (0, {max_size}%]")
                continue
            # The risk cap is enforced DIRECTLY rather than left implied by
            # "size cap x stop cap": at 40% and a 5% stop they happen to agree
            # (2%), and a later edit to either number must not be able to raise
            # the loss one trade can take without this firing.
            implied_risk_pct = action.size_pct * action.stop_loss_pct / 100.0
            if implied_risk_pct > max_risk + 1e-9:
                reject(
                    f"implied risk {implied_risk_pct:.2f}% of equity "
                    f"({action.size_pct:g}% position at a {action.stop_loss_pct:g}% "
                    f"stop) exceeds the {max_risk:g}% cap"
                )
                continue
            projected_heat_pct = heat_used_pct + action.size_pct
            if projected_heat_pct > max_heat + 1e-9:
                reject(
                    f"portfolio heat {projected_heat_pct:.1f}% "
                    f"(open {heat_used_pct:.1f}% + this {action.size_pct:g}%) would "
                    f"exceed the {max_heat:g}% exposure cap"
                )
                continue
            venue_reason = self._venue_rejection(action, strategy, snapshot)
            if venue_reason:
                reject(venue_reason)
                continue
            heat_used_pct = projected_heat_pct
            approved.append(action)

        return approved, rejected

    # ----- Trade execution -----

    # ----- Live execution safety (R1) -----
    #
    # Invariant: a live position may not exist without a confirmed protective
    # stop, and every failure mode ends in ONE defined outcome:
    #
    #   entry not accepted        -> refuse the entry (nothing was opened)
    #   fill unconfirmed          -> refuse the entry (nothing to protect)
    #   stop rejected / timed out -> flatten the filled size, then refuse
    #   partial fill              -> protect the FILLED size, not the request
    #   take-profit rejected      -> keep the (stop-protected) position, alert
    #   close rejected / error    -> retry, then alert; never report a close
    #                                that did not happen
    #   flatten also rejected     -> emergency alert; position may be naked
    #
    # Every one of these is recorded in the journal's ``errors`` list and on
    # Telegram, and the ones that leave the venue's answer in doubt block
    # further entries for the rest of the wakeup.

    def _record_live_failure(self, kind: str, message: str, **fields: Any) -> None:
        """Record a live execution fault: log, journal error, Telegram alert.

        A live fault that only reaches the log is invisible to the person
        driving a $97 account, so all three sinks are written every time.
        """
        logger.error(kind, **fields)
        self._execution_errors.append(message)
        try:
            self.notifier.notify_error(message, context=f"live {kind}")
        except Exception as e:  # a notifier must never break execution
            logger.warning("live_failure_notify_failed", error=str(e))

    def _live_position_size(self, pair: str) -> float:
        """Absolute size of any open position on ``pair`` (0.0 if none).

        Used when the venue does not report how much of an entry actually
        filled: the position itself is then the source of truth for what has
        to be protected.
        """
        try:
            positions = self.exchange.get_positions(pair)
        except Exception as e:
            logger.warning("live_position_read_failed", pair=pair, error=str(e)[:150])
            return 0.0
        for p in positions or []:
            if p.get("pair") == pair:
                try:
                    return abs(float(p.get("size", 0) or 0))
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def _flatten(self, pair: str, entry_side: str, amount: float) -> dict | None:
        """Reduce-only market order that closes ``amount`` of ``pair``.

        The defined outcome for an unprotected live position: the only
        position we are willing to hold is one with a confirmed stop, so an
        entry whose stop failed is taken off immediately.
        """
        close_side = "sell" if entry_side == "buy" else "buy"
        try:
            return self.exchange.place_market_order(
                pair, close_side, amount, reduce_only=True
            )
        except Exception as e:
            logger.error("flatten_order_failed", pair=pair, error=str(e)[:200])
            return None

    def _fee_pct(self, pair: str) -> float:
        limits = self._limits.get(pair)
        return limits.taker_fee_pct if limits else PAPER_TAKER_FEE_PCT

    def _abandon_unprotected_entry(
        self, pair: str, entry_side: str, filled: float, filled_usd: float,
        reason: str,
    ) -> None:
        """The ONE outcome for an entry we will not hold: flatten, then refuse.

        Called when the position that just opened cannot be given a protective
        stop. It is closed with a reduce-only market order; if even that is
        rejected the operator gets an emergency alert, because a naked live
        position is the one state this system must never sit in. Either way
        further entries are blocked until the next wakeup.
        """
        flattened = self._flatten(pair, entry_side, filled)
        self._entries_blocked_reason = f"{reason} ({pair})"
        if flattened:
            cost = round_trip_fee(filled_usd, self._fee_pct(pair))
            self._record_live_failure(
                "entry_flattened_unprotected",
                f"live entry FLATTENED for {pair}: {reason}. The "
                f"${filled_usd:,.2f} filled ({filled:g}) was closed immediately "
                f"at a round-trip cost of ~${cost:.4f}. No position was left "
                f"open. Further entries are blocked for this wakeup.",
                pair=pair, size=filled, size_usd=filled_usd,
            )
            return
        try:
            self.notifier.notify_emergency_stop(
                f"{pair} could not be protected AND could not be flattened \u2014 "
                f"a NAKED position may be open. Close it manually now."
            )
        except Exception as e:
            logger.warning("live_failure_notify_failed", error=str(e))
        self._record_live_failure(
            "entry_unprotected_unflattened",
            f"CRITICAL: live {pair} entry has no protective stop and the "
            f"flatten order was also rejected \u2014 a NAKED position may be "
            f"open. Manual intervention required. Further entries are blocked "
            f"for this wakeup. (Reason: {reason})",
            pair=pair, size=filled, size_usd=filled_usd,
        )

    def _execute_open(self, action: TradeAction, equity: float,
                      prices: dict[str, float]) -> dict | None:
        pair, order_side = action.pair, ORDER_SIDE[action.side]
        size_usd = equity * (action.size_pct / 100)

        price = prices.get(pair, 0)
        if price <= 0:
            logger.error("trade_no_price", pair=pair)
            return None

        # Quantise to the venue's lot step and refuse anything it would
        # reject. The check runs on the notional the exchange would actually
        # see: step rounding shaves value off, so sizing at exactly the
        # minimum is not enough to be legal.
        limits = self._limits.get(pair)
        qty = size_usd / price
        if limits is not None:
            qty = floor_to_step(qty, limits.amount_step)
            size_usd = qty * price
            problems = order_problems(limits, size_usd, qty)
            if problems:
                # Validation normally rejects this first with a reason the
                # journal and the digest can show; this is the backstop that
                # stops an unplaceable order being reported as a fill.
                logger.error(
                    "order_below_venue_minimum", pair=pair,
                    size_usd=round(size_usd, 2), problems=problems,
                )
                return None

        if size_usd < MIN_TRADE_USD:
            logger.info("trade_too_small", pair=pair, size_usd=round(size_usd, 2))
            return None

        if self.mode == "dry_run":
            logger.info("dry_run_order", pair=pair, side=order_side,
                        size_usd=round(size_usd, 2), price=price)
            return {
                "pair": pair, "side": action.side, "price": price,
                "size_usd": round(size_usd, 2),
                "stop_loss_pct": action.stop_loss_pct,
                "take_profit_pct": action.take_profit_pct,
                "confidence": action.confidence,
                "reasoning": action.reasoning,
                "mode": "dry_run", "executed": False,
            }

        if self.mode == "paper" and self.ledger is not None:
            pos = self.ledger.open_position(
                pair, order_side, price, size_usd,
                action.stop_loss_pct, action.take_profit_pct,
            )
            trade = {
                "pair": pair, "side": action.side, "price": price,
                "size_usd": round(size_usd, 2), "amount": pos["amount"],
                "stop_loss_pct": action.stop_loss_pct,
                "take_profit_pct": action.take_profit_pct,
                "confidence": action.confidence,
                "reasoning": action.reasoning,
                "mode": "paper", "executed": True,
                "timestamp": self._now_iso(),
            }
            self._append_trade(trade)
            self.notifier.notify_trade_open(
                pair=pair, side=order_side, price=price,
                amount=size_usd, confidence=action.confidence,
                strategy="ai_agent", mode="paper",
            )
            return trade

        if self.mode == "live":
            if self._entries_blocked_reason:
                logger.warning(
                    "live_entry_blocked", pair=pair,
                    reason=self._entries_blocked_reason,
                )
                return None

            # One quantised quantity for the entry and both triggers: they must
            # reduce exactly the position that was opened. A dropped connection
            # can hide a fill, so a raised call is treated exactly like a
            # refused one: the position is read back before anything is decided.
            try:
                order = self.exchange.place_market_order(pair, order_side, qty)
            except Exception as e:
                order = None
                logger.error("entry_order_raised", pair=pair, error=str(e)[:200])

            # The venue is the source of truth for how much actually filled.
            # Protective orders must cover the FILLED size, never the request:
            # a 60% fill with a full-size reduce-only stop is an order the
            # venue either rejects or cannot fully honour.
            filled = 0.0
            if order:
                try:
                    filled = float(order.get("filled") or 0)
                except (TypeError, ValueError):
                    filled = 0.0
            if filled <= 0:
                filled = self._live_position_size(pair)
            if limits is not None and filled > 0:
                filled = floor_to_step(filled, limits.amount_step)
            if filled <= 0:
                self._record_live_failure(
                    "order_failed" if order is None else "entry_fill_unconfirmed",
                    f"live entry REFUSED for {pair}: the market order "
                    f"{'was not accepted' if order is None else 'reported no fill'} "
                    f"and no open position is visible, so no position was opened.",
                    pair=pair, side=order_side,
                )
                return None

            entry_price = float((order or {}).get("average_price") or price)
            filled_usd = round(filled * entry_price, 2)

            # A partial fill can land below the venue's own minimum — at $97
            # that is easy, because a position is only ~$29. Then NO
            # protective order can be placed for it at all, and the only
            # honest outcome is to flatten and refuse the entry.
            if limits is not None and order_problems(
                limits, filled * entry_price, filled
            ):
                self._abandon_unprotected_entry(
                    pair, order_side, filled, filled_usd,
                    f"the filled size {filled:g} (${filled_usd:,.2f}) is below "
                    f"the venue minimum ${limits.min_notional:,.2f} and cannot "
                    f"carry a protective order",
                )
                return None

            # The stop goes on FIRST and is the invariant. A live position may
            # not exist without it, so nothing below this point can leave one
            # open: a stop that fails to place is answered by flattening.
            try:
                sl_placed = self.exchange.place_stop_market_order(
                    pair, order_side, filled,
                    self._trigger_price(
                        entry_price, action.stop_loss_pct, action.side, is_stop=True
                    ),
                )
            except Exception as e:
                sl_placed = None
                logger.error("stop_place_raised", pair=pair, error=str(e)[:200])

            if not sl_placed:
                self._abandon_unprotected_entry(
                    pair, order_side, filled, filled_usd,
                    "the protective stop was not accepted by the venue",
                )
                return None

            # Take-profit is best-effort: the STOP is what bounds the loss, so
            # a missing TP is recorded and alerted but does not force a
            # flatten \u2014 the position is still protected, and the AI can
            # close it on its next wakeup.
            try:
                tp_placed = self.exchange.place_take_profit_market_order(
                    pair, order_side, filled,
                    self._trigger_price(
                        entry_price, action.take_profit_pct, action.side, is_stop=False
                    ),
                )
            except Exception as e:
                tp_placed = None
                logger.error("take_profit_place_raised", pair=pair, error=str(e)[:200])
            if not tp_placed:
                self._record_live_failure(
                    "take_profit_missing",
                    f"live {pair} entry is protected by its stop but the "
                    f"take-profit was not accepted; the position will be "
                    f"closed by the AI instead of at a target.",
                    pair=pair, size=filled,
                )

            trade = {
                "pair": pair, "side": action.side,
                "price": entry_price,
                "size_usd": filled_usd,
                "amount": filled,
                "venue_min_notional": limits.min_notional if limits else None,
                "order_id": (order or {}).get("order_id"),
                "stop_loss_order": sl_placed.get("order_id"),
                "take_profit_order": tp_placed.get("order_id") if tp_placed else None,
                "stop_loss_pct": action.stop_loss_pct,
                "take_profit_pct": action.take_profit_pct,
                "confidence": action.confidence,
                "reasoning": action.reasoning,
                "mode": "live", "executed": True,
                "timestamp": self._now_iso(),
            }
            self._append_trade(trade)
            self.notifier.notify_trade_open(
                pair=pair, side=order_side, price=entry_price,
                amount=filled_usd, confidence=action.confidence,
                strategy="ai_agent", mode="live",
            )
            return trade

        return None

    @staticmethod
    def _trigger_price(entry: float, distance_pct: float, side: str, is_stop: bool) -> float:
        """Trigger price for SL/TP given entry, distance and direction.

        Long: stop below entry, take-profit above. Short: mirrored.
        """
        if side == "long":
            factor = -distance_pct if is_stop else distance_pct
        else:
            factor = distance_pct if is_stop else -distance_pct
        return entry * (1 + factor / 100)

    def _execute_close(self, action: TradeAction,
                       positions: list[dict], prices: dict[str, float]) -> dict | None:
        pair = action.pair
        pos = next((p for p in positions if p.get("pair") == pair), None)
        if not pos:
            logger.info("close_no_position", pair=pair)
            return None

        if self.mode == "dry_run":
            logger.info("dry_run_close", pair=pair)
            return {
                "pair": pair, "side": "close", "executed": False,
                "mode": "dry_run", "reasoning": action.reasoning,
                "confidence": action.confidence,
            }

        if self.mode == "paper" and self.ledger is not None:
            price = prices.get(pair, pos.get("entry_price", 0))
            rec = self.ledger.close_position(pair, price, f"AI close: {action.reasoning[:80]}")
            if not rec:
                return None
            trade = {
                "pair": pair, "side": "close",
                "entry_price": rec["entry_price"], "price": rec["exit_price"],
                "net_pnl": rec["net_pnl"], "pnl_pct": rec["pnl_pct"],
                "reasoning": action.reasoning,
                "confidence": action.confidence,
                "mode": "paper", "executed": True,
                "timestamp": self._now_iso(),
            }
            self._append_trade(trade)
            self.notifier.notify_trade_close(
                pair=pair, side=rec["position_side"],
                entry_price=rec["entry_price"], exit_price=rec["exit_price"],
                pnl_pct=rec.get("pnl_pct_net", rec["pnl_pct"]),
                pnl_usd=rec["net_pnl"],
                reason="AI close", mode="paper",
            )
            return trade

        if self.mode == "live":
            close_side = "sell" if pos.get("side") == "long" else "buy"
            amount = abs(float(pos.get("size", 0) or 0))
            if amount <= 0:
                logger.info("close_no_size", pair=pair)
                return None

            # A close is the safety-critical direction, so it is retried a
            # bounded number of times instead of leaving the position open on
            # one transient error.
            order = None
            for attempt in (1, 2, 3):
                try:
                    order = self.exchange.place_market_order(
                        pair, close_side, amount, reduce_only=True
                    )
                except Exception as e:
                    logger.error("close_order_raised", pair=pair,
                                 attempt=attempt, error=str(e)[:200])
                    order = None
                if order:
                    break
                if attempt < 3:
                    time.sleep(attempt)
            if not order:
                self._record_live_failure(
                    "close_order_failed",
                    f"live close FAILED for {pair} after 3 attempts: the "
                    f"reduce-only market order was not accepted. The position "
                    f"is still open and still protected by its stop.",
                    pair=pair, amount=amount,
                )
                return None

            # A market close that did not take the whole position leaves a
            # remainder whose stop must NOT be cancelled, or the remainder
            # would be naked. Keep closing what the venue says is left, a
            # bounded number of times: the order's OWN reported fill drives the
            # loop, and a position read is used only when the venue reports no
            # fill at all (a lagging read must not trigger a redundant order).
            try:
                filled = float(order.get("filled") or 0)
            except (TypeError, ValueError):
                filled = 0.0
            remaining = amount if filled <= 0 else max(0.0, amount - filled)
            attempts_left = 3
            while remaining > amount * 0.01 and attempts_left > 0:
                attempts_left -= 1
                try:
                    more = self.exchange.place_market_order(
                        pair, close_side, remaining, reduce_only=True
                    )
                except Exception as e:
                    logger.error("close_remainder_raised", pair=pair,
                                 error=str(e)[:200])
                    more = None
                if not more:
                    break
                try:
                    took = float(more.get("filled") or 0)
                except (TypeError, ValueError):
                    took = 0.0
                if took > 0:
                    remaining = max(0.0, remaining - took)
                else:
                    remaining = self._live_position_size(pair)
                if attempts_left:
                    time.sleep(1)

            if remaining > amount * 0.01:
                self._record_live_failure(
                    "close_incomplete",
                    f"live close INCOMPLETE for {pair}: ~{remaining:g} of the "
                    f"position is still open. Its protective orders were left "
                    f"in place, so the remainder is not naked.",
                    pair=pair, remaining=remaining,
                )
                return None

            # Confirmed flat: only now retire any leftover protective orders.
            try:
                self.exchange.cancel_all_orders(pair)
            except Exception as e:
                logger.warning("cancel_protective_failed", pair=pair, error=str(e))
            trade = {
                "pair": pair, "side": "close",
                "price": order.get("average_price", 0), "amount": amount,
                "order_id": order.get("order_id"),
                "reasoning": action.reasoning,
                "confidence": action.confidence,
                "mode": "live", "executed": True,
                "timestamp": self._now_iso(),
            }
            self._append_trade(trade)
            return trade

        return None

    def process_triggers(self, pairs: list[str]) -> list[dict]:
        """Paper mode: check SL/TP triggers and close hit positions."""
        if self.mode != "paper" or self.ledger is None:
            return []
        if not self.ledger.positions:
            return []
        prices = self._current_prices(list(self.ledger.positions.keys()) or pairs)
        closed = self.ledger.check_triggers(prices)
        for rec in closed:
            self._append_trade({
                "pair": rec["pair"], "side": "close",
                "entry_price": rec["entry_price"], "price": rec["exit_price"],
                "net_pnl": rec["net_pnl"], "pnl_pct": rec["pnl_pct"],
                "reason": rec["reason"], "mode": "paper",
                "triggered": True, "timestamp": self._now_iso(),
            })
            self.notifier.notify_trade_close(
                pair=rec["pair"], side=rec["position_side"],
                entry_price=rec["entry_price"], exit_price=rec["exit_price"],
                pnl_pct=rec.get("pnl_pct_net", rec["pnl_pct"]),
                pnl_usd=rec["net_pnl"],
                reason=rec["reason"], mode="paper",
            )
        return closed

    # ----- Main wakeup -----

    def run_wakeup(self) -> dict:
        """Run one complete wakeup cycle. Returns a summary dict.

        Every failure path journals and returns status="error" — nothing
        fails silently.
        """
        start_time = time.time()
        wakeup_id = self._now().strftime("%Y%m%d_%H%M%S")
        logger.info("wakeup_started", wakeup_id=wakeup_id, mode=self.mode)

        result: dict[str, Any] = {
            "wakeup_id": wakeup_id,
            "timestamp": self._now_iso(),
            "mode": self.mode,
            "status": "started",
            "actions_taken": [],
            "rejections": [],
            "closed_triggers": [],
            "errors": [],
            "total_actions": 0,
            "approved_actions": 0,
            "executed_trades": 0,
        }

        # Live-execution guard state is per-wakeup: a fault that blocked
        # entries last time must not silently persist, and must not be lost
        # from the journal either.
        self._execution_errors = []
        self._entries_blocked_reason = None

        strategy = self._read_strategy()
        pairs = list(strategy.get("pairs", self.pairs))
        progress = self._read_progress()

        # Pairs can come from strategy.json rather than the config, so refresh
        # the venue table for whatever is actually tradable this wakeup.
        self._venue_limits(pairs)
        # History recorded at a different account size is labelled, never
        # rewritten: its P&L and drawdown describe a different account.
        ledger_origin = (
            self.ledger.data.get("start_equity") if self.ledger is not None else None
        )
        scale_notice = legacy_scale_notice(ledger_origin, self.starting_equity)
        if scale_notice:
            logger.warning(
                "account_scale_mismatch",
                notice=scale_notice,
                ledger_start_equity=ledger_origin,
                configured_start_equity=self.starting_equity,
            )

        try:
            # 1. Paper SL/TP triggers first — protect existing positions.
            closed_triggers = self.process_triggers(pairs)
            result["closed_triggers"] = [
                {"pair": c["pair"], "reason": c["reason"],
                 "net_pnl": c["net_pnl"], "pnl_pct": c["pnl_pct"],
                 "pnl_pct_net": c.get("pnl_pct_net"),
                 "size_usd": c.get("size_usd")}
                for c in closed_triggers
            ]

            # 2. Account view
            equity, positions, prices = self._get_equity_and_positions(pairs)
            if equity <= 0 and self.mode == "live":
                raise RuntimeError("live account equity is 0 — check API keys/permissions")
            result["equity"] = round(equity, 2)

            # 3. Market data (with real ok-flag, no substring sniffing)
            market_context, data_ok = fetch_market_context(
                self.exchange, pairs, strategy.get("timeframe", self.timeframe)
            )
            if not data_ok:
                raise RuntimeError("market data unavailable for all pairs")

            portfolio_context = self._build_portfolio_context(equity, positions)
            strategy_context = self._build_strategy_context(strategy)

            snapshot = self._risk_snapshot(strategy, progress, equity, positions)
            # The halt state that GOVERNS this wakeup's decisions, captured
            # before anything executes. The halt is evaluated a second time after
            # execution (a fill moves equity, so that state can differ), and the
            # journal must describe the state the entries were decided under, not
            # the one after them: measured, a fill's fee can move equity across
            # the line and the journal then said "halted": true for a wakeup that
            # bought. An engagement after execution shows up in the next
            # wakeup's block, which is when it actually starts blocking.
            decision_halt = {
                "halted": snapshot["trading_halted"],
                "drawdown_pct": snapshot["current_drawdown_pct"],
                "peak_equity": round(snapshot["peak_equity"], 2),
                "cooldown_until": snapshot.get("dd_cooldown_until"),
                "rearms_used": snapshot.get("dd_rearm_count"),
                "rearmed_this_wakeup": snapshot.get("dd_rearmed"),
            }
            risk_context = self._build_risk_context(strategy, snapshot, progress)

            # 4. AI decision
            decision = self.ai.decide(
                market_context=market_context,
                portfolio_context=portfolio_context,
                strategy_context=strategy_context,
                risk_context=risk_context,
            )
            if not decision.ok:
                raise RuntimeError(f"AI engine failed: {decision.reasoning}")

            self._write_json(self.decisions_path, {
                "wakeup_id": wakeup_id,
                "decision": {
                    "actions": [
                        {
                            "pair": a.pair, "side": a.side,
                            "size_pct": a.size_pct,
                            "stop_loss_pct": a.stop_loss_pct,
                            "take_profit_pct": a.take_profit_pct,
                            "confidence": a.confidence,
                            "reasoning": a.reasoning,
                        }
                        for a in decision.actions
                    ],
                    "market_outlook": decision.market_outlook,
                    "risk_assessment": decision.risk_assessment,
                    "reasoning": decision.reasoning,
                },
                "model": decision.model,
                "decided_at": decision.decided_at,
            })

            # 5. Validate, then execute (closes first, then opens)
            approved, rejected = self._validate_decision(
                decision, strategy, positions, snapshot
            )
            result["rejections"] = rejected
            result["total_actions"] = len(decision.actions)
            result["approved_actions"] = len(approved)

            executed_trades: list[dict] = []
            closes = [a for a in approved if a.side == "close"]
            opens = [a for a in approved if a.side != "close"]
            for action in closes + opens:
                try:
                    if action.side == "close":
                        trade = self._execute_close(action, positions, prices)
                    else:
                        trade = self._execute_open(action, equity, prices)
                    if trade:
                        executed_trades.append(trade)
                except Exception as e:
                    msg = f"execution failed for {action.pair} {action.side}: {e}"
                    result["errors"].append(msg)
                    logger.error("trade_execution_error", error=msg)

            result["actions_taken"] = executed_trades
            result["executed_trades"] = len(executed_trades)
            # Live safety faults (a stop that failed, a flatten, a refused
            # close) belong in the journal's errors, not only in the log.
            result["errors"].extend(self._execution_errors)

            # 6. Refresh account state after execution
            equity, positions, prices = self._get_equity_and_positions(pairs)
            snapshot = self._risk_snapshot(strategy, progress, equity, positions)

            # 7. Handoff notes for the next wakeup
            next_progress = {
                "last_wakeup": wakeup_id,
                "last_actions": [
                    {
                        "pair": a.pair, "side": a.side,
                        "confidence": a.confidence,
                        "executed": any(t.get("pair") == a.pair for t in executed_trades),
                    }
                    for a in decision.actions
                ],
                "last_outlook": decision.market_outlook,
                "peak_equity": snapshot["peak_equity"],
                # The halt's own state travels with the handoff: a cooldown that
                # is not persisted is not a cooldown, and a re-armed baseline
                # must survive the next process.
                "dd_cooldown_until": snapshot.get("dd_cooldown_until"),
                "dd_rearm_count": snapshot.get("dd_rearm_count"),
                "dd_rearm_window_start": snapshot.get("dd_rearm_window_start"),
                "day": self._now().strftime("%Y-%m-%d"),
                "day_start_equity": snapshot["day_start_equity"],
                "notes": (
                    f"Wakeup {wakeup_id}: AI proposed {len(decision.actions)}, "
                    f"{len(approved)} approved, {len(executed_trades)} executed, "
                    f"{len(closed_triggers)} SL/TP triggers fired. "
                    f"Outlook: {decision.market_outlook}. Equity: ${equity:,.2f}."
                ),
            }
            self._write_progress(next_progress)

            # 8. Journal (audit trail)
            self._append_journal({
                "wakeup_id": wakeup_id,
                "timestamp": self._now_iso(),
                "mode": self.mode,
                "status": "success",
                "account_scale": {
                    "start_equity": ledger_origin,
                    "configured_start_equity": self.starting_equity,
                    "config_key": self.starting_equity_source,
                    "notice": scale_notice,
                },
                "venue_limits": {
                    pair: lim.as_dict()
                    for pair, lim in sorted(self._limits.items())
                },
                "market_outlook": decision.market_outlook,
                "risk_assessment": decision.risk_assessment,
                "ai_reasoning": decision.reasoning,
                "actions_requested": len(decision.actions),
                "actions_approved": len(approved),
                "actions_executed": len(executed_trades),
                # What the model actually proposed, kept for the audit trail.
                # Counts alone make the model un-auditable after the fact: a
                # decision that was rejected, or that never traded, leaves no
                # other trace, so the calls could never be replayed offline.
                "actions_proposed": [
                    {
                        "pair": a.pair, "side": a.side,
                        "size_pct": a.size_pct,
                        "stop_loss_pct": a.stop_loss_pct,
                        "take_profit_pct": a.take_profit_pct,
                        "confidence": a.confidence,
                        "reasoning": a.reasoning,
                    }
                    for a in decision.actions
                ],
                "rejections": rejected,
                "closed_triggers": result["closed_triggers"],
                # The halt's lifecycle, so a paused account is visible in the
                # audit trail rather than inferred from a wall of rejections.
                # The DECISION-time state (step 3), i.e. what the entries in
                # this same record were approved or refused under.
                "drawdown_halt": decision_halt,
                "equity": round(equity, 2),
                "errors": result["errors"],
            })

            result["status"] = "success"
            result["market_outlook"] = decision.market_outlook

        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.error("wakeup_failed", error=str(e), exc_info=True)
            self.notifier.notify_error(str(e), context=f"wakeup {wakeup_id}")
            # Always journal the failure — audit trail must never have gaps.
            try:
                self._append_journal({
                    "wakeup_id": wakeup_id,
                    "timestamp": self._now_iso(),
                    "mode": self.mode,
                    "status": "error",
                    "errors": result["errors"],
                    "closed_triggers": result["closed_triggers"],
                })
                self._write_progress({
                    **progress,
                    "last_wakeup": wakeup_id,
                    "notes": f"Wakeup {wakeup_id} FAILED: {result['errors'][0]}",
                })
            except Exception as journal_err:
                logger.critical("journal_write_failed", error=str(journal_err))

        result["elapsed_seconds"] = round(time.time() - start_time, 2)
        logger.info(
            "wakeup_completed", wakeup_id=wakeup_id, status=result["status"],
            elapsed=result["elapsed_seconds"],
            trades=len(result.get("actions_taken", [])),
        )
        return result

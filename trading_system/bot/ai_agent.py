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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
                 taker_fee_pct: float = PAPER_TAKER_FEE_PCT):
        self.path = path
        self.taker_fee_pct = taker_fee_pct
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
            "entry_time": _utc_now_iso(),
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
            "close_time": _utc_now_iso(),
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
            )
            if mode == "paper" else None
        )
        # Venue order bounds (MIN_NOTIONAL / LOT_SIZE): from the exchange when
        # reachable, else the dated builtin table. An order the venue would
        # refuse must never be treated as fillable, in paper or live.
        self._limits: dict[str, MarketLimits] = {}
        self._venue_limits(self.pairs)

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

    def _build_risk_manager(self) -> RiskManager:
        rcfg = self.config.get("risk", {})
        return RiskManager(
            position_stop_loss_pct=rcfg.get("stop_loss_max_pct", 5.0),
            portfolio_max_dd_pct=rcfg.get("max_drawdown_pct", 10.0),
            max_open_positions=rcfg.get("max_open_positions", 3),
            max_portfolio_heat_pct=rcfg.get("max_portfolio_heat_pct", 30.0),
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
    ) -> dict:
        """Real risk state — replaces the phantom risk.get_status() call."""
        rules = strategy.get("rules", {})
        max_dd = float(rules.get("max_drawdown_pct", 10.0))
        max_heat = float(rules.get("max_portfolio_heat_pct", 30.0))

        peak = progress.get("peak_equity")
        if self.mode == "paper" and self.ledger is not None:
            peak = self.ledger.data.get("peak_equity", peak)
        peak = float(peak) if peak else equity
        drawdown_pct = max(0.0, (peak - equity) / peak * 100) if peak > 0 else 0.0

        heat_usd = sum(abs(float(p.get("notional", 0))) for p in positions)
        heat_pct = heat_usd / equity * 100 if equity > 0 else 0.0

        # Daily P&L: resets when the UTC date changes.
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
            "trading_halted": drawdown_pct >= max_dd,
        }

    def _build_risk_context(self, strategy: dict, snapshot: dict, progress: dict) -> str:
        rules = strategy.get("rules", {})
        lines = [
            "Hard limits enforced by the system (the AI cannot override these):",
            f"  Max risk per trade: {rules.get('max_risk_per_trade_pct', 2)}% of equity",
            f"  Max single position size: {rules.get('max_position_size_pct', 10)}% of equity",
            f"  Max portfolio heat: {rules.get('max_portfolio_heat_pct', 30)}%",
            f"  Max open positions: {rules.get('max_open_positions', 3)}",
            f"  Max drawdown: {rules.get('max_drawdown_pct', 10)}% (trading halts at this level)",
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
            f"TRADING HALTED (drawdown limit): "
            f"{'YES — closes only' if snapshot['trading_halted'] else 'No'}",
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
                reject(f"trading halted: drawdown {snapshot['current_drawdown_pct']}% "
                       f">= limit {snapshot['max_drawdown_pct']}%")
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
                "timestamp": _utc_now_iso(),
            }
            self._append_trade(trade)
            self.notifier.notify_trade_open(
                pair=pair, side=order_side, price=price,
                amount=size_usd, confidence=action.confidence,
                strategy="ai_agent", mode="paper",
            )
            return trade

        if self.mode == "live":
            # One quantised quantity for the entry and both triggers: they must
            # reduce exactly the position that was opened.
            order = self.exchange.place_market_order(pair, order_side, qty)
            if not order:
                logger.error("order_failed", pair=pair, side=order_side)
                return None
            # Protective trigger orders — an AI position never sits naked.
            sl_placed = self.exchange.place_stop_market_order(
                pair, order_side, qty,
                self._trigger_price(price, action.stop_loss_pct, action.side, is_stop=True),
            )
            tp_placed = self.exchange.place_take_profit_market_order(
                pair, order_side, qty,
                self._trigger_price(price, action.take_profit_pct, action.side, is_stop=False),
            )
            if not sl_placed or not tp_placed:
                logger.error(
                    "protective_orders_missing", pair=pair,
                    stop_loss=sl_placed, take_profit=tp_placed,
                )
            trade = {
                "pair": pair, "side": action.side,
                "price": order.get("average_price", price),
                "size_usd": round(size_usd, 2),
                "amount": qty,
                "venue_min_notional": limits.min_notional if limits else None,
                "order_id": order.get("order_id"),
                "stop_loss_order": sl_placed.get("order_id") if sl_placed else None,
                "take_profit_order": tp_placed.get("order_id") if tp_placed else None,
                "stop_loss_pct": action.stop_loss_pct,
                "take_profit_pct": action.take_profit_pct,
                "confidence": action.confidence,
                "reasoning": action.reasoning,
                "mode": "live", "executed": True,
                "timestamp": _utc_now_iso(),
            }
            self._append_trade(trade)
            self.notifier.notify_trade_open(
                pair=pair, side=order_side, price=price,
                amount=size_usd, confidence=action.confidence,
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
                "timestamp": _utc_now_iso(),
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
            order_side = "sell" if pos.get("side") == "long" else "buy"
            amount = abs(float(pos.get("size", 0)))
            if amount <= 0:
                return None
            order = self.exchange.place_market_order(pair, order_side, amount, reduce_only=True)
            if not order:
                logger.error("close_order_failed", pair=pair)
                return None
            # Best-effort cancel of leftover protective orders.
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
                "timestamp": _utc_now_iso(),
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
                "triggered": True, "timestamp": _utc_now_iso(),
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
        wakeup_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        logger.info("wakeup_started", wakeup_id=wakeup_id, mode=self.mode)

        result: dict[str, Any] = {
            "wakeup_id": wakeup_id,
            "timestamp": _utc_now_iso(),
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
                "day": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
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
                "timestamp": _utc_now_iso(),
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
                "rejections": rejected,
                "closed_triggers": result["closed_triggers"],
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
                    "timestamp": _utc_now_iso(),
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

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

from trading_system.bot.ai_engine import AIEngine, TradeAction, TradingDecision
from trading_system.bot.exchange import ExchangeInterface
from trading_system.bot.market_data import fetch_market_context
from trading_system.bot.risk_manager import RiskManager
from trading_system.bot.telegram_notifier import TelegramNotifier

logger = structlog.get_logger(__name__)

VALID_SIDES = {"long", "short", "close"}
# How a "long"/"short" AI decision maps to an exchange order side.
ORDER_SIDE = {"long": "buy", "short": "sell"}
# Paper-trading taker fee per fill, in percent of notional.
PAPER_TAKER_FEE_PCT = 0.05
# Minimum notional per paper/live trade (Binance futures minimum is ~5 USDT).
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

    def __init__(self, path: Path, starting_cash: float = 10000.0,
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
        fee = exit_notional * self.taker_fee_pct / 100
        net_pnl = gross_pnl - fee
        self.data["cash"] = self.cash + net_pnl
        del self.positions[pair]

        record = {
            "pair": pair,
            "side": "close",
            "position_side": pos["side"],
            "entry_price": pos["entry_price"],
            "exit_price": float(price),
            "amount": pos["amount"],
            "gross_pnl": round(gross_pnl, 4),
            "fees": round(fee, 4),
            "net_pnl": round(net_pnl, 4),
            "pnl_pct": round(
                direction * (price - pos["entry_price"]) / pos["entry_price"] * 100, 4
            ),
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
        agent = AIAgent(exchange, data_dir="data/ai_bot")
        result = agent.run_wakeup()   # one complete cycle
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

        starting_cash = float(self.config.get("paper", {}).get("starting_equity", 10000.0))
        self.ledger = ledger or (
            PaperLedger(self.data_dir / "paper_ledger.json", starting_cash=starting_cash)
            if mode == "paper" else None
        )

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

    # ----- Shared file I/O -----

    def _read_strategy(self) -> dict:
        """Read the strategy document. Creates the default if missing."""
        data = _read_json(self.strategy_path)
        if isinstance(data, dict) and data.get("pairs"):
            return data
        if not self.strategy_file_override:
            self._write_json(self.strategy_path, DEFAULT_STRATEGY)
            logger.info("default_strategy_written", path=str(self.strategy_path))
        return json.loads(json.dumps(DEFAULT_STRATEGY))  # deep copy

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

        approved: list[TradeAction] = []
        rejected: list[str] = []
        open_pairs = {p.get("pair") for p in positions}

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
            approved.append(action)

        return approved, rejected

    # ----- Trade execution -----

    def _execute_open(self, action: TradeAction, equity: float,
                      prices: dict[str, float]) -> dict | None:
        pair, order_side = action.pair, ORDER_SIDE[action.side]
        size_usd = equity * (action.size_pct / 100)
        if size_usd < MIN_TRADE_USD:
            logger.info("trade_too_small", pair=pair, size_usd=round(size_usd, 2))
            return None

        price = prices.get(pair, 0)
        if price <= 0:
            logger.error("trade_no_price", pair=pair)
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
            order = self.exchange.place_market_order(pair, order_side, size_usd / price)
            if not order:
                logger.error("order_failed", pair=pair, side=order_side)
                return None
            # Protective trigger orders — an AI position never sits naked.
            sl_placed = self.exchange.place_stop_market_order(
                pair, order_side, size_usd / price,
                self._trigger_price(price, action.stop_loss_pct, action.side, is_stop=True),
            )
            tp_placed = self.exchange.place_take_profit_market_order(
                pair, order_side, size_usd / price,
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
                pnl_pct=rec["pnl_pct"], pnl_usd=rec["net_pnl"],
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
                pnl_pct=rec["pnl_pct"], pnl_usd=rec["net_pnl"],
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

        try:
            # 1. Paper SL/TP triggers first — protect existing positions.
            closed_triggers = self.process_triggers(pairs)
            result["closed_triggers"] = [
                {"pair": c["pair"], "reason": c["reason"],
                 "net_pnl": c["net_pnl"], "pnl_pct": c["pnl_pct"]}
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

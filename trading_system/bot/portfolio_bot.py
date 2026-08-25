"""
Portfolio-Aware Trading Bot.

Aggregates signals from multiple strategies using weighted voting.
Each strategy has an allocation weight that determines its influence
on the final trading decision and its share of the portfolio.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from trading_system.bot.exchange import ExchangeInterface
from trading_system.bot.notification import NotificationManager
from trading_system.bot.telegram_notifier import TelegramNotifier
from trading_system.bot.risk import RiskManager
from trading_system.bot.sltp_manager import SLTPManager
from trading_system.bot.state import BotState
from trading_system.config import SystemConfig, BotConfig
from trading_system.strategies import get_strategy

logger = structlog.get_logger(__name__)


class StrategyInstance:
    """A single strategy instance bound to a pair/timeframe with its params and weight."""

    def __init__(self, config: dict):
        self.name: str = config["name"]
        self.pair: str = config["pair"]
        self.timeframe: str = config["timeframe"]
        self.weight: float = config.get("weight", 1.0)
        self.params: dict = config.get("params", {})
        self.strategy = get_strategy(self.name)
        self.label = f"{self.name}_{self.pair.split('/')[0]}_{self.timeframe}"

    def generate_signal(self, df: pd.DataFrame) -> int:
        """Generate the latest signal (-1, 0, or 1)."""
        signals = self.strategy.generate_signals(df, self.params)
        return int(signals.iloc[-1]) if len(signals) > 0 else 0

    def __repr__(self):
        return f"StrategyInstance({self.label}, w={self.weight:.1%})"


class PortfolioTradingBot:
    """
    Portfolio trading bot that combines signals from multiple strategies.

    Signal aggregation:
      - Each strategy produces -1 (short), 0 (flat), or +1 (long)
      - Weighted score = sum(weight_i * signal_i) for each pair
      - If weighted_score > threshold: go long
      - If weighted_score < -threshold: go short
      - Otherwise: stay flat

    Position sizing:
      - Each strategy's share = equity * weight * risk_per_trade
      - Total exposure capped at max_portfolio_exposure
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.bot_config = config.bot
        self.portfolio_config = config.bot  # portfolio is embedded in bot config

        # Initialize components
        self.exchange = ExchangeInterface(config.exchange)
        self.risk = RiskManager(config.risk)
        self.state = BotState(config.bot.state_file)
        self.notifications = NotificationManager(
            config.bot.notification_enabled,
            config.bot.notification_webhook,
        )

        # Telegram notifier (if configured)
        telegram_cfg = getattr(config.bot, 'telegram', None)
        self.telegram = TelegramNotifier(
            bot_token=getattr(telegram_cfg, 'bot_token', '') if telegram_cfg else '',
            chat_id=getattr(telegram_cfg, 'chat_id', '') if telegram_cfg else '',
            enabled=getattr(telegram_cfg, 'enabled', False) if telegram_cfg else False,
        )

        # Build strategy instances from portfolio config
        self.strategy_instances: list[StrategyInstance] = []
        self._load_strategies()

        # SL manager (3x ATR, disaster protection only)
        self.sltp = SLTPManager(sl_atr_mult=3.0)

        # Signal threshold (weighted score must exceed this to trade)
        self.signal_threshold = 0.3

        # State tracking
        self._running = False
        self._last_signals: dict[str, int] = {}
        self._trade_log: list[dict] = []

    def _load_strategies(self):
        """Load strategy instances from the YAML config."""
        # Try to load from portfolio config in YAML
        portfolio_file = Path("configs/bot_live.yaml")
        if portfolio_file.exists():
            import yaml
            with open(portfolio_file) as f:
                raw = yaml.safe_load(f) or {}

            strategies_cfg = raw.get("portfolio", {}).get("strategies", [])
            for s_cfg in strategies_cfg:
                try:
                    instance = StrategyInstance(s_cfg)
                    self.strategy_instances.append(instance)
                    logger.info("strategy_loaded",
                                name=instance.label, weight=instance.weight)
                except Exception as e:
                    logger.error("strategy_load_failed", config=s_cfg, error=str(e))

        if not self.strategy_instances:
            # Fallback: use defaults
            defaults = [
                {"name": "MACD", "pair": "ETH/USDT:USDT", "timeframe": "4h",
                 "weight": 0.407, "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False}},
                {"name": "ROC_Momentum", "pair": "ETH/USDT:USDT", "timeframe": "4h",
                 "weight": 0.167, "params": {"roc_period": 3, "roc_threshold": -1, "smooth_period": 1, "trend_ema": 50, "trend_filter": False}},
                {"name": "MACD", "pair": "BTC/USDT:USDT", "timeframe": "4h",
                 "weight": 0.426, "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False}},
            ]
            for s_cfg in defaults:
                self.strategy_instances.append(StrategyInstance(s_cfg))

        total_weight = sum(s.weight for s in self.strategy_instances)
        logger.info("portfolio_loaded",
                     n_strategies=len(self.strategy_instances),
                     total_weight=total_weight)

    def _get_strategies_for_pair(self, pair: str) -> list[StrategyInstance]:
        """Get all strategy instances for a given pair."""
        return [s for s in self.strategy_instances if s.pair == pair]

    def aggregate_signals(self, pair: str, data: dict[str, pd.DataFrame]) -> dict:
        """
        Aggregate signals from all strategies for a pair.

        Returns dict with:
          - direction: 1 (long), -1 (short), 0 (flat)
          - weighted_score: the weighted sum
          - signals: per-strategy signals
          - confidence: how strong the consensus is
        """
        strategies = self._get_strategies_for_pair(pair)
        if not strategies:
            return {"direction": 0, "weighted_score": 0, "signals": {}, "confidence": 0}

        signals = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for si in strategies:
            tf = si.timeframe
            df = data.get(tf)
            if df is None or df.empty or len(df) < 50:
                signals[si.label] = 0
                continue

            signal = si.generate_signal(df)
            signals[si.label] = signal
            weighted_sum += si.weight * signal
            total_weight += si.weight

        # Normalize if some strategies didn't have data
        if total_weight > 0:
            weighted_sum /= total_weight

        # Determine direction
        if weighted_sum > self.signal_threshold:
            direction = 1
        elif weighted_sum < -self.signal_threshold:
            direction = -1
        else:
            direction = 0

        # Confidence = how far from zero (0 to 1)
        confidence = min(abs(weighted_sum), 1.0)

        return {
            "direction": direction,
            "weighted_score": weighted_sum,
            "signals": signals,
            "confidence": confidence,
        }

    def start(self) -> None:
        """Start the trading bot."""
        logger.info("bot_starting",
                     mode=self.bot_config.mode,
                     n_strategies=len(self.strategy_instances))

        if self.bot_config.mode in ("live", "dry_run"):
            if not self.exchange.connect():
                logger.error("exchange_connection_failed")
                return
            for pair in self.config.exchange.pairs:
                leverage = int(self.config.backtest.execution.leverage)
                self.exchange.set_leverage(pair, leverage)

        self._running = True
        self.notifications.send(f"Portfolio bot started in {self.bot_config.mode} mode "
                                f"({len(self.strategy_instances)} strategies)")
        self.telegram.notify_bot_start(self.bot_config.mode, len(self.strategy_instances))

        try:
            while self._running:
                self._check_cycle()
                time.sleep(self.bot_config.check_interval_seconds)
        except KeyboardInterrupt:
            logger.info("bot_stopped_by_user")
            self.telegram.notify_bot_stop("User interrupt")
        except Exception as e:
            logger.error("bot_error", error=str(e))
            self.notifications.notify_error(str(e))
            self.telegram.notify_error(str(e))
        finally:
            self._running = False
            self.state.save()
            logger.info("bot_shutdown")

    def stop(self) -> None:
        """Stop the trading bot."""
        self._running = False

    def _check_cycle(self) -> None:
        """One iteration of the main trading loop."""
        now = datetime.now(timezone.utc)
        logger.info("check_cycle", time=now.isoformat())

        for pair in self.config.exchange.pairs:
            try:
                self._process_pair(pair, now)
            except Exception as e:
                logger.error("pair_error", pair=pair, error=str(e))
                self.notifications.notify_error(f"Error on {pair}: {e}")

    def _process_pair(self, pair: str, now: datetime) -> None:
        """Process a single trading pair."""
        strategies = self._get_strategies_for_pair(pair)
        if not strategies:
            return

        # Fetch data for each timeframe used by strategies
        timeframes = list(set(s.timeframe for s in strategies))
        data = {}
        for tf in timeframes:
            df = self.exchange.get_ohlcv(pair, tf, limit=250)
            if not df.empty:
                data[tf] = df

        if not data:
            return

        # Check for abnormal price movement on the primary timeframe
        primary_tf = timeframes[0]
        if primary_tf in data and len(data[primary_tf]) >= 2:
            df = data[primary_tf]
            price_change = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2]
            self.risk.check_abnormal_conditions(price_change)

        # ── SL Check FIRST (before new signal generation) ──────
        current_price = float(data[primary_tf]["close"].iloc[-1]) if primary_tf in data else 0
        sltp_actions = self.sltp.check_price(pair, current_price)
        for action in sltp_actions:
            if action["action"] == "close":
                logger.info("sltp_closing", pair=pair, reason=action["reason"],
                            pnl=action.get("pnl_pct", 0))
                # Create a fake position dict for close_position
                fake_pos = {
                    "pair": pair,
                    "side": "long" if action["side"] == "sell" else "short",
                    "size": 0,  # Will be handled by paper/live execution
                    "notional": 0,
                }
                self._close_position(pair, fake_pos, action["reason"])
                self.state.record_trade({
                    "pair": pair, "side": action["side"],
                    "mode": self.bot_config.mode,
                    "action": "sltp_close",
                    "reason": action["reason"],
                    "entry": action.get("entry", 0),
                    "exit_price": action.get("exit", 0),
                    "pnl_pct": action.get("pnl_pct", 0),
                    "strategy": action.get("strategy", "portfolio"),
                })

        # ── Aggregate signals ──────────────────────────────────────
        agg = self.aggregate_signals(pair, data)

        logger.info("signals_aggregated",
                     pair=pair,
                     direction=agg["direction"],
                     score=agg["weighted_score"],
                     confidence=agg["confidence"],
                     signals=agg["signals"])

        # Get current state
        positions = self.exchange.get_positions(pair)
        ticker = self.exchange.get_ticker(pair)
        balance = self.exchange.get_balance()
        current_equity = balance.get("total", 0)
        self.risk.update(current_equity)

        # Execute based on aggregated signal
        self._execute_portfolio_signal(
            pair=pair,
            agg_signal=agg,
            ticker=ticker,
            equity=current_equity,
            current_positions=positions,
        )

        # Record signal for state tracking
        self.state.set("last_signals", {
            **self.state.get("last_signals", {}),
            pair: {
                "direction": agg["direction"],
                "score": agg["weighted_score"],
                "signals": agg["signals"],
                "timestamp": now.isoformat(),
            }
        })

    def _execute_portfolio_signal(
        self,
        pair: str,
        agg_signal: dict,
        ticker: dict[str, float],
        equity: float,
        current_positions: list[dict],
    ) -> None:
        """Execute trading based on aggregated portfolio signal."""
        direction = agg_signal["direction"]
        confidence = agg_signal["confidence"]

        existing = [p for p in current_positions if p["pair"] == pair]

        if direction == 0:
            # Close any existing position (signal says flat)
            for pos in existing:
                self._close_position(pair, pos, "portfolio_exit")
            return

        is_long = direction > 0

        if existing:
            pos = existing[0]
            pos_is_long = pos["side"] == "long"

            if (is_long and pos_is_long) or (not is_long and not pos_is_long):
                # Already in the right direction
                return

            # Close and reverse
            self._close_position(pair, pos, "portfolio_reverse")

        # Open new position
        self._open_position(pair, is_long, equity, ticker, confidence)

    def _open_position(
        self,
        pair: str,
        is_long: bool,
        equity: float,
        ticker: dict[str, float],
        confidence: float,
    ) -> None:
        """Open a new position with portfolio-aware sizing."""
        # Position sizing: base risk * confidence adjustment
        risk_pct = self.config.risk.max_risk_per_trade
        # Scale down position for low confidence
        adjusted_risk = risk_pct * min(confidence, 1.0)
        notional = equity * adjusted_risk * self.config.backtest.execution.leverage
        notional = min(notional, equity * self.config.risk.max_order_size)

        side = "buy" if is_long else "sell"
        price = ticker.get("ask" if is_long else "bid", ticker.get("last", 0))

        if price <= 0 or notional <= 0:
            return

        amount = notional / price

        # Risk check
        current_exposure = sum(abs(p.get("notional", 0))
                               for p in self.exchange.get_positions(pair))
        can_open, reason = self.risk.can_open_position(notional, current_exposure, equity)

        if not can_open:
            logger.warning("position_rejected", pair=pair, reason=reason)
            return

        if self.bot_config.mode == "paper":
            self._paper_execute(pair, side, amount, price, confidence)
            self.telegram.notify_trade_open(
                pair=pair, side=side, price=price, amount=amount,
                confidence=confidence, strategy="portfolio", mode="paper",
            )
            # Register with SL manager
            primary_tf = strategies[0].timeframe if strategies else "4h"
            df = data.get(primary_tf)
            if df is not None and not df.empty:
                self.sltp.open_position(
                    pair=pair, side="long" if is_long else "short",
                    entry_price=price, size=amount,
                    strategy="portfolio", df=df,
                )
        elif self.bot_config.mode == "live":
            self._live_execute(pair, side, amount, price, confidence)
            self.telegram.notify_trade_open(
                pair=pair, side=side, price=price, amount=amount,
                confidence=confidence, strategy="portfolio", mode="live",
            )
            primary_tf = strategies[0].timeframe if strategies else "4h"
            df = data.get(primary_tf)
            if df is not None and not df.empty:
                self.sltp.open_position(
                    pair=pair, side="long" if is_long else "short",
                    entry_price=price, size=amount,
                    strategy="portfolio", df=df,
                )
        elif self.bot_config.mode == "dry_run":
            logger.info("dry_run_order", pair=pair, side=side, amount=amount,
                        price=price, confidence=confidence)

    def _paper_execute(self, pair, side, amount, price, confidence):
        """Simulate order execution for paper trading."""
        logger.info("paper_order", pair=pair, side=side, amount=amount,
                    price=price, confidence=f"{confidence:.2f}")
        trade = {
            "pair": pair, "side": side, "amount": amount,
            "price": price, "mode": "paper",
            "confidence": confidence,
            "strategy": "portfolio",
        }
        self.state.record_trade(trade)
        self._trade_log.append({**trade, "timestamp": datetime.now(timezone.utc).isoformat()})
        self.notifications.notify_trade({
            "pair": pair, "side": side, "price": price, "pnl": 0,
        })

    def _live_execute(self, pair, side, amount, price, confidence):
        """Execute real order on exchange."""
        order = self.exchange.place_market_order(pair, side, amount)
        if order:
            trade = {
                "pair": pair, "side": side, "amount": amount,
                "price": order.get("average_price", price),
                "order_id": order["order_id"],
                "mode": "live", "strategy": "portfolio",
                "confidence": confidence,
            }
            self.state.record_order(order["order_id"])
            self.state.record_trade(trade)
            self._trade_log.append({**trade, "timestamp": datetime.now(timezone.utc).isoformat()})
            self.notifications.notify_trade({
                "pair": pair, "side": side,
                "price": order.get("average_price", price), "pnl": 0,
            })

    def _close_position(self, pair: str, position: dict, reason: str):
        """Close an existing position."""
        side = "sell" if position["side"] == "long" else "buy"
        amount = abs(position.get("size", 0))

        if amount <= 0:
            return

        logger.info("closing_position", pair=pair, side=side, amount=amount, reason=reason)

        entry_price = position.get("entry_price", 0)
        exit_price = position.get("mark_price", 0)
        pnl_pct = position.get("unrealized_pnl_pct", 0)
        pnl_usd = position.get("unrealized_pnl", 0)

        if self.bot_config.mode == "paper":
            logger.info("paper_close", pair=pair, side=side, amount=amount)
            self.state.record_trade({
                "pair": pair, "side": side, "amount": amount,
                "mode": "paper", "action": "close", "reason": reason,
            })
            self.telegram.notify_trade_close(
                pair=pair, side=side, entry_price=entry_price,
                exit_price=exit_price, pnl_pct=pnl_pct, pnl_usd=pnl_usd,
                reason=reason, mode="paper",
            )
        elif self.bot_config.mode == "live":
            order = self.exchange.place_market_order(pair, side, amount, reduce_only=True)
            if order:
                exit_price = order.get("average_price", exit_price)
                self.state.record_trade({
                    "pair": pair, "side": side, "amount": amount,
                    "price": exit_price,
                    "order_id": order["order_id"],
                    "mode": "live", "action": "close", "reason": reason,
                })
                self.telegram.notify_trade_close(
                    pair=pair, side=side, entry_price=entry_price,
                    exit_price=exit_price, pnl_pct=pnl_pct, pnl_usd=pnl_usd,
                    reason=reason, mode="live",
                )

    def get_status(self) -> dict:
        """Get current bot status including portfolio signals and SL/TP."""
        risk_status = self.risk.get_status()
        last_signals = self.state.get("last_signals", {})
        sltp_status = self.sltp.get_status()

        strategies_status = []
        for si in self.strategy_instances:
            strategies_status.append({
                "label": si.label,
                "pair": si.pair,
                "timeframe": si.timeframe,
                "weight": si.weight,
                "last_signal": last_signals.get(si.pair, {}).get("signals", {}).get(si.label, "N/A"),
            })

        return {
            "mode": self.bot_config.mode,
            "running": self._running,
            "n_strategies": len(self.strategy_instances),
            "strategies": strategies_status,
            "last_signals": last_signals,
            "risk": risk_status,
            "sltp": sltp_status,
            "total_trades": len(self._trade_log),
        }

    def run_once(self) -> dict:
        """
        Run a single check cycle (for testing / backtesting).
        Returns the aggregated signals for all pairs.
        """
        results = {}
        for pair in self.config.exchange.pairs:
            strategies = self._get_strategies_for_pair(pair)
            timeframes = list(set(s.timeframe for s in strategies))

            data = {}
            for tf in timeframes:
                df = self.exchange.get_ohlcv(pair, tf, limit=250)
                if not df.empty:
                    data[tf] = df

            agg = self.aggregate_signals(pair, data)
            results[pair] = agg

        return results

"""
Bot orchestrator — main trading loop.

Handles:
- Market data ingestion
- Signal generation
- Order placement
- Position tracking
- Risk management
- State persistence
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import structlog

from trading_system.backtester.engine import BacktestEngine
from trading_system.bot.exchange import ExchangeInterface
from trading_system.bot.notification import NotificationManager
from trading_system.bot.risk import RiskManager
from trading_system.bot.state import BotState
from trading_system.config import BotConfig, BacktestConfig, SystemConfig
from trading_system.strategies import get_strategy

logger = structlog.get_logger(__name__)


class TradingBot:
    """Main trading bot that orchestrates data, signals, and execution."""

    def __init__(
        self,
        config: SystemConfig,
        strategy_names: list[str],
        strategy_params: dict[str, dict],
    ):
        self.config = config
        self.bot_config = config.bot

        # Initialize components
        self.exchange = ExchangeInterface(config.exchange)
        self.risk = RiskManager(config.risk)
        self.state = BotState(config.bot.state_file)
        self.notifications = NotificationManager(
            config.bot.notification_enabled,
            config.bot.notification_webhook,
        )

        # Strategies
        self.strategies = {}
        self.strategy_params = strategy_params
        for name in strategy_names:
            self.strategies[name] = get_strategy(name)

        # Position sizing
        self.position_sizing = config.backtest.execution.position_sizing

        self._running = False

    def start(self) -> None:
        """Start the trading bot."""
        logger.info("bot_starting", mode=self.bot_config.mode, strategies=list(self.strategies.keys()))

        # Connect to exchange
        if self.bot_config.mode in ("live", "dry_run"):
            if not self.exchange.connect():
                logger.error("exchange_connection_failed")
                return

            # Set leverage
            for pair in self.config.exchange.pairs:
                leverage = int(self.config.backtest.execution.leverage)
                self.exchange.set_leverage(pair, leverage)

        self._running = True
        self.notifications.send(f"Bot started in {self.bot_config.mode} mode")

        try:
            while self._running:
                self._check_cycle()
                time.sleep(self.bot_config.check_interval_seconds)
        except KeyboardInterrupt:
            logger.info("bot_stopped_by_user")
        except Exception as e:
            logger.error("bot_error", error=str(e))
            self.notifications.notify_error(str(e))
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

        for pair in self.config.exchange.pairs:
            try:
                self._process_pair(pair, now)
            except Exception as e:
                logger.error("pair_processing_error", pair=pair, error=str(e))
                self.notifications.notify_error(f"Error processing {pair}: {e}")

    def _process_pair(self, pair: str, now: datetime) -> None:
        """Process a single trading pair."""
        # Get current market data
        for timeframe in self.config.exchange.timeframes:
            df = self.exchange.get_ohlcv(pair, timeframe, limit=250)
            if df.empty:
                continue

            # Check for abnormal price movement
            if len(df) >= 2:
                price_change = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2]
                self.risk.check_abnormal_conditions(price_change)

            # Generate signals from each strategy
            all_signals = {}
            for name, strategy in self.strategies.items():
                params = self.strategy_params.get(name, {})
                signals = strategy.generate_signals(df, params)
                all_signals[name] = signals.iloc[-1] if len(signals) > 0 else 0

            # Get current position state
            positions = self.exchange.get_positions(pair)
            ticker = self.exchange.get_ticker(pair)
            balance = self.exchange.get_balance()

            # Process signals
            self._execute_signals(
                pair=pair,
                timeframe=timeframe,
                signals=all_signals,
                ticker=ticker,
                balance=balance,
                current_positions=positions,
            )

    def _execute_signals(
        self,
        pair: str,
        timeframe: str,
        signals: dict[str, int],
        ticker: dict[str, float],
        balance: dict[str, float],
        current_positions: list[dict],
    ) -> None:
        """Execute trading signals."""
        current_equity = balance.get("total", 0)
        self.risk.update(current_equity)

        for strategy_name, signal in signals.items():
            if signal == 0:
                continue

            # Check if we already have a position in this direction
            existing = [p for p in current_positions if p["pair"] == pair]
            is_long = signal > 0

            # Determine action
            if existing:
                pos = existing[0]
                pos_is_long = pos["side"] == "long"

                if (is_long and pos_is_long) or (not is_long and not pos_is_long):
                    continue  # Already in the right direction

                # Close existing and open opposite
                self._close_position(pair, pos, strategy_name)
            else:
                # Open new position
                self._open_position(pair, is_long, strategy_name, current_equity, ticker)

    def _open_position(
        self,
        pair: str,
        is_long: bool,
        strategy_name: str,
        equity: float,
        ticker: dict[str, float],
    ) -> None:
        """Open a new position."""
        # Calculate position size
        risk_pct = self.config.risk.max_risk_per_trade
        notional = equity * risk_pct * self.config.backtest.execution.leverage

        side = "buy" if is_long else "sell"
        price = ticker.get("ask" if is_long else "bid", ticker.get("last", 0))

        if price <= 0 or notional <= 0:
            return

        amount = notional / price

        # Risk check
        current_exposure = sum(abs(p.get("notional", 0)) for p in self.exchange.get_positions(pair))
        can_open, reason = self.risk.can_open_position(notional, current_exposure, equity)

        if not can_open:
            logger.warning("position_rejected", pair=pair, reason=reason)
            return

        if self.bot_config.mode == "paper":
            # Simulated execution
            logger.info("paper_order", pair=pair, side=side, amount=amount, price=price)
            self.state.record_trade({
                "pair": pair,
                "side": side,
                "amount": amount,
                "price": price,
                "strategy": strategy_name,
                "mode": "paper",
            })
            self.notifications.notify_trade({
                "pair": pair, "side": side, "price": price, "pnl": 0,
            })

        elif self.bot_config.mode == "live":
            order = self.exchange.place_market_order(pair, side, amount)
            if order:
                self.state.record_order(order["order_id"])
                self.state.record_trade({
                    "pair": pair,
                    "side": side,
                    "amount": amount,
                    "price": order.get("average_price", price),
                    "order_id": order["order_id"],
                    "strategy": strategy_name,
                    "mode": "live",
                })
                self.notifications.notify_trade({
                    "pair": pair, "side": side,
                    "price": order.get("average_price", price),
                    "pnl": 0,
                })

        elif self.bot_config.mode == "dry_run":
            # Log intended order but don't execute
            logger.info("dry_run_order", pair=pair, side=side, amount=amount, price=price)

    def _close_position(self, pair: str, position: dict, strategy_name: str) -> None:
        """Close an existing position."""
        side = "sell" if position["side"] == "long" else "buy"
        amount = abs(position.get("size", 0))

        if amount <= 0:
            return

        if self.bot_config.mode == "paper":
            logger.info("paper_close", pair=pair, side=side, amount=amount)

        elif self.bot_config.mode == "live":
            order = self.exchange.place_market_order(pair, side, amount, reduce_only=True)
            if order:
                self.state.record_trade({
                    "pair": pair,
                    "side": side,
                    "amount": amount,
                    "price": order.get("average_price", 0),
                    "strategy": strategy_name,
                    "mode": "live",
                    "action": "close",
                })

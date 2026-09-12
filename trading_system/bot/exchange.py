"""
Exchange interface using ccxt.

Wraps ccxt for unified access to Binance futures.
Handles order placement, position queries, and account info.
"""

from __future__ import annotations

import ccxt
import pandas as pd
import structlog

from trading_system.config import ExchangeConfig

logger = structlog.get_logger(__name__)


class ExchangeInterface:
    """Unified exchange interface using ccxt."""

    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.exchange = ccxt.binanceusdm({
            "apiKey": config.api_key or None,
            "secret": config.api_secret or None,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        if config.sandbox:
            self.exchange.set_sandbox_mode(True)

        self._connected = False

    def connect(self) -> bool:
        """Initialize exchange connection."""
        try:
            self.exchange.load_markets()
            self._connected = True
            logger.info("exchange_connected", exchange=self.config.name)
            return True
        except Exception as e:
            logger.error("exchange_connection_failed", error=str(e))
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_balance(self) -> dict[str, float]:
        """Get account balance."""
        try:
            balance = self.exchange.fetch_balance()
            return {
                "total": float(balance.get("total", {}).get("USDT", 0)),
                "free": float(balance.get("free", {}).get("USDT", 0)),
                "used": float(balance.get("used", {}).get("USDT", 0)),
            }
        except Exception as e:
            logger.error("balance_fetch_failed", error=str(e))
            return {"total": 0, "free": 0, "used": 0}

    def get_positions(self, pair: str = "") -> list[dict]:
        """Get open positions."""
        try:
            positions = self.exchange.fetch_positions([pair] if pair else None)
            return [
                {
                    "pair": p["symbol"],
                    "side": p["side"],
                    "size": float(p["contracts"] or 0),
                    "notional": float(p["notional"] or 0),
                    "entry_price": float(p["entryPrice"] or 0),
                    "unrealized_pnl": float(p["unrealizedPnl"] or 0),
                    "leverage": float(p["leverage"] or 1),
                    "liquidation_price": float(p["liquidationPrice"] or 0),
                }
                for p in positions
                if float(p.get("contracts", 0) or 0) != 0
            ]
        except Exception as e:
            logger.error("positions_fetch_failed", error=str(e))
            return []

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        """Coerce possibly-None/str ccxt values to float safely."""
        try:
            return float(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    def get_ticker(self, pair: str) -> dict[str, float]:
        """Get current ticker (tolerates None fields ccxt omits)."""
        try:
            ticker = self.exchange.fetch_ticker(pair)
            return {
                "bid": self._to_float(ticker.get("bid")),
                "ask": self._to_float(ticker.get("ask")),
                "last": self._to_float(ticker.get("last")),
                "volume": self._to_float(ticker.get("quoteVolume")),
            }
        except Exception as e:
            logger.error("ticker_fetch_failed", pair=pair, error=str(e))
            return {"bid": 0, "ask": 0, "last": 0, "volume": 0}

    def get_ohlcv(
        self,
        pair: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch recent OHLCV data."""
        try:
            candles = self.exchange.fetch_ohlcv(pair, timeframe, limit=limit)
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")
            return df
        except Exception as e:
            logger.error("ohlcv_fetch_failed", pair=pair, error=str(e))
            return pd.DataFrame()

    def set_leverage(self, pair: str, leverage: int) -> bool:
        """Set leverage for a pair."""
        try:
            self.exchange.set_leverage(leverage, pair)
            logger.info("leverage_set", pair=pair, leverage=leverage)
            return True
        except Exception as e:
            logger.error("leverage_set_failed", pair=pair, error=str(e))
            return False

    def place_market_order(
        self,
        pair: str,
        side: str,
        amount: float,
        reduce_only: bool = False,
    ) -> dict | None:
        """Place a market order."""
        try:
            params = {}
            if reduce_only:
                params["reduceOnly"] = True

            order = self.exchange.create_order(
                symbol=pair,
                type="market",
                side=side,
                amount=amount,
                params=params,
            )

            logger.info(
                "order_placed",
                pair=pair,
                side=side,
                amount=amount,
                order_id=order.get("id"),
                price=order.get("average"),
            )

            return {
                "order_id": order.get("id"),
                "status": order.get("status"),
                "filled": float(order.get("filled", 0)),
                "average_price": float(order.get("average", 0)),
                "fee": order.get("fee", {}),
            }

        except Exception as e:
            logger.error("order_failed", pair=pair, side=side, amount=amount, error=str(e))
            return None

    def place_limit_order(
        self,
        pair: str,
        side: str,
        amount: float,
        price: float,
        reduce_only: bool = False,
    ) -> dict | None:
        """Place a limit order."""
        try:
            params = {}
            if reduce_only:
                params["reduceOnly"] = True

            order = self.exchange.create_order(
                symbol=pair,
                type="limit",
                side=side,
                amount=amount,
                price=price,
                params=params,
            )

            return {
                "order_id": order.get("id"),
                "status": order.get("status"),
                "filled": float(order.get("filled", 0)),
                "price": price,
            }

        except Exception as e:
            logger.error("limit_order_failed", pair=pair, error=str(e))
            return None

    def cancel_order(self, order_id: str, pair: str) -> bool:
        """Cancel an order."""
        try:
            self.exchange.cancel_order(order_id, pair)
            logger.info("order_cancelled", order_id=order_id, pair=pair)
            return True
        except Exception as e:
            logger.error("cancel_failed", order_id=order_id, error=str(e))
            return False

    def place_stop_market_order(
        self,
        pair: str,
        entry_side: str,
        amount: float,
        stop_price: float,
    ) -> dict | None:
        """Place a reduce-only STOP_MARKET order (protective stop-loss).

        entry_side is the side that OPENED the position ("buy" for a long,
        "sell" for a short); the stop fires on the opposite side.
        """
        return self._trigger_order(
            pair, entry_side, amount, stop_price, "STOP_MARKET"
        )

    def place_take_profit_market_order(
        self,
        pair: str,
        entry_side: str,
        amount: float,
        stop_price: float,
    ) -> dict | None:
        """Place a reduce-only TAKE_PROFIT_MARKET order."""
        return self._trigger_order(
            pair, entry_side, amount, stop_price, "TAKE_PROFIT_MARKET"
        )

    def _trigger_order(
        self,
        pair: str,
        entry_side: str,
        amount: float,
        stop_price: float,
        order_type: str,
    ) -> dict | None:
        close_side = "sell" if entry_side == "buy" else "buy"
        try:
            order = self.exchange.create_order(
                symbol=pair,
                type=order_type,
                side=close_side,
                amount=amount,
                params={
                    "stopPrice": stop_price,
                    "reduceOnly": True,
                    "workingType": "MARK_PRICE",
                    "priceProtect": True,
                  },
            )
            logger.info(
                "trigger_order_placed", pair=pair, type=order_type,
                trigger_price=stop_price, order_id=order.get("id"),
            )
            return {"order_id": order.get("id"), "status": order.get("status")}
        except Exception as e:
            logger.error("trigger_order_failed", pair=pair, type=order_type,
                         error=str(e))
            return None

    def cancel_all_orders(self, pair: str) -> bool:
        """Cancel all open orders for a pair (used after manual/AI closes)."""
        try:
            self.exchange.cancel_all_orders(pair)
            return True
        except Exception as e:
            logger.warning("cancel_all_orders_failed", pair=pair, error=str(e))
            return False

    def get_funding_rate(self, pair: str) -> float:
        """Get current funding rate."""
        try:
            info = self.exchange.fetch_funding_rate(pair)
            return float(info.get("fundingRate", 0))
        except Exception:
            return 0.0

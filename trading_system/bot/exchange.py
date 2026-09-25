"""
Exchange interface using ccxt.

Wraps ccxt for unified access to Binance futures.
Handles order placement, position queries, and account info.
"""

from __future__ import annotations

import time

import ccxt
import pandas as pd
import structlog

from trading_system.bot.venue_limits import DEFAULT_TAKER_FEE_PCT, MarketLimits
from trading_system.config import ExchangeConfig

logger = structlog.get_logger(__name__)


class ExchangeInterface:
    """Unified exchange interface using ccxt.

    Market DATA falls back to Kraken spot when Binance futures is
    unreachable: GitHub-hosted (US) runners get HTTP 451 geo-blocked by
    Binance, while the main bot has run reliably from them via Kraken.
    Orders and balances always use Binance — live mode only ever runs
    where Binance is reachable.
    """

    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.exchange = ccxt.binanceusdm({
            "apiKey": config.api_key or None,
            "secret": config.api_secret or None,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        # Spot fallback for market data only (never orders).
        self._spot = ccxt.kraken({"enableRateLimit": True})
        if config.sandbox:
            self.exchange.set_sandbox_mode(True)

        self._connected = False

    @staticmethod
    def _spot_symbol(pair: str) -> str | None:
        """BTC/USDT:USDT (perp) -> BTC/USDT (spot); None for non-USDT perps."""
        if pair.endswith(":USDT") and "/USDT:" in pair:
            return pair.split(":")[0]
        return None

    def _spot_ohlcv(
        self, pair: str, timeframe: str, limit: int
    ) -> pd.DataFrame:
        """Fetch OHLCV from the Kraken spot fallback (empty df on failure)."""
        spot_symbol = self._spot_symbol(pair)
        if not spot_symbol:
            return pd.DataFrame()
        try:
            candles = self._spot.fetch_ohlcv(spot_symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            logger.info(
                "ohlcv_spot_fallback", pair=pair, spot_symbol=spot_symbol, rows=len(df)
            )
            return df.set_index("timestamp")
        except Exception as e:
            logger.warning(
                "ohlcv_spot_fallback_failed", pair=pair, spot_symbol=spot_symbol,
                error=str(e)[:150],
            )
            return pd.DataFrame()

    def connect(self) -> bool:
        """Initialize exchange connection (3 attempts, escalating backoff).

        GitHub-hosted runners occasionally hit transient failures/timeouts
        against Binance; a single attempt turned those into instant dead
        runs with no audit trail.
        """
        last_error = ""
        for attempt in (1, 2, 3):
            try:
                self.exchange.load_markets()
                self._connected = True
                logger.info(
                    "exchange_connected", exchange=self.config.name, attempt=attempt
                )
                return True
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "exchange_connect_retry", attempt=attempt, error=last_error[:200]
                )
                if attempt < 3:
                    time.sleep(3 * attempt)
        logger.error("exchange_connection_failed", error=last_error)
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
        """Get current ticker (Binance futures, Kraken spot fallback)."""
        try:
            ticker = self.exchange.fetch_ticker(pair)
            result = {
                "bid": self._to_float(ticker.get("bid")),
                "ask": self._to_float(ticker.get("ask")),
                "last": self._to_float(ticker.get("last")),
                "volume": self._to_float(ticker.get("quoteVolume")),
            }
            if result["last"] > 0:
                return result
        except Exception as e:
            logger.warning("ticker_fetch_failed_trying_spot", pair=pair, error=str(e)[:150])

        # Spot fallback: prices are near-identical to the perp for BTC/ETH.
        spot_symbol = self._spot_symbol(pair)
        if spot_symbol:
            try:
                ticker = self._spot.fetch_ticker(spot_symbol)
                logger.info("ticker_spot_fallback", pair=pair, spot_symbol=spot_symbol)
                return {
                    "bid": self._to_float(ticker.get("bid")),
                    "ask": self._to_float(ticker.get("ask")),
                    "last": self._to_float(ticker.get("last")),
                    "volume": self._to_float(ticker.get("quoteVolume")),
                }
            except Exception as e:
                logger.warning(
                    "ticker_spot_fallback_failed", pair=pair, error=str(e)[:150]
                )
        return {"bid": 0, "ask": 0, "last": 0, "volume": 0}

    def get_ohlcv(
        self,
        pair: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch recent OHLCV data (Binance futures, Kraken spot fallback)."""
        try:
            candles = self.exchange.fetch_ohlcv(pair, timeframe, limit=limit)
            df = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")
            return df
        except Exception as e:
            logger.warning("ohlcv_fetch_failed_trying_spot", pair=pair, error=str(e)[:150])
        return self._spot_ohlcv(pair, timeframe, limit)

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

    def get_market_limits(self, pair: str) -> MarketLimits | None:
        """Venue order bounds (MIN_NOTIONAL / LOT_SIZE) straight from ccxt.

        Returns None when the market isn't loaded — which is the normal case
        on a geo-blocked runner — so callers must decide explicitly whether to
        fall back to the builtin table rather than treating "unknown" as
        "unlimited".
        """
        try:
            market = (self.exchange.markets or {}).get(pair)
            if not market:
                return None
            limits = market.get("limits") or {}
            min_notional = (limits.get("cost") or {}).get("min")
            min_amount = (limits.get("amount") or {}).get("min")
            if min_notional is None or min_amount is None:
                return None
            step = (market.get("precision") or {}).get("amount") or min_amount
            taker = market.get("taker")
            return MarketLimits(
                pair=pair,
                min_notional=float(min_notional),
                amount_step=float(step),
                min_amount=float(min_amount),
                taker_fee_pct=(
                    float(taker) * 100 if taker else DEFAULT_TAKER_FEE_PCT
                ),
                source="exchange",
            )
        except Exception as e:
            logger.warning("market_limits_failed", pair=pair, error=str(e)[:150])
            return None

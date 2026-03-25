"""
Async WebSocket collector for Binance BTCUSDT market data.

Connects to combined streams for trades, depth updates, and kline candles.
Maintains a local order book and routes events to registered callbacks.
"""

import asyncio
import json
import logging
import warnings
from typing import Callable, Optional

import websockets

warnings.warn(
    "binance_ws.py is deprecated — Binance is geo-blocked in the US. "
    "Use coinbase_ws.py or coinbase_auth_ws.py instead.",
    DeprecationWarning, stacklevel=2,
)

from config.settings import settings

logger = logging.getLogger(__name__)


class BinanceCollector:
    """Real-time Binance market data collector via WebSocket."""

    def __init__(
        self,
        symbol: str = "btcusdt",
        on_trade: Optional[Callable] = None,
        on_depth: Optional[Callable] = None,
        on_kline: Optional[Callable] = None,
    ):
        self.symbol = symbol.lower()
        self.on_trade = on_trade
        self.on_depth = on_depth
        self.on_kline = on_kline

        # Local order book: {"bids": {price_str: qty_str}, "asks": {price_str: qty_str}}
        self.local_orderbook: dict = {"bids": {}, "asks": {}}
        self.last_update_id: int = 0

        self._running: bool = False
        self._ws = None
        self._reconnect_delay: float = 1.0
        self._max_reconnect_delay: float = 30.0

    def _build_url(self) -> str:
        streams = f"{self.symbol}@trade/{self.symbol}@depth@100ms/{self.symbol}@kline_1m"
        return f"{settings.BINANCE_WS_URL}/stream?streams={streams}"

    async def start(self) -> None:
        """Connect to Binance combined stream and begin processing messages."""
        self._running = True
        self._reconnect_delay = 1.0
        url = self._build_url()

        while self._running:
            try:
                logger.info("Connecting to Binance WebSocket: %s", url)
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    self._ws = ws
                    self._reconnect_delay = 1.0  # Reset on successful connect
                    logger.info("Connected to Binance WebSocket")

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw_msg)
                            await self._route_message(msg)
                        except json.JSONDecodeError:
                            logger.warning("Received non-JSON message, skipping")
                        except Exception:
                            logger.exception("Error processing message")

            except websockets.ConnectionClosed as e:
                if not self._running:
                    break
                logger.warning("WebSocket connection closed: %s. Reconnecting in %.1fs", e, self._reconnect_delay)
            except Exception:
                if not self._running:
                    break
                logger.exception("WebSocket error. Reconnecting in %.1fs", self._reconnect_delay)

            if self._running:
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

        logger.info("Binance WebSocket collector stopped")

    async def _route_message(self, msg: dict) -> None:
        """Route a combined-stream message to the appropriate handler."""
        stream = msg.get("stream", "")
        data = msg.get("data", {})

        if not stream or not data:
            return

        if stream.endswith("@trade"):
            await self._handle_trade(data)
        elif stream.startswith(f"{self.symbol}@depth"):
            await self._handle_depth(data)
        elif stream.startswith(f"{self.symbol}@kline"):
            await self._handle_kline(data)

    async def _handle_trade(self, data: dict) -> None:
        """Parse a trade event and invoke the on_trade callback."""
        trade = {
            "price": float(data["p"]),
            "qty": float(data["q"]),
            "is_buyer_maker": data["m"],
            "timestamp_ms": data["T"],
        }

        if self.on_trade is not None:
            await self.on_trade(trade)

    async def _handle_depth(self, data: dict) -> None:
        """Update local order book from a depth update and invoke the on_depth callback."""
        # Process bid updates
        for bid in data.get("b", []):
            price, qty = bid[0], bid[1]
            if qty == "0" or float(qty) == 0:
                self.local_orderbook["bids"].pop(price, None)
            else:
                self.local_orderbook["bids"][price] = qty

        # Process ask updates
        for ask in data.get("a", []):
            price, qty = ask[0], ask[1]
            if qty == "0" or float(qty) == 0:
                self.local_orderbook["asks"].pop(price, None)
            else:
                self.local_orderbook["asks"][price] = qty

        self.last_update_id = data.get("u", self.last_update_id)

        if self.on_depth is not None:
            snapshot = self.get_orderbook_snapshot()
            await self.on_depth(snapshot)

    async def _handle_kline(self, data: dict) -> None:
        """Parse a kline event; only invoke on_kline when the candle is closed."""
        k = data.get("k", {})

        kline = {
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "is_closed": k["x"],
            "timestamp_ms": k["t"],
            "close_time_ms": k["T"],
        }

        if kline["is_closed"] and self.on_kline is not None:
            await self.on_kline(kline)

    async def stop(self) -> None:
        """Stop the collector and close the WebSocket connection."""
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                logger.debug("Error closing WebSocket, ignoring")
            self._ws = None
        logger.info("Binance collector stop requested")

    def get_orderbook_snapshot(self) -> dict:
        """
        Return the current top 20 bid and ask levels.

        Bids sorted descending by price, asks sorted ascending by price.
        Each level is a dict with 'price' (float) and 'qty' (float).
        """
        bids_sorted = sorted(
            self.local_orderbook["bids"].items(),
            key=lambda x: float(x[0]),
            reverse=True,
        )[:20]

        asks_sorted = sorted(
            self.local_orderbook["asks"].items(),
            key=lambda x: float(x[0]),
        )[:20]

        return {
            "bids": [{"price": float(p), "qty": float(q)} for p, q in bids_sorted],
            "asks": [{"price": float(p), "qty": float(q)} for p, q in asks_sorted],
        }

"""
Async WebSocket collector for Coinbase BTC-USD market data.

Uses Coinbase Advanced Trade WebSocket for:
- ticker (trades/price updates)
- level2 (order book)
- candles (1m klines)
"""

import asyncio
import json
import logging
import time
from typing import Callable, Optional

import websockets

logger = logging.getLogger(__name__)

COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"


class CoinbaseCollector:
    """Real-time Coinbase market data collector via WebSocket."""

    def __init__(
        self,
        symbol: str = "BTC-USD",
        on_trade: Optional[Callable] = None,
        on_depth: Optional[Callable] = None,
        on_kline: Optional[Callable] = None,
    ):
        self.symbol = symbol
        self.on_trade = on_trade
        self.on_depth = on_depth
        self.on_kline = on_kline

        # Local order book: price (float) -> qty (float)
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}

        self._running: bool = False
        self._ws = None
        self._reconnect_delay: float = 1.0

        # Kline building state
        self._current_kline: Optional[dict] = None
        self._kline_minute: int = -1

    async def start(self) -> None:
        """Connect to Coinbase WebSocket and process messages."""
        self._running = True
        self._reconnect_delay = 1.0

        while self._running:
            try:
                logger.info("Connecting to Coinbase WebSocket...")
                async with websockets.connect(
                    COINBASE_WS_URL, ping_interval=20, ping_timeout=10
                ) as ws:
                    self._ws = ws
                    self._reconnect_delay = 1.0
                    logger.info("Connected to Coinbase WebSocket")

                    # Subscribe to channels
                    await self._subscribe(ws)

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw_msg)
                            await self._route_message(msg)
                        except json.JSONDecodeError:
                            pass
                        except Exception:
                            logger.exception("Error processing message")

            except websockets.ConnectionClosed:
                if not self._running:
                    break
                logger.warning("Connection closed. Reconnecting in %.1fs", self._reconnect_delay)
            except Exception:
                if not self._running:
                    break
                logger.exception("WS error. Reconnecting in %.1fs", self._reconnect_delay)

            if self._running:
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)

    async def _subscribe(self, ws) -> None:
        """Subscribe to market_trades channel."""
        sub_msg = {
            "type": "subscribe",
            "product_ids": [self.symbol],
            "channel": "market_trades",
        }
        await ws.send(json.dumps(sub_msg))
        logger.info("Subscribed to market_trades for %s", self.symbol)

        # Start orderbook polling in background
        asyncio.create_task(self._poll_orderbook())

    async def _poll_orderbook(self) -> None:
        """Poll Coinbase REST API for orderbook every 3 seconds."""
        import httpx

        async with httpx.AsyncClient(timeout=10) as client:
            while self._running:
                try:
                    resp = await client.get(
                        f"https://api.exchange.coinbase.com/products/{self.symbol}/book",
                        params={"level": 2},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        self._bids.clear()
                        self._asks.clear()
                        for bid in data.get("bids", []):
                            self._bids[float(bid[0])] = float(bid[1])
                        for ask in data.get("asks", []):
                            self._asks[float(ask[0])] = float(ask[1])

                        if self.on_depth is not None:
                            snapshot = self.get_orderbook_snapshot()
                            await self.on_depth(snapshot)
                except Exception as e:
                    logger.debug("Orderbook poll error: %s", e)

                await asyncio.sleep(3)

    async def _route_message(self, msg: dict) -> None:
        """Route message to appropriate handler."""
        channel = msg.get("channel", "")
        events = msg.get("events", [])

        if channel == "market_trades":
            for event in events:
                await self._handle_trades(event)

    async def _handle_trades(self, event: dict) -> None:
        """Handle market_trades events."""
        trades = event.get("trades", [])
        for t in trades:
            trade = {
                "price": float(t["price"]),
                "qty": float(t["size"]),
                "is_buyer_maker": t["side"] == "SELL",  # SELL side = buyer is maker
                "timestamp_ms": int(
                    _parse_coinbase_ts(t.get("time", "")) * 1000
                ),
            }

            if self.on_trade is not None:
                await self.on_trade(trade)

            # Build 1m klines from trades
            await self._update_kline(trade)

    async def _update_kline(self, trade: dict) -> None:
        """Build 1m candles from trades."""
        ts_s = trade["timestamp_ms"] / 1000
        minute = int(ts_s // 60)
        price = trade["price"]
        qty = trade["qty"]

        if minute != self._kline_minute:
            # Emit previous candle if exists
            if self._current_kline is not None and self.on_kline is not None:
                self._current_kline["is_closed"] = True
                await self.on_kline(self._current_kline)

            # Start new candle
            self._kline_minute = minute
            self._current_kline = {
                "timestamp_ms": minute * 60 * 1000,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": qty,
                "is_closed": False,
            }
        else:
            if self._current_kline is not None:
                self._current_kline["high"] = max(self._current_kline["high"], price)
                self._current_kline["low"] = min(self._current_kline["low"], price)
                self._current_kline["close"] = price
                self._current_kline["volume"] += qty

    async def stop(self) -> None:
        """Stop the collector."""
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("Coinbase collector stopped")

    def get_orderbook_snapshot(self) -> dict:
        """Return top 20 bid/ask levels as [[price, qty], ...] lists."""
        bids = sorted(self._bids.items(), key=lambda x: x[0], reverse=True)[:20]
        asks = sorted(self._asks.items(), key=lambda x: x[0])[:20]

        return {
            "bids": [[p, q] for p, q in bids],
            "asks": [[p, q] for p, q in asks],
        }


def _parse_coinbase_ts(ts_str: str) -> float:
    """Parse Coinbase ISO timestamp to epoch seconds."""
    if not ts_str:
        return time.time()
    try:
        from datetime import datetime, timezone
        # Handle "2026-03-25T03:48:00.000000Z" format
        ts_str = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_str)
        return dt.timestamp()
    except Exception:
        return time.time()

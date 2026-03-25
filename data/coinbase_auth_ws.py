"""
Authenticated Coinbase WebSocket client using CDP API key.
Provides real-time ticker (L1 best bid/ask) and trade streams.
"""

import asyncio
import base64
import json
import logging
import secrets
import time
from typing import Callable, Optional

import jwt
import websockets
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

logger = logging.getLogger(__name__)

COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"


class CoinbaseAuthCollector:
    """Real-time Coinbase data via authenticated WebSocket (CDP key).

    Provides sub-second ticker updates with best_bid/best_ask (L1),
    plus market_trades stream.
    """

    def __init__(
        self,
        key_id: str,
        private_key_b64: str,
        symbol: str = "BTC-USD",
        on_trade: Optional[Callable] = None,
        on_ticker: Optional[Callable] = None,
        on_kline: Optional[Callable] = None,
    ):
        self.key_id = key_id
        self.symbol = symbol
        self.on_trade = on_trade
        self.on_ticker = on_ticker
        self.on_kline = on_kline

        # Load Ed25519 key
        key_bytes = base64.b64decode(private_key_b64)
        priv = Ed25519PrivateKey.from_private_bytes(key_bytes[:32])
        self._pem = priv.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )

        self._running = False
        self._ws = None
        self._reconnect_delay = 1.0

        # Kline building
        self._kline_minute = -1
        self._current_kline: Optional[dict] = None

        # Latest state
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.last_price: float = 0.0
        self.spread: float = 0.0
        self.volume_24h: float = 0.0
        self.ticker_count: int = 0
        self.trade_count: int = 0

    def _make_jwt(self) -> str:
        payload = {
            "sub": self.key_id,
            "iss": "coinbase-cloud",
            "nbf": int(time.time()),
            "exp": int(time.time()) + 120,
        }
        return jwt.encode(
            payload, self._pem, algorithm="EdDSA",
            headers={"kid": self.key_id, "typ": "JWT", "nonce": secrets.token_hex(16)},
        )

    async def start(self) -> None:
        self._running = True
        self._reconnect_delay = 1.0

        while self._running:
            try:
                logger.info("Connecting to Coinbase authenticated WebSocket...")
                async with websockets.connect(COINBASE_WS_URL, ping_interval=20) as ws:
                    self._ws = ws
                    self._reconnect_delay = 1.0
                    await self._subscribe(ws)
                    logger.info("Connected + subscribed to ticker + market_trades")

                    # Re-auth every 90 seconds (JWT expires in 120s)
                    last_auth = time.time()

                    async for raw_msg in ws:
                        if not self._running:
                            break

                        # Re-authenticate periodically
                        if time.time() - last_auth > 90:
                            await self._subscribe(ws)
                            last_auth = time.time()

                        try:
                            msg = json.loads(raw_msg)
                            await self._route(msg)
                        except json.JSONDecodeError:
                            pass

            except websockets.ConnectionClosed:
                if not self._running:
                    break
                logger.warning("Connection closed. Reconnecting in %.0fs", self._reconnect_delay)
            except Exception:
                if not self._running:
                    break
                logger.exception("WS error. Reconnecting in %.0fs", self._reconnect_delay)

            if self._running:
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)

    async def _subscribe(self, ws) -> None:
        token = self._make_jwt()
        for channel in ["ticker", "market_trades"]:
            await ws.send(json.dumps({
                "type": "subscribe",
                "product_ids": [self.symbol],
                "channel": channel,
                "jwt": token,
            }))

    async def _route(self, msg: dict) -> None:
        channel = msg.get("channel", "")
        events = msg.get("events", [])

        if channel == "ticker":
            for ev in events:
                await self._handle_ticker(ev)
        elif channel == "market_trades":
            for ev in events:
                await self._handle_trades(ev)

    async def _handle_ticker(self, event: dict) -> None:
        for t in event.get("tickers", []):
            self.best_bid = float(t.get("best_bid", 0))
            self.best_ask = float(t.get("best_ask", 0))
            self.last_price = float(t.get("price", 0))
            self.spread = self.best_ask - self.best_bid
            self.volume_24h = float(t.get("volume_24_h", 0))
            self.ticker_count += 1

            if self.on_ticker is not None:
                await self.on_ticker({
                    "price": self.last_price,
                    "best_bid": self.best_bid,
                    "best_ask": self.best_ask,
                    "spread": self.spread,
                    "volume_24h": self.volume_24h,
                })

    async def _handle_trades(self, event: dict) -> None:
        for t in event.get("trades", []):
            trade = {
                "price": float(t["price"]),
                "qty": float(t["size"]),
                "is_buyer_maker": t["side"] == "SELL",
                "timestamp_ms": int(_parse_ts(t.get("time", "")) * 1000),
            }
            self.trade_count += 1

            if self.on_trade is not None:
                await self.on_trade(trade)

            await self._update_kline(trade)

    async def _update_kline(self, trade: dict) -> None:
        minute = int(trade["timestamp_ms"] / 1000 // 60)
        price = trade["price"]
        qty = trade["qty"]

        if minute != self._kline_minute:
            if self._current_kline is not None and self.on_kline is not None:
                self._current_kline["is_closed"] = True
                await self.on_kline(self._current_kline)

            self._kline_minute = minute
            self._current_kline = {
                "timestamp_ms": minute * 60 * 1000,
                "open": price, "high": price, "low": price, "close": price,
                "volume": qty, "is_closed": False,
            }
        elif self._current_kline is not None:
            self._current_kline["high"] = max(self._current_kline["high"], price)
            self._current_kline["low"] = min(self._current_kline["low"], price)
            self._current_kline["close"] = price
            self._current_kline["volume"] += qty

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None


def _parse_ts(ts_str: str) -> float:
    if not ts_str:
        return time.time()
    try:
        from datetime import datetime
        ts_str = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_str).timestamp()
    except Exception:
        return time.time()

"""Authenticated Kalshi WebSocket client for user fills and order updates."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

import websockets
from websockets.exceptions import ConnectionClosed

from config.settings import settings
from kalshi.auth import KalshiAuth

logger = logging.getLogger(__name__)

WS_PATH = "/trade-api/ws/v2"
PROD_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
DEMO_WS_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"


def derive_ws_url(rest_base_url: str) -> str:
    """Infer the matching WebSocket URL from the configured REST base URL."""
    if "demo-api.kalshi.co" in rest_base_url:
        return DEMO_WS_URL
    return PROD_WS_URL


def dollars_to_cents(value: str | int | float | Decimal | None) -> int:
    """Convert a dollar-denominated Kalshi string value to integer cents."""
    if value in (None, ""):
        return 0
    try:
        return int((Decimal(str(value)) * Decimal("100")).quantize(Decimal("1")))
    except (InvalidOperation, ValueError, TypeError):
        return 0


def fp_to_int(value: str | int | float | Decimal | None) -> int:
    """Convert Kalshi fixed-point numeric strings like '1.00' to ints."""
    if value in (None, ""):
        return 0
    try:
        return int(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError):
        return 0


def _coerce_cents(value: str | int | float | Decimal | None) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError):
        return 0


def _extract_price_cents(message: dict[str, Any], side: str = "") -> int:
    normalized_side = (side or "").lower()

    if normalized_side == "yes":
        if message.get("yes_price") not in (None, ""):
            return _coerce_cents(message["yes_price"])
        return dollars_to_cents(message.get("yes_price_dollars"))

    if normalized_side == "no":
        if message.get("no_price") not in (None, ""):
            return _coerce_cents(message["no_price"])
        return dollars_to_cents(message.get("no_price_dollars"))

    for cents_key in ("yes_price", "no_price"):
        if message.get(cents_key) not in (None, ""):
            return _coerce_cents(message[cents_key])

    for dollars_key in ("yes_price_dollars", "no_price_dollars"):
        cents = dollars_to_cents(message.get(dollars_key))
        if cents:
            return cents

    return 0


class KalshiWebSocketClient:
    """Authenticated Kalshi WebSocket client with reconnect + callbacks."""

    def __init__(
        self,
        auth: KalshiAuth | None = None,
        *,
        ws_url: str | None = None,
        channels: tuple[str, ...] = ("fill", "user_orders"),
        market_tickers: list[str] | None = None,
        on_fill: Callable[[dict[str, Any]], Any] | None = None,
        on_user_order: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.auth = auth or KalshiAuth(
            settings.KALSHI_API_KEY_ID,
            settings.KALSHI_PRIVATE_KEY_PATH,
        )
        self.ws_url = ws_url or derive_ws_url(settings.KALSHI_BASE_URL)
        self.channels = channels
        self.market_tickers = market_tickers or []
        self.on_fill = on_fill
        self.on_user_order = on_user_order

        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ws = None
        self._connected = threading.Event()
        self._message_id = 1
        self._reconnect_delay = 1.0
        self._subscriptions: dict[str, int] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def wait_until_connected(self, timeout_s: float = 10.0) -> bool:
        return self._connected.wait(timeout_s)

    def start_in_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="kalshi-ws",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._running = False
        self._connected.clear()

        loop = self._loop
        ws = self._ws
        if loop is not None and loop.is_running() and ws is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(ws.close(), loop)
                future.result(timeout=timeout_s)
            except Exception:
                logger.debug("Kalshi WS close during shutdown failed", exc_info=True)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout_s)

    async def run(self) -> None:
        self._reconnect_delay = 1.0

        while self._running:
            try:
                headers = self.auth.sign_request("GET", WS_PATH)
                logger.info("Connecting to Kalshi WebSocket %s", self.ws_url)

                async with websockets.connect(
                    self.ws_url,
                    additional_headers=headers,
                    open_timeout=30,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    self._connected.set()
                    self._subscriptions.clear()
                    self._reconnect_delay = 1.0
                    logger.info("Connected to Kalshi WebSocket")

                    await self._subscribe(ws)

                    async for raw_message in ws:
                        if not self._running:
                            break
                        await self._handle_raw_message(raw_message)

            except ConnectionClosed:
                if not self._running:
                    break
                logger.warning(
                    "Kalshi WebSocket closed. Reconnecting in %.1fs",
                    self._reconnect_delay,
                )
            except Exception:
                if not self._running:
                    break
                logger.exception(
                    "Kalshi WebSocket error. Reconnecting in %.1fs",
                    self._reconnect_delay,
                )
            finally:
                self._connected.clear()
                self._ws = None

            if self._running:
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)

    async def _subscribe(self, ws) -> None:
        for channel in self.channels:
            params: dict[str, Any] = {"channels": [channel]}
            if self.market_tickers:
                params["market_tickers"] = self.market_tickers

            message = {
                "id": self._message_id,
                "cmd": "subscribe",
                "params": params,
            }
            self._message_id += 1

            await ws.send(json.dumps(message))
            logger.info("Kalshi WS subscribe sent: %s", channel)

    async def _handle_raw_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.debug("Ignoring non-JSON Kalshi WS message: %r", raw_message)
            return

        try:
            await self._route_message(message)
        except Exception:
            logger.exception("Kalshi WS message handling failed")

    async def _route_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type", "")

        if message_type == "fill":
            await self._invoke_callback(self.on_fill, self.normalize_fill_message(message))
            return

        if message_type == "user_order":
            await self._invoke_callback(
                self.on_user_order,
                self.normalize_user_order_message(message),
            )
            return

        if message_type == "subscribed":
            msg = message.get("msg", {})
            channel = msg.get("channel", "")
            sid = msg.get("sid", message.get("sid"))
            if channel and sid is not None:
                self._subscriptions[channel] = int(sid)
            logger.info("Kalshi WS subscribed: channel=%s sid=%s", channel, sid)
            return

        if message_type == "error":
            err = message.get("msg", {})
            logger.error(
                "Kalshi WS error id=%s code=%s msg=%s",
                message.get("id"),
                err.get("code"),
                err.get("msg"),
            )
            return

        logger.debug("Unhandled Kalshi WS message type=%s", message_type)

    async def _invoke_callback(
        self,
        callback: Callable[[dict[str, Any]], Any] | None,
        payload: dict[str, Any],
    ) -> None:
        if callback is None:
            return

        result = callback(payload)
        if inspect.isawaitable(result):
            await result

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self.run())
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()
            self._loop = None

    @staticmethod
    def normalize_fill_message(message: dict[str, Any]) -> dict[str, Any]:
        msg = message.get("msg", message)
        side = (msg.get("side") or msg.get("purchased_side") or "").lower()
        return {
            "order_id": msg.get("order_id", ""),
            "ticker": msg.get("market_ticker") or msg.get("ticker", ""),
            "trade_id": msg.get("trade_id", ""),
            "side": side,
            "action": (msg.get("action") or "").lower(),
            "count": max(fp_to_int(msg.get("count_fp", msg.get("count"))), 0),
            "price_cents": _extract_price_cents(msg, side),
            "is_taker": bool(msg.get("is_taker", False)),
            "ts": msg.get("ts"),
            "raw": msg,
        }

    @staticmethod
    def normalize_user_order_message(message: dict[str, Any]) -> dict[str, Any]:
        msg = message.get("msg", message)
        side = (msg.get("side") or "").lower()
        if not side and "is_yes" in msg:
            side = "yes" if bool(msg.get("is_yes")) else "no"

        fill_count = fp_to_int(msg.get("fill_count_fp", msg.get("fill_count")))
        status = str(msg.get("status", "")).lower()
        if fill_count == 0 and status == "executed":
            fill_count = 1

        return {
            "order_id": msg.get("order_id", ""),
            "ticker": msg.get("market_ticker") or msg.get("ticker", ""),
            "status": status,
            "side": side,
            "price_cents": _extract_price_cents(msg, side),
            "fill_count": fill_count,
            "remaining_count": fp_to_int(
                msg.get("remaining_count_fp", msg.get("remaining_count"))
            ),
            "initial_count": fp_to_int(
                msg.get("initial_count_fp", msg.get("initial_count", msg.get("count")))
            ),
            "client_order_id": msg.get("client_order_id", ""),
            "created_time": msg.get("created_time", ""),
            "expiration_time": msg.get("expiration_time", ""),
            "raw": msg,
        }

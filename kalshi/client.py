from __future__ import annotations

import logging
import time

import httpx

from config.settings import settings
from .auth import KalshiAuth

logger = logging.getLogger(__name__)

# Retryable HTTP status codes (GET only — never retry POST/order placement)
_RETRYABLE_STATUS = {429, 500, 502, 503}
_MAX_RETRIES = 3
_BACKOFF_BASE_S = 1.0

# Simple token bucket rate limiter
_RATE_LIMIT_RPS = 10
_MIN_INTERVAL_S = 1.0 / _RATE_LIMIT_RPS


class KalshiClient:
    """Full Kalshi REST client with RSA-PSS authentication.

    Features:
    - Rate limiting: 10 req/s token bucket
    - Retry with exponential backoff on GET 429/5xx (never retries POST)
    - Methods return raw dicts (callers can wrap in typed models)
    """

    API_PREFIX = "/trade-api/v2"

    def __init__(self) -> None:
        self.base_url = settings.KALSHI_BASE_URL
        self.auth = KalshiAuth(settings.KALSHI_API_KEY_ID, settings.KALSHI_PRIVATE_KEY_PATH)
        self.http = httpx.Client(timeout=30.0)
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Simple token bucket: enforce minimum interval between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < _MIN_INTERVAL_S:
            time.sleep(_MIN_INTERVAL_S - elapsed)
        self._last_request_time = time.monotonic()

    def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> dict:
        """Make an authenticated request to the Kalshi API.

        GET requests retry on 429/5xx with exponential backoff.
        POST/DELETE requests are NOT retried (no idempotency key).
        """
        full_path = f"{self.API_PREFIX}{path}"
        url = f"{self.base_url}{path}"
        method_upper = method.upper()
        can_retry = method_upper == "GET"
        max_attempts = _MAX_RETRIES if can_retry else 1

        for attempt in range(max_attempts):
            self._rate_limit()
            headers = self.auth.sign_request(method_upper, full_path)

            try:
                response = self.http.request(
                    method=method_upper,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_body,
                )
            except httpx.TransportError as exc:
                if attempt < max_attempts - 1 and can_retry:
                    wait = _BACKOFF_BASE_S * (2 ** attempt)
                    logger.warning("Transport error on %s %s (attempt %d), retrying in %.1fs: %s",
                                   method_upper, path, attempt + 1, wait, exc)
                    time.sleep(wait)
                    continue
                raise

            if response.status_code in _RETRYABLE_STATUS and attempt < max_attempts - 1 and can_retry:
                wait = _BACKOFF_BASE_S * (2 ** attempt)
                logger.warning("HTTP %d on %s %s (attempt %d), retrying in %.1fs",
                               response.status_code, method_upper, path, attempt + 1, wait)
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        # Should not reach here, but just in case
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def get_balance(self) -> float:
        """Get account balance in dollars."""
        data = self._request("GET", "/portfolio/balance")
        return data.get("balance", 0) / 100.0

    def get_positions(self, event_ticker: str | None = None) -> list:
        """Get current portfolio positions."""
        params = {}
        if event_ticker:
            params["event_ticker"] = event_ticker
        data = self._request("GET", "/portfolio/positions", params=params or None)
        return data.get("market_positions", [])

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(
        self,
        ticker: str,
        side: str,
        action: str,
        count: int,
        order_type: str = "market",
        yes_price: int | None = None,
        no_price: int | None = None,
    ) -> dict:
        """Place an order on a market.

        Args:
            ticker: Market ticker.
            side: "yes" or "no".
            action: "buy" or "sell".
            count: Number of contracts.
            order_type: "market" or "limit".
            yes_price: Optional yes price in cents (for limit orders).
            no_price: Optional no price in cents (for limit orders).

        Returns:
            Order response dict.
        """
        body: dict = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": order_type,
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price

        return self._request("POST", "/portfolio/orders", json_body=body)

    def get_orders(self, ticker: str | None = None, status: str | None = None) -> list:
        """Get orders, optionally filtered by ticker and/or status."""
        params = {}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        data = self._request("GET", "/portfolio/orders", params=params or None)
        return data.get("orders", [])

    def get_order(self, order_id: str) -> dict:
        """Get a single order by ID."""
        data = self._request("GET", f"/portfolio/orders/{order_id}")
        return data.get("order", data)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order by ID."""
        return self._request("DELETE", f"/portfolio/orders/{order_id}")

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    def get_markets(
        self,
        event_ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict:
        """Get markets with optional filters.

        Args:
            event_ticker: Filter by event ticker.
            status: Filter by market status (e.g. "open").
            limit: Max number of results.
            cursor: Pagination cursor.

        Returns:
            Full response dict containing 'markets' list and 'cursor'.
        """
        params: dict = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/markets", params=params)

    def get_market(self, ticker: str) -> dict:
        """Get details for a single market."""
        data = self._request("GET", f"/markets/{ticker}")
        return data.get("market", data)

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """Get orderbook for a market.

        Args:
            ticker: Market ticker.
            depth: Number of levels to return.

        Returns:
            Orderbook dict with 'yes' and 'no' arrays.
        """
        return self._request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})

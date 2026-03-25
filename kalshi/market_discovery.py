from __future__ import annotations

from datetime import datetime, timezone

from .client import KalshiClient


class KXBTCMarketTracker:
    """Track and discover KXBTC 15-minute Bitcoin markets on Kalshi."""

    def __init__(self, client: KalshiClient) -> None:
        self.client = client

    def get_active_markets(self) -> list[dict]:
        """Fetch all currently open KXBTC15M markets.

        Returns:
            List of market dicts from the Kalshi API.
        """
        data = self.client._request(
            "GET", "/markets",
            params={"series_ticker": "KXBTC15M", "status": "open", "limit": 10},
        )
        return data.get("markets", [])

    def get_next_market(self) -> dict | None:
        """Find the market closing soonest among active KXBTC15M markets.

        Returns:
            The market dict with the earliest close_time, or None if no
            active markets exist.
        """
        markets = self.get_active_markets()
        if not markets:
            return None

        return min(markets, key=lambda m: self._parse_close_time(m))

    def get_market_time_remaining(self, market: dict) -> float:
        """Calculate seconds remaining until a market closes.

        Args:
            market: Market dict containing a 'close_time' field in ISO format.

        Returns:
            Seconds until the market's close_time. May be negative if
            the market has already closed.
        """
        close_dt = self._parse_close_time(market)
        now = datetime.now(timezone.utc)
        return (close_dt - now).total_seconds()

    @staticmethod
    def _parse_close_time(market: dict) -> datetime:
        """Parse a market's close_time string to a timezone-aware datetime.

        Handles both 'Z' suffix and '+00:00' offset formats.
        """
        raw = market["close_time"]
        # Handle trailing 'Z' that fromisoformat doesn't accept in older Python
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)

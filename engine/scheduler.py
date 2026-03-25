"""Market scheduling for the Kalshi KXBTC trading loop.

Decides which market to target and whether the timing is right to trade.
"""

import logging
from datetime import datetime, timezone

from kalshi.market_discovery import KXBTCMarketTracker

logger = logging.getLogger(__name__)


class MarketScheduler:
    """Select the next KXBTC market and decide trade timing based on phase."""

    # Phase boundaries (seconds until market close)
    PHASE_COLLECTING = 120   # > 120s: still gathering data
    PHASE_DECIDING = 60      # 60-120s: run inference, place order
    PHASE_MONITORING = 30    # 30-60s: watch fill status
    # < 30s: hands off

    def __init__(self, tracker: KXBTCMarketTracker):
        self.tracker = tracker
        self.current_market: dict | None = None
        self.last_traded_ticker: str = ""

    async def update(self) -> dict | None:
        """Refresh active markets and select the next one to trade.

        Returns:
            The current target market dict, or None.
        """
        try:
            market = self.tracker.get_next_market()
        except Exception:
            logger.exception("Failed to fetch active KXBTC markets")
            return self.current_market

        if market is None:
            self.current_market = None
            return None

        ticker = market.get("ticker", "")

        if ticker != (self.current_market or {}).get("ticker", ""):
            logger.info(
                "New target market: %s (close_time=%s)",
                ticker, market.get("close_time", "?"),
            )
            self.current_market = market

        return self.current_market

    def get_phase(self) -> str:
        """Determine the current trading phase based on time to close.

        Returns:
            One of 'collecting', 'deciding', 'monitoring', 'hands_off',
            or 'no_market' if there is no active market.
        """
        if self.current_market is None:
            return "no_market"

        remaining = self.tracker.get_market_time_remaining(self.current_market)

        if remaining > self.PHASE_COLLECTING:
            return "collecting"
        if remaining > self.PHASE_DECIDING:
            return "deciding"
        if remaining > self.PHASE_MONITORING:
            return "monitoring"
        return "hands_off"

    def should_trade(self) -> bool:
        """Return True if conditions are met to place a trade.

        Conditions:
          - Phase is 'deciding'
          - A market is selected
          - This market has not already been traded
        """
        if self.current_market is None:
            return False

        ticker = self.current_market.get("ticker", "")
        if ticker == self.last_traded_ticker:
            return False

        return self.get_phase() == "deciding"

    def mark_traded(self, ticker: str) -> None:
        """Record that a trade was placed for this ticker."""
        self.last_traded_ticker = ticker
        logger.info("Marked %s as traded", ticker)

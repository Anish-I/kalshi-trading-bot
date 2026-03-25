"""Position tracking and P&L calculation for Kalshi binary contracts."""

import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


class PositionManager:
    """Track open positions and compute realized P&L on Kalshi markets."""

    def __init__(self):
        # ticker -> {side, count, entry_price, timestamp}
        self.positions: dict[str, dict] = {}
        self.trade_log: list[dict] = []
        self.realized_pnl_cents: int = 0

    def open_position(
        self,
        ticker: str,
        side: str,
        count: int,
        entry_price_cents: int,
    ) -> None:
        """Record a new position.

        Args:
            ticker: Kalshi market ticker.
            side: "yes" or "no".
            count: Number of contracts.
            entry_price_cents: Price paid per contract in cents (1-99).
        """
        self.positions[ticker] = {
            "side": side,
            "count": count,
            "entry_price": entry_price_cents,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            "Position opened: %s %s x%d @ %dc",
            ticker, side, count, entry_price_cents,
        )

    def close_position(self, ticker: str, settled_yes: bool) -> int:
        """Close a position after market settlement.

        Kalshi settlement rules:
          - YES contract pays 100c if the answer is YES, 0 otherwise.
          - If you buy YES at Xc:
              YES settles YES -> P&L = (100 - X) * count
              YES settles NO  -> P&L = -X * count
          - If you buy NO at Xc (equivalent to selling YES at (100-X)c):
              NO settles NO  -> P&L = (100 - X) * count
              NO settles YES -> P&L = -X * count

        Args:
            ticker: Market ticker to close.
            settled_yes: True if the market settled YES.

        Returns:
            Realized P&L in cents.
        """
        pos = self.positions.pop(ticker, None)
        if pos is None:
            logger.warning("No open position for %s, nothing to close", ticker)
            return 0

        side = pos["side"]
        entry = pos["entry_price"]
        count = pos["count"]

        if side == "yes":
            pnl = ((100 - entry) * count) if settled_yes else (-entry * count)
        else:  # side == "no"
            pnl = ((100 - entry) * count) if not settled_yes else (-entry * count)

        self.realized_pnl_cents += pnl

        record = {
            "ticker": ticker,
            "side": side,
            "count": count,
            "entry_price": entry,
            "settled_yes": settled_yes,
            "pnl_cents": pnl,
            "opened_at": pos["timestamp"],
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.trade_log.append(record)

        logger.info(
            "Position closed: %s %s x%d entry=%dc settled_yes=%s pnl=%+dc  total=%+dc",
            ticker, side, count, entry, settled_yes, pnl, self.realized_pnl_cents,
        )
        return pnl

    def has_open_position(self) -> bool:
        """Return True if any position is open."""
        return len(self.positions) > 0

    def get_position(self, ticker: str | None = None) -> dict | None:
        """Return position info for a ticker, or the first open position."""
        if ticker:
            return self.positions.get(ticker)
        if self.positions:
            first_ticker = next(iter(self.positions))
            return {**self.positions[first_ticker], "ticker": first_ticker}
        return None

    def get_trade_log_df(self) -> pd.DataFrame:
        """Return the trade log as a DataFrame."""
        if not self.trade_log:
            return pd.DataFrame(columns=[
                "ticker", "side", "count", "entry_price",
                "settled_yes", "pnl_cents", "opened_at", "closed_at",
            ])
        return pd.DataFrame(self.trade_log)

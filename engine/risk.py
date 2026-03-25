"""Risk management for the Kalshi trading engine.

Enforces position limits, daily loss limits, and consecutive-loss halt.
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class RiskManager:
    """Gate-keeper that decides whether a new trade is allowed."""

    def __init__(
        self,
        max_contracts: int = 10,
        daily_loss_limit_cents: int = 5000,
        consecutive_loss_halt: int = 5,
    ):
        self.max_contracts = max_contracts
        self.daily_loss_limit = daily_loss_limit_cents
        self.consecutive_loss_halt = consecutive_loss_halt
        self.daily_pnl_cents: int = 0
        self.consecutive_losses: int = 0
        self.is_active: bool = True  # kill switch
        self.trades_today: list[dict] = []

    # ------------------------------------------------------------------
    # Pre-trade checks
    # ------------------------------------------------------------------

    def can_trade(self) -> tuple[bool, str]:
        """Check all risk conditions before entering a trade.

        Returns:
            (allowed, reason) tuple.
        """
        if not self.is_active:
            return False, "kill switch activated"

        if self.daily_pnl_cents <= -self.daily_loss_limit:
            return False, (
                f"daily loss limit hit: {self.daily_pnl_cents}c "
                f"<= -{self.daily_loss_limit}c"
            )

        if self.consecutive_losses >= self.consecutive_loss_halt:
            return False, (
                f"consecutive loss halt: {self.consecutive_losses} "
                f">= {self.consecutive_loss_halt}"
            )

        return True, "ok"

    def check_order_size(self, contracts: int) -> tuple[bool, str]:
        """Validate that the order size is within limits.

        Args:
            contracts: Number of contracts to buy.

        Returns:
            (allowed, reason) tuple.
        """
        if contracts <= self.max_contracts:
            return True, "ok"
        return False, (
            f"order size {contracts} exceeds max {self.max_contracts}"
        )

    # ------------------------------------------------------------------
    # Post-trade bookkeeping
    # ------------------------------------------------------------------

    def record_trade(self, pnl_cents: int, trade_info: dict) -> None:
        """Record a completed trade and update running totals.

        Args:
            pnl_cents: Realized P&L in cents (positive = profit).
            trade_info: Arbitrary dict with trade metadata.
        """
        self.daily_pnl_cents += pnl_cents

        if pnl_cents < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        record = {
            **trade_info,
            "pnl_cents": pnl_cents,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self.trades_today.append(record)

        logger.info(
            "Trade recorded: pnl=%+dc  daily=%+dc  consec_losses=%d",
            pnl_cents,
            self.daily_pnl_cents,
            self.consecutive_losses,
        )

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        """Reset daily counters (call at midnight UTC)."""
        logger.info(
            "Daily reset: pnl was %+dc across %d trades",
            self.daily_pnl_cents,
            len(self.trades_today),
        )
        self.daily_pnl_cents = 0
        self.consecutive_losses = 0
        self.trades_today = []

    def kill(self) -> None:
        """Activate the kill switch -- no further trades allowed."""
        self.is_active = False
        logger.warning("Kill switch ACTIVATED -- trading halted")

    def resume(self) -> None:
        """Deactivate the kill switch -- trading allowed again."""
        self.is_active = True
        logger.info("Kill switch deactivated -- trading resumed")

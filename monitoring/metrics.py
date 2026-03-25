"""Trading performance metrics tracker."""

import math
from datetime import datetime, timezone


class TradingMetrics:
    """Tracks trade history and computes performance statistics."""

    def __init__(self):
        self.trades: list[dict] = []
        self.daily_pnl_cents: int = 0
        self.peak_pnl_cents: int = 0
        self.max_drawdown_cents: int = 0

    def record_trade(
        self,
        ticker: str,
        side: str,
        entry_price: float,
        pnl_cents: int,
        result: str,
    ) -> None:
        """Record a completed trade and update running statistics.

        Args:
            ticker: Market ticker symbol.
            side: 'yes' or 'no'.
            entry_price: Entry price in cents.
            pnl_cents: Realized P&L in cents.
            result: 'win', 'loss', or 'scratch'.
        """
        self.trades.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": ticker,
                "side": side,
                "entry_price": entry_price,
                "pnl_cents": pnl_cents,
                "result": result,
            }
        )

        self.daily_pnl_cents += pnl_cents

        if self.daily_pnl_cents > self.peak_pnl_cents:
            self.peak_pnl_cents = self.daily_pnl_cents

        drawdown = self.peak_pnl_cents - self.daily_pnl_cents
        if drawdown > self.max_drawdown_cents:
            self.max_drawdown_cents = drawdown

    def get_win_rate(self) -> float:
        """Return the fraction of winning trades (0.0 if no trades)."""
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t["result"] == "win")
        return wins / len(self.trades)

    def get_sharpe(self) -> float:
        """Compute annualized Sharpe ratio from per-trade returns.

        Assumes 96 trades per day (one per 15-minute window).
        Returns 0.0 if fewer than 2 trades recorded.
        """
        if len(self.trades) < 2:
            return 0.0

        returns = [t["pnl_cents"] for t in self.trades]
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance)

        if std == 0:
            return 0.0

        trades_per_day = 96
        return (mean / std) * math.sqrt(trades_per_day)

    def get_stats(self) -> dict:
        """Return a summary dict of current performance."""
        wins = sum(1 for t in self.trades if t["result"] == "win")
        losses = sum(1 for t in self.trades if t["result"] == "loss")

        return {
            "total_trades": len(self.trades),
            "wins": wins,
            "losses": losses,
            "win_rate": self.get_win_rate(),
            "daily_pnl": self.daily_pnl_cents,
            "max_drawdown": self.max_drawdown_cents,
            "sharpe": self.get_sharpe(),
        }

    def reset_daily(self) -> None:
        """Reset daily counters while preserving trade history."""
        self.daily_pnl_cents = 0
        self.peak_pnl_cents = 0
        self.max_drawdown_cents = 0

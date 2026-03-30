"""Pair state machine for YES+NO acquisition strategy.

States:
  idle → quoting_both → yes_only_filled → pair_completed
                      → no_only_filled  → pair_completed
                      → pair_completed
       → orphan_unwind → resolved
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

STATES = ("idle", "quoting_both", "yes_only_filled", "no_only_filled",
          "pair_completed", "orphan_unwind", "resolved")


@dataclass
class PairTrade:
    ticker: str = ""
    state: str = "idle"
    # Timing
    created_at: str = ""
    updated_at: str = ""
    market_close_time: str = ""
    # YES leg
    yes_order_id: str = ""
    yes_submitted_price: int = 0
    yes_filled_price: int = 0
    yes_filled: bool = False
    yes_qty: int = 0
    # NO leg
    no_order_id: str = ""
    no_submitted_price: int = 0
    no_filled_price: int = 0
    no_filled: bool = False
    no_qty: int = 0
    # Pair totals
    pair_cost_cents: int = 0
    pair_profit_cents: int = 0
    fees_cents: float = 0.0
    # Exit
    exit_reason: str = ""

    def transition(self, new_state: str, reason: str = "") -> None:
        old = self.state
        self.state = new_state
        self.updated_at = datetime.now(timezone.utc).isoformat()
        if reason:
            self.exit_reason = reason
        logger.info("PAIR %s: %s → %s %s", self.ticker, old, new_state,
                     f"({reason})" if reason else "")

    def record_yes_fill(self, price: int, qty: int = 1) -> None:
        self.yes_filled = True
        self.yes_filled_price = price
        self.yes_qty = qty
        self._update_pair_cost()
        if self.no_filled:
            self.transition("pair_completed", "both_filled")
        elif self.state == "quoting_both":
            self.transition("yes_only_filled")

    def record_no_fill(self, price: int, qty: int = 1) -> None:
        self.no_filled = True
        self.no_filled_price = price
        self.no_qty = qty
        self._update_pair_cost()
        if self.yes_filled:
            self.transition("pair_completed", "both_filled")
        elif self.state == "quoting_both":
            self.transition("no_only_filled")

    def _update_pair_cost(self) -> None:
        yp = self.yes_filled_price if self.yes_filled else self.yes_submitted_price
        np_ = self.no_filled_price if self.no_filled else self.no_submitted_price
        self.pair_cost_cents = yp + np_
        self.pair_profit_cents = 100 - self.pair_cost_cents

    @property
    def is_orphan(self) -> bool:
        return self.state in ("yes_only_filled", "no_only_filled")

    @property
    def is_complete(self) -> bool:
        return self.state == "pair_completed"

    @property
    def is_terminal(self) -> bool:
        return self.state in ("pair_completed", "resolved")

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "state": self.state,
            "yes_price": self.yes_filled_price if self.yes_filled else self.yes_submitted_price,
            "no_price": self.no_filled_price if self.no_filled else self.no_submitted_price,
            "yes_filled": self.yes_filled,
            "no_filled": self.no_filled,
            "pair_cost": self.pair_cost_cents,
            "profit": self.pair_profit_cents,
            "exit_reason": self.exit_reason,
            "created_at": self.created_at,
        }


class PairTracker:
    """Track all active and historical pair trades."""

    def __init__(self):
        self.active: dict[str, PairTrade] = {}  # ticker → PairTrade
        self.completed: list[PairTrade] = []

    def start_pair(self, ticker: str, yes_price: int, no_price: int,
                   close_time: str = "") -> PairTrade:
        pair = PairTrade(
            ticker=ticker,
            state="quoting_both",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            market_close_time=close_time,
            yes_submitted_price=yes_price,
            no_submitted_price=no_price,
        )
        pair._update_pair_cost()
        self.active[ticker] = pair
        logger.info("PAIR START %s: YES@%dc NO@%dc cost=%dc profit=%dc",
                     ticker, yes_price, no_price, pair.pair_cost_cents, pair.pair_profit_cents)
        return pair

    def complete_pair(self, ticker: str) -> PairTrade | None:
        pair = self.active.pop(ticker, None)
        if pair:
            self.completed.append(pair)
        return pair

    def resolve_orphan(self, ticker: str, reason: str = "timeout") -> PairTrade | None:
        pair = self.active.get(ticker)
        if pair and pair.is_orphan:
            pair.transition("orphan_unwind", reason)
            pair.transition("resolved", reason)
            self.completed.append(self.active.pop(ticker))
            return pair
        return None

    def get_stats(self) -> dict:
        completed_pairs = [p for p in self.completed if p.is_complete]
        orphans = [p for p in self.completed if p.exit_reason and "orphan" in p.state or p.exit_reason == "timeout"]
        total_profit = sum(p.pair_profit_cents for p in completed_pairs)
        return {
            "active": len(self.active),
            "completed_pairs": len(completed_pairs),
            "orphans": len(orphans),
            "total_profit_cents": total_profit,
            "avg_pair_cost": round(sum(p.pair_cost_cents for p in completed_pairs) / len(completed_pairs)) if completed_pairs else 0,
        }

"""Pair risk management — orphan leg rules, time stops, size caps."""

import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_ORPHAN_TIMEOUT_S = 30  # cancel unfilled leg after 30s
DEFAULT_DANGER_MOVE_CENTS = 5  # cancel if book moved 5c against completion
DEFAULT_MAX_PAIRS_PER_MARKET = 3
DEFAULT_MAX_ACTIVE_PAIRS = 5
DEFAULT_PAIR_CAP_CENTS = 96  # max total cost for both legs


class PairRiskManager:
    """Risk rules for pair-acquisition trading."""

    def __init__(
        self,
        orphan_timeout_s: float = DEFAULT_ORPHAN_TIMEOUT_S,
        danger_move_cents: int = DEFAULT_DANGER_MOVE_CENTS,
        max_pairs_per_market: int = DEFAULT_MAX_PAIRS_PER_MARKET,
        max_active_pairs: int = DEFAULT_MAX_ACTIVE_PAIRS,
        pair_cap_cents: int = DEFAULT_PAIR_CAP_CENTS,
        budget_cents: int = 2500,  # $25 family budget
    ):
        self.orphan_timeout_s = orphan_timeout_s
        self.danger_move_cents = danger_move_cents
        self.max_pairs_per_market = max_pairs_per_market
        self.max_active_pairs = max_active_pairs
        self.pair_cap_cents = pair_cap_cents
        self.budget_cents = budget_cents
        self.exposure_cents = 0
        self.completed_count = 0
        self.orphan_count = 0
        self.orphan_loss_cents = 0

    def can_open_pair(self, pair_cost_cents: int) -> tuple[bool, str]:
        """Check if we can open a new pair."""
        if self.exposure_cents + pair_cost_cents > self.budget_cents:
            return False, f"budget: {self.exposure_cents + pair_cost_cents}c > {self.budget_cents}c"

        if pair_cost_cents > self.pair_cap_cents:
            return False, f"pair_cap: {pair_cost_cents}c > {self.pair_cap_cents}c"

        # Stop if orphan rate is too high
        total = self.completed_count + self.orphan_count
        if total >= 5 and self.orphan_count / total > 0.5:
            return False, f"orphan_rate: {self.orphan_count}/{total} > 50%"

        return True, "ok"

    def should_unwind_orphan(self, orphan_age_s: float, current_opposite_ask: int,
                              original_opposite_price: int) -> tuple[bool, str]:
        """Check if an orphan leg should be unwound."""
        if orphan_age_s > self.orphan_timeout_s:
            return True, f"timeout: {orphan_age_s:.0f}s > {self.orphan_timeout_s}s"

        adverse_move = current_opposite_ask - original_opposite_price
        if adverse_move > self.danger_move_cents:
            return True, f"danger_move: {adverse_move}c > {self.danger_move_cents}c"

        return False, "ok"

    def record_pair_complete(self, pair_cost_cents: int) -> None:
        self.completed_count += 1
        self.exposure_cents = max(0, self.exposure_cents - pair_cost_cents)

    def record_orphan(self, loss_cents: int) -> None:
        self.orphan_count += 1
        self.orphan_loss_cents += loss_cents

    def record_entry(self, cost_cents: int) -> None:
        self.exposure_cents += cost_cents

    def summary(self) -> dict:
        total = self.completed_count + self.orphan_count
        return {
            "completed_pairs": self.completed_count,
            "orphans": self.orphan_count,
            "orphan_rate": round(self.orphan_count / total * 100) if total else 0,
            "orphan_loss_cents": self.orphan_loss_cents,
            "exposure_cents": self.exposure_cents,
            "budget_cents": self.budget_cents,
        }

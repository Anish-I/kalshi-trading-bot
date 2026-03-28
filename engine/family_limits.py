"""Per-family bankroll caps, position limits, and cooldowns.

Reads family config from market_registry and enforces budget isolation
so one bad family can't wipe the account.
"""

import logging
from datetime import datetime, timezone

from config.market_registry import MarketFamily, MARKET_FAMILIES, PHASE4_CANDIDATES

logger = logging.getLogger(__name__)


class FamilyLimits:
    """Track per-family exposure and enforce budget caps."""

    def __init__(self):
        self._exposure: dict[str, int] = {}  # family_name → cents currently at risk
        self._trade_count: dict[str, int] = {}
        self._last_trade_time: dict[str, str] = {}
        self._cooldown_s: float = 30.0  # minimum seconds between trades in same family

    def _get_family(self, family_name: str) -> MarketFamily | None:
        return MARKET_FAMILIES.get(family_name) or PHASE4_CANDIDATES.get(family_name)

    def record_entry(self, family_name: str, exposure_cents: int) -> None:
        """Record a new position entry."""
        self._exposure[family_name] = self._exposure.get(family_name, 0) + exposure_cents
        self._trade_count[family_name] = self._trade_count.get(family_name, 0) + 1
        self._last_trade_time[family_name] = datetime.now(timezone.utc).isoformat()

    def record_exit(self, family_name: str, exposure_cents: int) -> None:
        """Record a position exit (reduces exposure)."""
        self._exposure[family_name] = max(0, self._exposure.get(family_name, 0) - exposure_cents)

    def can_enter(self, family_name: str, proposed_cents: int) -> tuple[bool, str]:
        """Check if a new entry is allowed under family limits.

        Returns (allowed, reason).
        """
        family = self._get_family(family_name)
        if family is None:
            return False, f"unknown family: {family_name}"

        current = self._exposure.get(family_name, 0)
        if current + proposed_cents > family.budget_cents:
            return False, (
                f"family budget exceeded: {current + proposed_cents}c > "
                f"{family.budget_cents}c cap for {family_name}"
            )

        # Cooldown check
        last = self._last_trade_time.get(family_name)
        if last:
            try:
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds()
                if elapsed < self._cooldown_s:
                    return False, f"cooldown: {elapsed:.0f}s < {self._cooldown_s}s"
            except Exception:
                pass

        return True, "ok"

    def get_exposure(self, family_name: str) -> int:
        """Current exposure in cents for a family."""
        return self._exposure.get(family_name, 0)

    def get_all_exposure(self) -> dict[str, int]:
        """All family exposures."""
        return dict(self._exposure)

    def get_total_exposure(self) -> int:
        """Total exposure across all families."""
        return sum(self._exposure.values())

    def summary(self) -> dict:
        """Dashboard-friendly summary."""
        result = {}
        all_families = {**MARKET_FAMILIES, **PHASE4_CANDIDATES}
        for name, family in all_families.items():
            exp = self._exposure.get(name, 0)
            result[name] = {
                "exposure_cents": exp,
                "budget_cents": family.budget_cents,
                "utilization_pct": round(exp / family.budget_cents * 100, 1) if family.budget_cents else 0,
                "trades": self._trade_count.get(name, 0),
            }
        return result

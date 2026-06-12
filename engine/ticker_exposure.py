"""Per-ticker exposure cap with restart-safe persistence.

Hard ceiling on how many contracts / how much notional may be open on any single
market ticker within a UTC day. This is the control that prevents a single market
from running to a catastrophic loss (the 26MAR25 incident: two BTC_15M NO trades
lost -$110 combined while the nominal per-trade size was tiny).

Design notes:
  * State is persisted to JSON so a process restart can't silently reset open
    exposure mid-day (the in-memory RiskManager reset on every restart — that's
    how the cap was effectively bypassed).
  * Exposure is tracked per UTC day; on a new day the open-exposure map is reset.
  * ``check`` is read-only (use it in the pre-trade gate); ``record`` mutates and
    persists (call it only after an order is actually placed).
  * If the state file's directory is unwritable (e.g. a Windows ``D:/`` path on a
    dev Mac), the tracker degrades to in-memory and logs a warning rather than
    crashing the trader.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _utc_date_str(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d")


class TickerExposureTracker:
    """Tracks and caps open contracts/notional per ticker for the current UTC day."""

    def __init__(
        self,
        state_path: str | Path,
        max_contracts_per_ticker: int,
        max_notional_cents_per_ticker: int,
        now: datetime | None = None,
    ):
        self.state_path = Path(state_path)
        self.max_contracts = int(max_contracts_per_ticker)
        self.max_notional_cents = int(max_notional_cents_per_ticker)
        self._date = _utc_date_str(now)
        # ticker -> {"contracts": int, "notional_cents": int}
        self._open: dict[str, dict[str, int]] = {}
        self._writable = True
        self._load()

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        try:
            if self.state_path.exists():
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                if data.get("utc_date") == self._date:
                    self._open = {
                        str(k): {
                            "contracts": int(v.get("contracts", 0)),
                            "notional_cents": int(v.get("notional_cents", 0)),
                        }
                        for k, v in data.get("open", {}).items()
                    }
                # else: stale day — start fresh (don't carry yesterday's exposure)
        except Exception:
            logger.warning("ticker_exposure: failed to load %s", self.state_path, exc_info=True)
            self._open = {}

    def _save(self) -> None:
        if not self._writable:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"utc_date": self._date, "open": self._open}
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(self.state_path)
        except Exception:
            # Degrade to in-memory rather than crash the trader.
            self._writable = False
            logger.warning(
                "ticker_exposure: state path %s unwritable — degrading to in-memory",
                self.state_path,
                exc_info=True,
            )

    # ------------------------------------------------------------------ #
    # daily rollover
    # ------------------------------------------------------------------ #
    def _maybe_roll_day(self, now: datetime | None = None) -> None:
        today = _utc_date_str(now)
        if today != self._date:
            self._date = today
            self._open = {}
            self._save()

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def open_for(self, ticker: str) -> dict[str, int]:
        return self._open.get(ticker, {"contracts": 0, "notional_cents": 0})

    def check(
        self,
        ticker: str,
        contracts: int,
        notional_cents: int,
        now: datetime | None = None,
    ) -> tuple[bool, str]:
        """Return (allowed, reason) for adding this order to the ticker's exposure."""
        self._maybe_roll_day(now)
        cur = self.open_for(ticker)
        new_contracts = cur["contracts"] + int(contracts)
        new_notional = cur["notional_cents"] + int(notional_cents)
        if new_contracts > self.max_contracts:
            return (
                False,
                f"ticker contracts {new_contracts} > cap {self.max_contracts} "
                f"(open={cur['contracts']}, add={contracts})",
            )
        if new_notional > self.max_notional_cents:
            return (
                False,
                f"ticker notional {new_notional}c > cap {self.max_notional_cents}c "
                f"(open={cur['notional_cents']}c, add={notional_cents}c)",
            )
        return True, "ok"

    def record(
        self,
        ticker: str,
        contracts: int,
        notional_cents: int,
        now: datetime | None = None,
    ) -> None:
        """Add filled/placed exposure for a ticker and persist."""
        self._maybe_roll_day(now)
        cur = self._open.setdefault(ticker, {"contracts": 0, "notional_cents": 0})
        cur["contracts"] += int(contracts)
        cur["notional_cents"] += int(notional_cents)
        self._save()

    def release(self, ticker: str) -> None:
        """Clear a ticker's exposure (e.g. on settlement/unwind) and persist."""
        if ticker in self._open:
            del self._open[ticker]
            self._save()

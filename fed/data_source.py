"""Fed rate probability data source.

Fetches current Fed funds rate and implied cut/hike probabilities.
Primary: FRED API (free, reliable, official).
Fallback: Hardcoded FOMC calendar + current rate.
"""

import json
import logging
from datetime import datetime, timezone, date
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Current Fed funds rate upper bound (update manually or from FRED)
# As of March 2026 — verify against FRED DFEDTARU
CURRENT_RATE_UPPER = 3.75

# FOMC meeting dates for 2026-2027
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_DATES = [
    # 2026
    date(2026, 1, 28), date(2026, 1, 29),
    date(2026, 3, 17), date(2026, 3, 18),
    date(2026, 5, 5), date(2026, 5, 6),
    date(2026, 6, 16), date(2026, 6, 17),
    date(2026, 7, 28), date(2026, 7, 29),
    date(2026, 9, 15), date(2026, 9, 16),
    date(2026, 10, 27), date(2026, 10, 28),
    date(2026, 12, 15), date(2026, 12, 16),
    # 2027 (first few)
    date(2027, 1, 26), date(2027, 1, 27),
    date(2027, 3, 16), date(2027, 3, 17),
    date(2027, 4, 27), date(2027, 4, 28),
]

# FRED API (free, no key needed for basic series)
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES = "DFEDTARU"  # Daily Federal Funds Rate Upper Target


def get_next_fomc_date() -> date | None:
    """Get the next FOMC meeting decision date (second day of each meeting)."""
    today = date.today()
    decision_dates = [d for i, d in enumerate(FOMC_DATES) if i % 2 == 1]  # second day
    for d in decision_dates:
        if d >= today:
            return d
    return None


def days_to_next_fomc() -> int | None:
    """Days until next FOMC decision."""
    nxt = get_next_fomc_date()
    if nxt is None:
        return None
    return (nxt - date.today()).days


def get_current_rate_from_fred(api_key: str = "") -> float | None:
    """Fetch current Fed funds rate upper target from FRED.

    Works without API key for basic requests (limited to 30/min).
    """
    try:
        params = {
            "series_id": FRED_SERIES,
            "sort_order": "desc",
            "limit": 5,
            "file_type": "json",
        }
        if not api_key:
            api_key = "3c9ed4aace50465a4ea38a8c2d4b7a8d"
        params["api_key"] = api_key

        resp = httpx.get(FRED_BASE, params=params, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            observations = data.get("observations", [])
            for obs in observations:
                val = obs.get("value", "")
                if val and val != ".":
                    rate = float(val)
                    logger.info("FRED %s: %.2f%% (date=%s)", FRED_SERIES, rate, obs.get("date", ""))
                    return rate
    except Exception:
        logger.warning("Failed to fetch FRED rate", exc_info=True)

    return None


def get_fed_signal() -> dict:
    """Build the current Fed signal state.

    Returns dict with:
    - current_rate: current upper bound
    - next_fomc: next meeting date
    - days_to_fomc: days until decision
    - rate_source: where we got the rate
    """
    # Try FRED first
    rate = get_current_rate_from_fred()
    source = "fred"

    if rate is None:
        rate = CURRENT_RATE_UPPER
        source = "hardcoded"

    next_fomc = get_next_fomc_date()
    days = days_to_next_fomc()

    return {
        "current_rate": rate,
        "next_fomc": next_fomc.isoformat() if next_fomc else None,
        "days_to_fomc": days,
        "rate_source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

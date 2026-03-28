"""Weather calibration artifact loader and evaluator.

Mirrors engine/crypto_calibration.py pattern.
Loads D:/kalshi-models/weather_calibration.json and provides
bucket-level filtering for the weather market analyzer.
"""

import json
import logging
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

CALIBRATION_PATH = Path(settings.MODEL_DIR) / "weather_calibration.json"


def load_weather_calibration() -> dict:
    """Load the weather calibration artifact."""
    if not CALIBRATION_PATH.exists():
        logger.info("No weather calibration artifact at %s", CALIBRATION_PATH)
        return {}
    try:
        cal = json.loads(CALIBRATION_PATH.read_text())
        n_buckets = len(cal.get("buckets", {}))
        sufficient = sum(1 for b in cal.get("buckets", {}).values() if b.get("sufficient_data"))
        logger.info("Weather calibration loaded: %d buckets (%d sufficient), generated=%s",
                     n_buckets, sufficient, cal.get("generated", "?"))
        return cal
    except Exception:
        logger.warning("Failed to load weather calibration", exc_info=True)
        return {}


def should_skip_bucket(
    calibration: dict,
    city_tier: int,
    strike_type: str,
    edge_bucket: str = "unknown",
) -> tuple[bool, str]:
    """Check if a bucket should be skipped based on calibration data.

    Returns:
        (should_skip, reason)
    """
    if not calibration:
        return False, "no_calibration"

    bucket_key = f"tier{city_tier}_{strike_type}_{edge_bucket}"
    bucket = calibration.get("buckets", {}).get(bucket_key)

    if not bucket:
        return False, "no_bucket_data"

    if not bucket.get("sufficient_data", False):
        return False, f"insufficient_data (n={bucket.get('n_markets', 0)})"

    # Use shrunk win rate for evaluation
    shrunk_rate = bucket.get("shrunk_win_rate", 0.5)

    # Skip if EV is clearly negative
    # For a 50c contract: EV = shrunk_rate * 50 - (1-shrunk_rate) * 50
    # Negative when shrunk_rate < 0.5
    # But cheap contracts (10c) can be +EV even at 15% win rate
    # So we use a more nuanced check:
    if shrunk_rate < 0.30:
        return True, f"low_shrunk_win_rate ({shrunk_rate:.0%})"

    return False, f"ok (shrunk_rate={shrunk_rate:.0%})"


def calibration_summary() -> dict:
    """Summary for the dashboard."""
    cal = load_weather_calibration()
    if not cal:
        return {"loaded": False}

    buckets = cal.get("buckets", {})
    return {
        "loaded": True,
        "version": cal.get("version", ""),
        "generated": cal.get("generated", ""),
        "total_markets": cal.get("total_markets", 0),
        "matched_markets": cal.get("matched_markets", 0),
        "global_win_rate": cal.get("global_win_rate", 0),
        "n_buckets": len(buckets),
        "n_sufficient": sum(1 for b in buckets.values() if b.get("sufficient_data")),
    }

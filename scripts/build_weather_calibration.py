"""
Build weather calibration artifact from settled markets + forecast snapshots.

Computes model accuracy per bucket:
  - city_tier x strike_type x edge_bucket
  - With minimum sample size and shrinkage toward neutral

Output: D:/kalshi-models/weather_calibration.json
"""
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

from kalshi.client import KalshiClient
from config.settings import WEATHER_CITIES, CITY_TIERS, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("weather_calibration")

SNAPSHOT_DIR = Path(settings.DATA_DIR) / "weather_snapshots"
OUTPUT_PATH = Path(settings.MODEL_DIR) / "weather_calibration.json"
MIN_BUCKET_SIZE = 5  # minimum trades per bucket before trusting
SHRINKAGE_WEIGHT = 0.3  # blend bucket rate with global rate


def _get_city_tier(city_short: str) -> int:
    for tier, cities in CITY_TIERS.items():
        if city_short in cities:
            return tier
    return 3


def _edge_bucket(edge: float) -> str:
    if edge < 0.10:
        return "<10%"
    elif edge < 0.20:
        return "10-20%"
    elif edge < 0.30:
        return "20-30%"
    elif edge < 0.50:
        return "30-50%"
    else:
        return "50%+"


def _load_snapshots() -> dict:
    """Load all forecast snapshots, keyed by (city_short, date)."""
    snapshots = {}
    if not SNAPSHOT_DIR.exists():
        return snapshots

    for f in sorted(SNAPSHOT_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").strip().splitlines():
            try:
                rec = json.loads(line)
                key = (rec["city_short"], rec["target_date"])
                snapshots[key] = rec
            except Exception:
                continue

    log.info("Loaded %d forecast snapshots", len(snapshots))
    return snapshots


def _fetch_settled_weather_markets(client: KalshiClient) -> list[dict]:
    """Fetch all settled/finalized weather markets from Kalshi.

    The list endpoint only supports 'closed' status filter. We fetch closed
    markets, then check each individually for 'finalized' status and result.
    """
    settled = []
    checked = 0
    for city in WEATHER_CITIES:
        series = city["series_ticker"]
        try:
            data = client._request(
                "GET", "/markets",
                params={"series_ticker": series, "status": "closed", "limit": 200},
            )
            closed_markets = data.get("markets", [])

            # Check each closed market individually for finalized status
            for m in closed_markets:
                ticker = m.get("ticker", "")
                try:
                    detail = client.get_market(ticker)
                    checked += 1
                    if detail.get("status") in ("finalized", "settled") and detail.get("result"):
                        detail["_city_short"] = city["short"]
                        detail["_city_tier"] = _get_city_tier(city["short"])
                        detail["_market_type"] = city["type"]
                        settled.append(detail)
                except Exception:
                    continue
        except Exception:
            log.warning("Failed to fetch markets for %s", series, exc_info=True)

    log.info("Checked %d closed markets, found %d finalized with results", checked, len(settled))
    return settled


def build_calibration():
    client = KalshiClient()
    snapshots = _load_snapshots()
    settled = _fetch_settled_weather_markets(client)

    if not settled:
        log.warning("No settled weather markets found. Cannot build calibration.")
        return

    # Build buckets: (tier, strike_type, edge_bucket) → list of (predicted_prob, actual_outcome)
    buckets = defaultdict(list)
    global_outcomes = []
    matched = 0
    unmatched = 0

    for mkt in settled:
        ticker = mkt.get("ticker", "")
        result = mkt.get("result", "")
        city_short = mkt.get("_city_short", "")
        tier = mkt.get("_city_tier", 3)

        if not result:
            continue

        # Parse date from ticker (format varies)
        # Try to extract date and find matching snapshot
        # For now, we'll use the close_time to derive the target date
        close_time = mkt.get("close_time", "")
        if not close_time:
            unmatched += 1
            continue

        try:
            target_date = close_time[:10]  # YYYY-MM-DD
        except Exception:
            unmatched += 1
            continue

        # Look up forecast snapshot
        snap_key = (city_short, target_date)
        snap = snapshots.get(snap_key)
        if not snap:
            unmatched += 1
            continue

        # Determine strike type from ticker
        if "-T" in ticker:
            strike_type = "threshold"
        elif "-B" in ticker:
            strike_type = "bracket"
        else:
            strike_type = "unknown"

        # For now, use a simple edge proxy: the model's forecast confidence
        # Real edge requires knowing the market price at time of trade
        # which requires archive data (Phase 2A must run first)
        forecast_mean = snap.get("bias_adjusted_mean_f", snap.get("mean_f", 0))
        forecast_std = snap.get("std_f", 2.5)

        # We don't have the exact model_prob for this market at trade time
        # So we'll use a proxy: was the result consistent with the forecast?
        actual_yes = result.lower() == "yes"

        # Record outcome
        edge_bucket = "unknown"  # Will be refined when we have archive data
        bucket_key = f"tier{tier}_{strike_type}_{edge_bucket}"
        buckets[bucket_key].append({"actual_yes": actual_yes})
        global_outcomes.append(actual_yes)
        matched += 1

    log.info("Matched %d markets to snapshots, %d unmatched", matched, unmatched)

    if not global_outcomes:
        log.warning("No matched markets. Cannot build calibration.")
        return

    # Compute global win rate
    global_win_rate = sum(1 for o in global_outcomes if o) / len(global_outcomes)

    # Build calibration artifact
    calibration = {
        "version": "weather_v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_markets": len(settled),
        "matched_markets": matched,
        "unmatched_markets": unmatched,
        "global_win_rate": round(global_win_rate, 4),
        "min_bucket_size": MIN_BUCKET_SIZE,
        "shrinkage_weight": SHRINKAGE_WEIGHT,
        "buckets": {},
    }

    for bucket_key, outcomes in buckets.items():
        n = len(outcomes)
        raw_win_rate = sum(1 for o in outcomes if o["actual_yes"]) / n if n else 0

        # Shrinkage: blend bucket rate with global rate
        if n >= MIN_BUCKET_SIZE:
            shrunk_rate = (1 - SHRINKAGE_WEIGHT) * raw_win_rate + SHRINKAGE_WEIGHT * global_win_rate
        else:
            shrunk_rate = global_win_rate  # insufficient data, use global

        calibration["buckets"][bucket_key] = {
            "n_markets": n,
            "raw_win_rate": round(raw_win_rate, 4),
            "shrunk_win_rate": round(shrunk_rate, 4),
            "sufficient_data": n >= MIN_BUCKET_SIZE,
        }

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(calibration, indent=2))
    log.info("Weather calibration saved to %s", OUTPUT_PATH)
    log.info("Buckets: %d total, %d with sufficient data",
             len(calibration["buckets"]),
             sum(1 for b in calibration["buckets"].values() if b["sufficient_data"]))

    return calibration


if __name__ == "__main__":
    build_calibration()

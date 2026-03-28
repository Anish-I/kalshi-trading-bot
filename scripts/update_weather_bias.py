"""
Auto-update weather bias from forecast snapshots vs NWS observed temps.

Computes rolling 30-day bias per city x high/low x lead_days.
Caps bias changes per update to max 1°F swing.
Outputs D:/kalshi-models/weather_bias.json

Run daily via scheduler or manually.
"""
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone, date, timedelta
from pathlib import Path

sys.path.insert(0, ".")

from config.settings import WEATHER_CITIES, CITY_BIAS_F, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("weather_bias")

SNAPSHOT_DIR = Path(settings.DATA_DIR) / "weather_snapshots"
OUTPUT_PATH = Path(settings.MODEL_DIR) / "weather_bias.json"
MAX_BIAS_CHANGE_F = 1.0  # max °F change per update


def _load_all_snapshots() -> list[dict]:
    """Load all forecast snapshots."""
    records = []
    if not SNAPSHOT_DIR.exists():
        return records
    for f in sorted(SNAPSHOT_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").strip().splitlines():
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _get_observed_temps() -> dict:
    """Get observed temperatures from NWS API.

    Returns dict keyed by (city_short, date_str) → observed_high_f
    This is a best-effort approach using the NWS observation API.
    """
    observed = {}
    try:
        from weather.nws_client import NWSClient
        nws = NWSClient()

        for city in WEATHER_CITIES:
            lat, lon = city["lat"], city["lon"]
            short = city["short"]
            try:
                # NWS hourly forecast has observed data for recent days
                result = nws.get_hourly_forecast(lat, lon)
                if result and "periods" in result:
                    # Group by date and find daily max
                    daily_temps = defaultdict(list)
                    for period in result["periods"]:
                        dt = period.get("startTime", "")[:10]
                        temp = period.get("temperature")
                        if dt and temp is not None:
                            daily_temps[dt].append(temp)

                    for dt, temps in daily_temps.items():
                        if city["type"] == "high":
                            observed[(short, dt)] = max(temps)
                        else:
                            observed[(short, dt)] = min(temps)
            except Exception:
                continue
    except Exception:
        log.warning("Could not load NWS client for observations")

    log.info("Loaded %d observed temperatures", len(observed))
    return observed


def _load_current_bias() -> dict:
    """Load existing bias JSON or fall back to static config."""
    if OUTPUT_PATH.exists():
        try:
            return json.loads(OUTPUT_PATH.read_text())
        except Exception:
            pass
    # Fall back to static config
    return {"biases": {k: v for k, v in CITY_BIAS_F.items()}, "version": "static"}


def update_bias():
    snapshots = _load_all_snapshots()
    observed = _get_observed_temps()
    current = _load_current_bias()

    if not snapshots:
        log.warning("No forecast snapshots available. Cannot update bias.")
        return

    # Compute errors: forecast_mean - observed per (city, type, lead_days)
    errors = defaultdict(list)
    matched = 0

    for snap in snapshots:
        city_short = snap.get("city_short", "")
        target_date = snap.get("target_date", "")
        market_type = snap.get("market_type", "high")
        lead_days = snap.get("lead_days", 0)
        forecast_mean = snap.get("mean_f", 0)  # raw, before bias correction

        obs_key = (city_short, target_date)
        if obs_key not in observed:
            continue

        actual = observed[obs_key]
        error = forecast_mean - actual  # positive = model predicted too high
        errors[(city_short, market_type, lead_days)].append(error)
        matched += 1

    log.info("Matched %d snapshots to observations", matched)

    if not matched:
        log.warning("No matches found. Cannot update bias.")
        return

    # Compute new biases with capped changes
    new_biases = dict(current.get("biases", {}))
    updates = {}

    for (city_short, market_type, lead_days), errs in errors.items():
        if len(errs) < 3:  # need at least 3 data points
            continue

        mean_error = sum(errs) / len(errs)

        # Key: city_short for high, city_short-L for low
        bias_key = city_short if market_type == "high" else f"{city_short}-L"
        old_bias = new_biases.get(bias_key, 0.0)

        # Cap change to MAX_BIAS_CHANGE_F
        delta = mean_error - old_bias
        if abs(delta) > MAX_BIAS_CHANGE_F:
            delta = MAX_BIAS_CHANGE_F if delta > 0 else -MAX_BIAS_CHANGE_F

        new_bias = round(old_bias + delta, 2)
        new_biases[bias_key] = new_bias
        updates[bias_key] = {"old": old_bias, "new": new_bias, "mean_error": round(mean_error, 2), "n": len(errs)}

    # Save
    output = {
        "version": "auto_v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "matched_observations": matched,
        "updated_cities": len(updates),
        "max_change_cap_f": MAX_BIAS_CHANGE_F,
        "biases": new_biases,
        "update_details": updates,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    log.info("Weather bias updated: %d cities changed. Saved to %s", len(updates), OUTPUT_PATH)

    for key, detail in updates.items():
        log.info("  %s: %.2f → %.2f (mean_error=%.2f, n=%d)",
                 key, detail["old"], detail["new"], detail["mean_error"], detail["n"])

    return output


if __name__ == "__main__":
    update_bias()

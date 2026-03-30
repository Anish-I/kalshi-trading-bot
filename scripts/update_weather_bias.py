"""
Auto-update weather bias from archived forecast snapshots vs settled outcomes.

This updater uses settlement-aligned actual temperatures inferred from finalized
Kalshi weather markets, not forecast periods. Biases are computed per
city-short x market-type x lead-days, with capped updates to avoid sudden jumps.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

from config.settings import CITY_BIAS_F, WEATHER_CITIES, settings
from kalshi.client import KalshiClient
from weather.historical_evaluator import HistoricalWeatherEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("weather_bias")

SNAPSHOT_DIR = Path(settings.DATA_DIR) / "weather_snapshots"
OUTPUT_PATH = Path(settings.MODEL_DIR) / "weather_bias.json"
MAX_BIAS_CHANGE_F = 1.0
MIN_SEGMENT_SAMPLES = 3


def _load_all_snapshots(snapshot_dir: Path | None = None) -> list[dict]:
    snapshot_dir = snapshot_dir or SNAPSHOT_DIR
    records = []
    if not snapshot_dir.exists():
        return records
    for path in sorted(snapshot_dir.glob("*.jsonl")):
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                records.append(json.loads(raw_line))
            except Exception:
                continue
    return records


def _load_current_bias() -> dict:
    if OUTPUT_PATH.exists():
        try:
            payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
            payload.setdefault("biases", {})
            payload.setdefault("biases_by_segment", {})
            return payload
        except Exception:
            pass
    return {
        "version": "static",
        "biases": {k: float(v) for k, v in CITY_BIAS_F.items()},
        "biases_by_segment": {},
    }


def _fetch_actuals(client: KalshiClient) -> dict[tuple[str, str], float]:
    evaluator = HistoricalWeatherEvaluator(client, meteo_client=None)
    actuals: dict[tuple[str, str], float] = {}

    for city in WEATHER_CITIES:
        try:
            markets = evaluator.fetch_settled_markets(city["series_ticker"])
        except Exception:
            log.warning("Failed to fetch settled markets for %s", city["series_ticker"], exc_info=True)
            continue

        for date_str, date_markets in evaluator.group_markets_by_date(markets).items():
            actual_temp = evaluator.infer_actual_temp(date_markets)
            if actual_temp is not None:
                actuals[(city["short"], date_str)] = actual_temp

    log.info("Loaded %d settlement-aligned actual temperatures", len(actuals))
    return actuals


def update_bias(
    client: KalshiClient | None = None,
    snapshot_dir: Path | None = None,
    output_path: Path | None = None,
) -> dict | None:
    client = client or KalshiClient()
    snapshot_dir = snapshot_dir or SNAPSHOT_DIR
    output_path = output_path or OUTPUT_PATH

    snapshots = _load_all_snapshots(snapshot_dir)
    if not snapshots:
        log.warning("No forecast snapshots available. Cannot update bias.")
        return None

    actuals = _fetch_actuals(client)
    if not actuals:
        log.warning("No settlement-aligned actual temperatures available. Cannot update bias.")
        return None

    current = _load_current_bias()

    errors_by_segment: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    matched = 0

    for snapshot in snapshots:
        city_short = snapshot.get("city_short", "")
        target_date = snapshot.get("target_date", "")
        market_type = snapshot.get("market_type", "high")
        lead_days = int(snapshot.get("lead_days", 0) or 0)
        raw_forecast_mean = snapshot.get("mean_f")
        if raw_forecast_mean is None:
            continue

        actual_temp = actuals.get((city_short, target_date))
        if actual_temp is None:
            continue

        error = float(raw_forecast_mean) - float(actual_temp)
        errors_by_segment[(city_short, market_type, lead_days)].append(error)
        matched += 1

    log.info("Matched %d snapshots to settled outcomes", matched)
    if not matched:
        log.warning("No snapshot/outcome matches. Cannot update bias.")
        return None

    new_biases = dict(current.get("biases", {}))
    new_segment_biases = dict(current.get("biases_by_segment", {}))
    updates = {}

    for (city_short, market_type, lead_days), errors in sorted(errors_by_segment.items()):
        if len(errors) < MIN_SEGMENT_SAMPLES:
            continue

        mean_error = sum(errors) / len(errors)
        segment_key = f"{city_short}|{market_type}|{lead_days}"
        old_bias = float(new_segment_biases.get(segment_key, new_biases.get(city_short, CITY_BIAS_F.get(city_short, 0.0))))
        delta = mean_error - old_bias
        if abs(delta) > MAX_BIAS_CHANGE_F:
            delta = MAX_BIAS_CHANGE_F if delta > 0 else -MAX_BIAS_CHANGE_F

        new_bias = round(old_bias + delta, 2)
        new_segment_biases[segment_key] = new_bias
        updates[segment_key] = {
            "city_short": city_short,
            "market_type": market_type,
            "lead_days": lead_days,
            "old": round(old_bias, 2),
            "new": new_bias,
            "mean_error": round(mean_error, 2),
            "n": len(errors),
        }

    # Keep a simple fallback bias per city_short using the shortest available lead.
    grouped_segments: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for segment_key, bias in new_segment_biases.items():
        city_short, _market_type, lead_days_str = segment_key.split("|", 2)
        grouped_segments[city_short].append((int(lead_days_str), float(bias)))

    for city_short, values in grouped_segments.items():
        values.sort(key=lambda item: item[0])
        new_biases[city_short] = round(values[0][1], 2)

    payload = {
        "version": "auto_v2",
        "generated": datetime.now(timezone.utc).isoformat(),
        "matched_observations": matched,
        "updated_segments": len(updates),
        "max_change_cap_f": MAX_BIAS_CHANGE_F,
        "biases": new_biases,
        "biases_by_segment": new_segment_biases,
        "update_details": updates,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Weather bias updated: %d segments changed. Saved to %s", len(updates), output_path)
    return payload


if __name__ == "__main__":
    update_bias()

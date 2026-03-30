"""Runtime loader for dynamic weather bias corrections."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config.settings import CITY_BIAS_F, settings

logger = logging.getLogger(__name__)

_BIAS_PATH = Path(settings.MODEL_DIR) / "weather_bias.json"
_CACHE: dict | None = None
_CACHE_MTIME: float | None = None


def load_weather_biases(force_reload: bool = False) -> dict:
    """Load cached weather biases from disk, falling back to static config."""
    global _CACHE, _CACHE_MTIME

    if not _BIAS_PATH.exists():
        return {
            "version": "static",
            "biases": dict(CITY_BIAS_F),
            "biases_by_segment": {},
        }

    mtime = _BIAS_PATH.stat().st_mtime
    if not force_reload and _CACHE is not None and _CACHE_MTIME == mtime:
        return _CACHE

    try:
        payload = json.loads(_BIAS_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to load weather bias file at %s", _BIAS_PATH, exc_info=True)
        return {
            "version": "static",
            "biases": dict(CITY_BIAS_F),
            "biases_by_segment": {},
        }

    payload.setdefault("biases", {})
    payload.setdefault("biases_by_segment", {})
    _CACHE = payload
    _CACHE_MTIME = mtime
    return payload


def _lead_days_for_target(target_date: str | None, now_utc: datetime | None = None) -> int | None:
    if not target_date:
        return None

    now_utc = now_utc or datetime.now(timezone.utc)
    try:
        current_date = now_utc.strftime("%Y-%m-%d")
        return (
            datetime.strptime(target_date, "%Y-%m-%d")
            - datetime.strptime(current_date, "%Y-%m-%d")
        ).days
    except Exception:
        return None


def get_city_bias(
    city_short: str,
    market_type: str | None = None,
    target_date: str | None = None,
) -> float:
    """Get the best available bias for a city, preferring segmented dynamic bias."""
    payload = load_weather_biases()
    segment_biases = payload.get("biases_by_segment", {}) or {}
    lead_days = _lead_days_for_target(target_date)

    if market_type and lead_days is not None:
        segment_key = f"{city_short}|{market_type}|{lead_days}"
        raw = segment_biases.get(segment_key)
        if raw is not None:
            return float(raw)

    raw = (payload.get("biases", {}) or {}).get(city_short)
    if raw is not None:
        return float(raw)

    return float(CITY_BIAS_F.get(city_short, 0.0))

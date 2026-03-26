"""Open-Meteo API client for Kalshi weather trading bot."""

import logging
import statistics

import httpx

logger = logging.getLogger(__name__)


class OpenMeteoClient:
    """Client for the Open-Meteo weather API (no API key required)."""

    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

    def __init__(self):
        self._client = httpx.Client(timeout=30.0)

    # ------------------------------------------------------------------ #
    #  Daily forecast
    # ------------------------------------------------------------------ #

    def get_daily_forecast(self, lat: float, lon: float, days: int = 3) -> list[dict]:
        """Return daily high/low forecasts for the next *days* days.

        Each element: {date: "YYYY-MM-DD", high_f: float, low_f: float}
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min",
            "temperature_unit": "fahrenheit",
            "timezone": "auto",
            "forecast_days": days,
        }

        resp = self._client.get(self.FORECAST_URL, params=params)
        resp.raise_for_status()
        daily = resp.json()["daily"]

        results: list[dict] = []
        for i, date_str in enumerate(daily["time"]):
            results.append({
                "date": date_str,
                "high_f": daily["temperature_2m_max"][i],
                "low_f": daily["temperature_2m_min"][i],
            })

        logger.info("Open-Meteo daily forecast for (%.4f, %.4f): %d days", lat, lon, len(results))
        return results

    # ------------------------------------------------------------------ #
    #  Ensemble spread
    # ------------------------------------------------------------------ #

    def get_ensemble_spread(self, lat: float, lon: float, target_date: str) -> dict:
        """Return ensemble model spread for the high temp on *target_date*.

        Queries GFS-seamless and ECMWF-IFS025 ensemble models.

        Returns::

            {
                "models": [{"name": str, "high_f": float}, ...],
                "mean": float,
                "std": float,
                "min": float,
                "max": float,
            }

        On failure, falls back to the regular forecast with a default
        std of 2.5 °F.
        """
        try:
            return self._fetch_ensemble(lat, lon, target_date)
        except Exception:
            logger.warning(
                "Ensemble API failed for %s at (%.4f, %.4f), using fallback",
                target_date, lat, lon, exc_info=True,
            )
            return self._ensemble_fallback(lat, lon, target_date)

    def get_gfs_ensemble(self, lat: float, lon: float, target_date: str, daily_var: str = "temperature_2m_max") -> dict:
        """Get full 31-member GFS ensemble forecast for daily max temp.

        Returns dict with member predictions, mean, std, and the fraction
        of members above/below any threshold (for direct probability estimation).

        This is the approach used by the most profitable Kalshi weather bots:
        P(above X) = count(members > X) / total_members
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": daily_var,
            "temperature_unit": "fahrenheit",
            "models": "gfs_seamless",
        }
        resp = self._client.get(self.ENSEMBLE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        # The ensemble API returns member-level data
        # Look for daily temperature_2m_max across all members
        daily = data.get("daily", {})
        times = daily.get("time", [])

        # Find the target date index
        date_idx = None
        for i, d in enumerate(times):
            if d == target_date:
                date_idx = i
                break

        if date_idx is None:
            raise ValueError(f"Date {target_date} not in ensemble forecast")

        # Collect all member values for this date
        # Ensemble API returns temperature_2m_max as a list (one per member per time)
        # or temperature_2m_max_member01, etc.
        member_values = []

        # Try flat array first (all members interleaved)
        raw_highs = daily.get(daily_var, [])
        if raw_highs and isinstance(raw_highs[0], list):
            # Nested: each sub-list is a member's time series
            for member_series in raw_highs:
                if date_idx < len(member_series):
                    member_values.append(member_series[date_idx])
        elif raw_highs:
            # Flat: just one value per time step (not true ensemble)
            if date_idx < len(raw_highs) and raw_highs[date_idx] is not None:
                member_values.append(raw_highs[date_idx])

        # Also try member-specific keys
        for key in sorted(daily.keys()):
            if daily_var in key and key != daily_var:
                vals = daily[key]
                if date_idx < len(vals) and vals[date_idx] is not None:
                    member_values.append(vals[date_idx])

        if not member_values:
            raise ValueError(f"No ensemble member data for {target_date}")

        n = len(member_values)
        mean_val = statistics.mean(member_values)
        std_val = statistics.stdev(member_values) if n > 1 else 2.5

        logger.info(
            "GFS ensemble for %s: %d members, mean=%.1fF, std=%.1fF, range=[%.1f, %.1f]",
            target_date, n, mean_val, std_val, min(member_values), max(member_values),
        )

        return {
            "members": member_values,
            "n_members": n,
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "min": round(min(member_values), 2),
            "max": round(max(member_values), 2),
        }

    def ensemble_prob_above(self, ensemble: dict, threshold: float) -> float:
        """P(temp > threshold) from ensemble member count."""
        members = ensemble["members"]
        above = sum(1 for m in members if m > threshold)
        return above / len(members)

    def ensemble_prob_below(self, ensemble: dict, threshold: float) -> float:
        """P(temp < threshold) from ensemble member count."""
        members = ensemble["members"]
        below = sum(1 for m in members if m < threshold)
        return below / len(members)

    def ensemble_prob_between(self, ensemble: dict, low: float, high: float) -> float:
        """P(low <= temp <= high) from ensemble member count."""
        members = ensemble["members"]
        between = sum(1 for m in members if low <= m <= high)
        return between / len(members)

    def _fetch_ensemble(self, lat: float, lon: float, target_date: str) -> dict:
        """Query the ensemble API and build the spread dict (legacy)."""
        gfs = self.get_gfs_ensemble(lat, lon, target_date)
        return {
            "models": [{"name": f"gfs_member_{i}", "high_f": v} for i, v in enumerate(gfs["members"])],
            "mean": gfs["mean"],
            "std": gfs["std"],
            "min": gfs["min"],
            "max": gfs["max"],
        }

    def _ensemble_fallback(self, lat: float, lon: float, target_date: str) -> dict:
        """Use the regular daily forecast as a single-model fallback."""
        try:
            forecasts = self.get_daily_forecast(lat, lon, days=7)
            for f in forecasts:
                if f["date"] == target_date:
                    high = f["high_f"]
                    return {
                        "models": [{"name": "open_meteo_deterministic", "high_f": high}],
                        "mean": round(high, 2),
                        "std": 2.5,
                        "min": round(high, 2),
                        "max": round(high, 2),
                    }
        except Exception:
            logger.error("Fallback forecast also failed", exc_info=True)

        # Absolute last resort — no data at all
        return {
            "models": [],
            "mean": 0.0,
            "std": 2.5,
            "min": 0.0,
            "max": 0.0,
        }

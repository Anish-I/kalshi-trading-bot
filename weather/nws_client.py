"""National Weather Service API client for Kalshi weather trading bot."""

import logging
import time
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)


class NWSClient:
    """Client for the National Weather Service (weather.gov) API."""

    BASE_URL = "https://api.weather.gov"

    def __init__(self, user_agent: str = "KalshiWeatherBot (kalshi-weather-bot@example.com)"):
        self._client = httpx.Client(
            headers={
                "User-Agent": user_agent,
                "Accept": "application/geo+json",
            },
            timeout=30.0,
        )
        self._grid_cache: dict[tuple[float, float], dict] = {}

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        """GET with one automatic retry after 2 s on failure."""
        for attempt in range(2):
            try:
                resp = self._client.get(url, params=params)
                resp.raise_for_status()
                return resp
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                if attempt == 0:
                    logger.warning("NWS request failed (%s), retrying in 2 s …", exc)
                    time.sleep(2)
                else:
                    logger.error("NWS request failed after retry: %s", exc)
                    raise

    # ------------------------------------------------------------------ #
    #  Grid point resolution
    # ------------------------------------------------------------------ #

    def get_grid_point(self, lat: float, lon: float) -> dict:
        """Resolve a lat/lon to the NWS grid point and forecast URLs.

        Returns a dict with keys:
            gridId, gridX, gridY, forecast, forecastHourly
        """
        cache_key = (round(lat, 4), round(lon, 4))
        if cache_key in self._grid_cache:
            return self._grid_cache[cache_key]

        url = f"{self.BASE_URL}/points/{lat},{lon}"
        resp = self._get(url)
        props = resp.json()["properties"]

        grid_data = {
            "gridId": props["gridId"],
            "gridX": props["gridX"],
            "gridY": props["gridY"],
            "forecast": props["forecast"],
            "forecastHourly": props["forecastHourly"],
        }

        self._grid_cache[cache_key] = grid_data
        logger.debug("Cached grid point for %s: %s/%s/%s",
                      cache_key, grid_data["gridId"], grid_data["gridX"], grid_data["gridY"])
        return grid_data

    # ------------------------------------------------------------------ #
    #  Forecasts
    # ------------------------------------------------------------------ #

    def get_forecast(self, lat: float, lon: float) -> list[dict]:
        """Return the 7-day (12-hour period) forecast for a location.

        Each element contains: name, startTime, endTime, temperature,
        temperatureUnit, isDaytime, shortForecast, detailedForecast.
        """
        grid = self.get_grid_point(lat, lon)
        resp = self._get(grid["forecast"])
        periods = resp.json()["properties"]["periods"]
        return [
            {
                "name": p["name"],
                "startTime": p["startTime"],
                "endTime": p["endTime"],
                "temperature": p["temperature"],
                "temperatureUnit": p["temperatureUnit"],
                "isDaytime": p["isDaytime"],
                "shortForecast": p["shortForecast"],
                "detailedForecast": p["detailedForecast"],
            }
            for p in periods
        ]

    def get_hourly_forecast(self, lat: float, lon: float) -> list[dict]:
        """Return the hourly forecast for a location.

        Each element contains the same keys as get_forecast periods.
        """
        grid = self.get_grid_point(lat, lon)
        resp = self._get(grid["forecastHourly"])
        periods = resp.json()["properties"]["periods"]
        return [
            {
                "name": p["name"],
                "startTime": p["startTime"],
                "endTime": p["endTime"],
                "temperature": p["temperature"],
                "temperatureUnit": p["temperatureUnit"],
                "isDaytime": p["isDaytime"],
                "shortForecast": p["shortForecast"],
                "detailedForecast": p.get("detailedForecast", ""),
            }
            for p in periods
        ]

    # ------------------------------------------------------------------ #
    #  Derived helpers
    # ------------------------------------------------------------------ #

    def get_daily_high(self, lat: float, lon: float, target_date: str) -> float | None:
        """Return the forecast high temperature (°F) for *target_date*.

        *target_date* must be ``"YYYY-MM-DD"``.  The method first checks the
        hourly forecast; if no hours match it falls back to the standard
        (12-hour period) forecast.
        """
        # --- try hourly first ---
        try:
            hourly = self.get_hourly_forecast(lat, lon)
            temps: list[float] = []
            for p in hourly:
                period_date = p["startTime"][:10]  # "YYYY-MM-DD"
                if period_date == target_date:
                    temp = p["temperature"]
                    # Convert to F if necessary
                    if p["temperatureUnit"] == "C":
                        temp = temp * 9.0 / 5.0 + 32.0
                    temps.append(float(temp))
            if temps:
                high = max(temps)
                logger.info("Hourly high for %s at (%.4f, %.4f): %.1f °F",
                            target_date, lat, lon, high)
                return high
        except Exception:
            logger.warning("Hourly forecast unavailable, falling back to standard forecast",
                           exc_info=True)

        # --- fallback: standard forecast ---
        try:
            forecast = self.get_forecast(lat, lon)
            for p in forecast:
                period_date = p["startTime"][:10]
                if period_date == target_date and p["isDaytime"]:
                    temp = p["temperature"]
                    if p["temperatureUnit"] == "C":
                        temp = temp * 9.0 / 5.0 + 32.0
                    logger.info("Standard forecast high for %s at (%.4f, %.4f): %.1f °F",
                                target_date, lat, lon, float(temp))
                    return float(temp)
        except Exception:
            logger.error("Standard forecast also unavailable", exc_info=True)

        logger.warning("No forecast data found for %s at (%.4f, %.4f)", target_date, lat, lon)
        return None

    # ------------------------------------------------------------------ #
    #  Observations
    # ------------------------------------------------------------------ #

    def get_observation(self, station_id: str) -> dict | None:
        """Return the latest observation from a given NWS station.

        Returns a dict with keys: temperature (°F), timestamp, textDescription.
        Returns ``None`` on any error.
        """
        try:
            url = f"{self.BASE_URL}/stations/{station_id}/observations/latest"
            resp = self._get(url)
            props = resp.json()["properties"]

            # Temperature comes in Celsius from the API
            temp_c = props.get("temperature", {}).get("value")
            if temp_c is not None:
                temp_f = temp_c * 9.0 / 5.0 + 32.0
            else:
                temp_f = None

            return {
                "temperature": round(temp_f, 1) if temp_f is not None else None,
                "timestamp": props.get("timestamp"),
                "textDescription": props.get("textDescription", ""),
            }
        except Exception:
            logger.error("Failed to get observation for station %s", station_id, exc_info=True)
            return None

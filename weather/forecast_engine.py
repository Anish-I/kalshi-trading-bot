"""Weather forecast engine combining NWS + Open-Meteo for Kalshi trading."""

import logging
import re
from datetime import datetime, timezone

from scipy.stats import norm

logger = logging.getLogger(__name__)


class WeatherForecastEngine:
    """Blends NWS and Open-Meteo forecasts, then computes probability
    distributions for temperature strike evaluation."""

    def __init__(self, nws_client, meteo_client):
        self.nws = nws_client
        self.meteo = meteo_client
        # Forecast error standard deviation in degF by horizon (days out)
        self.error_std_by_horizon: dict[int, float] = {
            0: 1.5,
            1: 2.5,
            2: 3.5,
            3: 4.5,
        }

    # ------------------------------------------------------------------ #
    #  Forecast retrieval
    # ------------------------------------------------------------------ #

    def get_forecast_high(self, city: dict, target_date: str) -> tuple[float, float]:
        """Return (forecast_mean, forecast_std) for the daily high in degF.

        Parameters
        ----------
        city : dict
            Must contain ``lat``, ``lon``, ``name``.
        target_date : str
            ISO date ``"YYYY-MM-DD"``.

        Returns
        -------
        tuple[float, float]
            ``(forecast_mean, forecast_std)``
        """
        lat, lon, name = city["lat"], city["lon"], city["name"]

        # --- NWS forecast ---
        nws_high: float | None = None
        try:
            nws_high = self.nws.get_daily_high(lat, lon, target_date)
            if nws_high is not None:
                logger.info("NWS high for %s on %s: %.1f degF", name, target_date, nws_high)
        except Exception:
            logger.warning("NWS forecast failed for %s on %s", name, target_date, exc_info=True)

        # --- Open-Meteo forecast ---
        meteo_high: float | None = None
        try:
            forecasts = self.meteo.get_daily_forecast(lat, lon, days=7)
            for fc in forecasts:
                if fc["date"] == target_date:
                    meteo_high = fc["high_f"]
                    break
            if meteo_high is not None:
                logger.info("Open-Meteo high for %s on %s: %.1f degF", name, target_date, meteo_high)
        except Exception:
            logger.warning("Open-Meteo forecast failed for %s on %s", name, target_date, exc_info=True)

        # --- Blend ---
        if nws_high is not None and meteo_high is not None:
            forecast_mean = (nws_high + meteo_high) / 2.0
            logger.info("Blended forecast for %s on %s: %.1f degF (NWS=%.1f, Meteo=%.1f)",
                        name, target_date, forecast_mean, nws_high, meteo_high)
        elif nws_high is not None:
            forecast_mean = nws_high
            logger.info("Using NWS-only forecast for %s on %s: %.1f degF", name, target_date, forecast_mean)
        elif meteo_high is not None:
            forecast_mean = meteo_high
            logger.info("Using Meteo-only forecast for %s on %s: %.1f degF", name, target_date, forecast_mean)
        else:
            raise ValueError(f"No forecast data available for {name} on {target_date}")

        # --- Forecast uncertainty ---
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        horizon_days = (datetime.strptime(target_date, "%Y-%m-%d") - datetime.strptime(today, "%Y-%m-%d")).days
        forecast_std = self.error_std_by_horizon.get(horizon_days, 3.0)

        logger.info("Forecast for %s on %s: mean=%.1f degF, std=%.1f degF (horizon=%d days)",
                    name, target_date, forecast_mean, forecast_std, horizon_days)

        return forecast_mean, forecast_std

    def get_forecast_low(self, city: dict, target_date: str) -> tuple[float, float]:
        """Return (forecast_mean, forecast_std) for the daily LOW temp in degF."""
        lat, lon, name = city["lat"], city["lon"], city["name"]

        nws_low: float | None = None
        try:
            periods = self.nws.get_forecast(lat, lon)
            for p in periods:
                start = p.get("startTime", "")
                if target_date in start and not p.get("isDaytime", True):
                    nws_low = float(p["temperature"])
                    break
            if nws_low is None:
                # Try hourly forecast
                hourly = self.nws.get_hourly_forecast(lat, lon)
                temps = [float(h["temperature"]) for h in hourly if target_date in h.get("startTime", "")]
                if temps:
                    nws_low = min(temps)
        except Exception:
            logger.debug("NWS low forecast failed for %s", name, exc_info=True)

        meteo_low: float | None = None
        try:
            forecasts = self.meteo.get_daily_forecast(lat, lon, days=7)
            for fc in forecasts:
                if fc["date"] == target_date:
                    meteo_low = fc.get("low_f")
                    break
        except Exception:
            logger.debug("Open-Meteo low forecast failed for %s", name, exc_info=True)

        if nws_low is not None and meteo_low is not None:
            forecast_mean = (nws_low + meteo_low) / 2.0
        elif nws_low is not None:
            forecast_mean = nws_low
        elif meteo_low is not None:
            forecast_mean = meteo_low
        else:
            raise ValueError(f"No low temp forecast for {name} on {target_date}")

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        horizon_days = (datetime.strptime(target_date, "%Y-%m-%d") - datetime.strptime(today, "%Y-%m-%d")).days
        forecast_std = self.error_std_by_horizon.get(horizon_days, 3.0)

        return forecast_mean, forecast_std

    def get_forecast_temp(self, city: dict, target_date: str) -> tuple[float, float]:
        """Return forecast for either high or low temp based on city type."""
        temp_type = city.get("type", "high")
        if temp_type == "low":
            return self.get_forecast_low(city, target_date)
        return self.get_forecast_high(city, target_date)

    # ------------------------------------------------------------------ #
    #  Probability calculations
    # ------------------------------------------------------------------ #

    def prob_above(self, mean: float, std: float, threshold: float) -> float:
        """P(temp > threshold)."""
        return float(1.0 - norm.cdf(threshold, loc=mean, scale=std))

    def prob_below(self, mean: float, std: float, threshold: float) -> float:
        """P(temp < threshold)."""
        return float(norm.cdf(threshold, loc=mean, scale=std))

    def prob_between(self, mean: float, std: float, low: float, high: float) -> float:
        """P(low <= temp <= high)."""
        return float(norm.cdf(high, loc=mean, scale=std) - norm.cdf(low, loc=mean, scale=std))

    # ------------------------------------------------------------------ #
    #  Strike evaluation
    # ------------------------------------------------------------------ #

    def evaluate_strike(self, mean: float, std: float, strike_type: str, strike_value: float) -> float:
        """Return the model probability using Gaussian fallback."""
        if strike_type == "above":
            return self.prob_above(mean, std, strike_value)
        elif strike_type == "below":
            return self.prob_below(mean, std, strike_value)
        elif strike_type == "between":
            low = strike_value - 0.5
            high = strike_value + 0.5
            return self.prob_between(mean, std, low, high)
        else:
            logger.warning("Unknown strike type %r, returning 0.5", strike_type)
            return 0.5

    def evaluate_strike_ensemble(
        self, city: dict, target_date: str, strike_type: str, strike_value: float,
    ) -> tuple[float, dict]:
        """Evaluate strike using GFS ensemble members (more accurate than Gaussian).

        Returns (probability, ensemble_info).
        Falls back to Gaussian if ensemble unavailable.
        """
        try:
            ensemble = self.meteo.get_gfs_ensemble(city["lat"], city["lon"], target_date)

            if strike_type == "above":
                prob = self.meteo.ensemble_prob_above(ensemble, strike_value)
            elif strike_type == "below":
                prob = self.meteo.ensemble_prob_below(ensemble, strike_value)
            elif strike_type == "between":
                low = strike_value - 0.5
                high = strike_value + 0.5
                prob = self.meteo.ensemble_prob_between(ensemble, low, high)
            else:
                prob = 0.5

            return prob, ensemble

        except Exception:
            logger.debug("Ensemble failed, falling back to Gaussian", exc_info=True)
            mean, std = self.get_forecast_temp(city, target_date)
            prob = self.evaluate_strike(mean, std, strike_type, strike_value)
            return prob, {"mean": mean, "std": std, "n_members": 0}

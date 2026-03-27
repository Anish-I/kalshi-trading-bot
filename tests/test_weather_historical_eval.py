"""Tests for honest historical weather evaluation helpers."""

import sys

sys.path.insert(0, ".")

from weather.historical_evaluator import HistoricalWeatherEvaluator
from weather.open_meteo_client import OpenMeteoClient


class DummyKalshiClient:
    def __init__(self, markets_by_series):
        self.markets_by_series = markets_by_series

    def _request(self, method, path, params=None):
        series_ticker = params["series_ticker"]
        return {"markets": self.markets_by_series.get(series_ticker, []), "cursor": None}


class DummyMeteoClient:
    def __init__(self, rows):
        self.rows = rows

    def get_previous_run_daily_forecast(self, lat, lon, start_date, end_date, lead_days=1):
        return list(self.rows)


def test_aggregate_daily_temperatures():
    rows = OpenMeteoClient._aggregate_daily_temperatures(
        [
            "2026-03-20T00:00",
            "2026-03-20T06:00",
            "2026-03-21T00:00",
            "2026-03-21T06:00",
        ],
        [41.0, 55.0, 37.0, 48.0],
    )

    assert rows == [
        {"date": "2026-03-20", "high_f": 55.0, "low_f": 41.0, "hour_count": 2},
        {"date": "2026-03-21", "high_f": 48.0, "low_f": 37.0, "hour_count": 2},
    ]


def test_infer_actual_temp_prefers_yes_bracket():
    markets = [
        {"ticker": "KXHIGHNY-26MAR20-T58", "result": "no", "rules_primary": "If the highest temperature is greater than 58°, then the market resolves to Yes."},
        {"ticker": "KXHIGHNY-26MAR20-B56.5", "result": "yes", "rules_primary": "If the highest temperature is between 56-57°, then the market resolves to Yes."},
    ]

    assert HistoricalWeatherEvaluator.infer_actual_temp(markets) == 56.5


def test_parse_date_from_weather_ticker():
    assert HistoricalWeatherEvaluator.parse_date_from_ticker("KXHIGHNY-26MAR20-T58") == "2026-03-20"


def test_evaluate_city_scores_archived_forecasts_without_leakage():
    city = {
        "name": "New York",
        "short": "NYC",
        "lat": 40.7829,
        "lon": -73.9654,
        "series_ticker": "KXHIGHNY",
        "type": "high",
    }
    markets = [
        {
            "ticker": "KXHIGHNY-26MAR20-B56.5",
            "result": "yes",
            "rules_primary": "If the highest temperature is between 56-57°, then the market resolves to Yes.",
        },
        {
            "ticker": "KXHIGHNY-26MAR20-T55",
            "result": "yes",
            "rules_primary": "If the highest temperature is greater than 55°, then the market resolves to Yes.",
        },
        {
            "ticker": "KXHIGHNY-26MAR20-T58",
            "result": "no",
            "rules_primary": "If the highest temperature is greater than 58°, then the market resolves to Yes.",
        },
        {
            "ticker": "KXHIGHNY-26MAR20-T51",
            "result": "no",
            "rules_primary": "If the highest temperature is less than 51°, then the market resolves to Yes.",
        },
    ]
    meteo_rows = [
        {"date": "2026-03-20", "high_f": 56.0, "low_f": 45.0, "hour_count": 24},
    ]

    evaluator = HistoricalWeatherEvaluator(
        DummyKalshiClient({"KXHIGHNY": markets}),
        DummyMeteoClient(meteo_rows),
    )
    results = evaluator.evaluate_city(city, lead_days=1)

    assert results["n_total_dates"] == 1
    assert results["n_covered_dates"] == 1
    assert results["mae_deg_f"] == 0.5
    assert results["bias_deg_f"] == -0.5
    assert results["market_accuracy"] == 1.0
    assert results["market_correct"] == 4
    assert results["market_total"] == 4

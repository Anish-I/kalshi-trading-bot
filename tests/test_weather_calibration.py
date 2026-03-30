import json
from pathlib import Path

import pandas as pd

from scripts import build_weather_calibration as build_module
from weather.market_analyzer import WeatherMarketAnalyzer


class DummyKalshiClient:
    def __init__(self, closed_markets: list[dict], market_details: dict[str, dict]) -> None:
        self._closed_markets = closed_markets
        self._market_details = market_details

    def _request(self, method, path, params=None):
        if path == "/markets":
            return {"markets": list(self._closed_markets)}
        raise AssertionError(f"Unexpected request: {method} {path}")

    def get_market(self, ticker: str):
        return dict(self._market_details[ticker])


def test_build_weather_calibration_uses_model_probabilities_and_archive_quotes(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "weather_snapshots"
    snapshot_dir.mkdir(parents=True)
    snapshot_path = snapshot_dir / "20260320.jsonl"
    snapshot_path.write_text(
        json.dumps({
            "city_short": "PHX",
            "market_type": "high",
            "target_date": "2026-03-20",
            "issue_time_utc": "2026-03-20T15:00:00+00:00",
            "lead_days": 1,
            "member_values_f": [54.0, 56.0, 57.0, 58.0],
            "mean_f": 56.25,
            "std_f": 1.7,
            "bias_adjusted_mean_f": 56.25,
        })
        + "\n",
        encoding="utf-8",
    )

    archive_dir = tmp_path / "market_archive" / "2026-03-20"
    archive_dir.mkdir(parents=True)
    pd.DataFrame([
        {
            "timestamp": "2026-03-20T15:00:00+00:00",
            "ticker": "KXHIGHTPHX-26MAR20-T55",
            "yes_ask": 0.45,
            "no_ask": 0.60,
        }
    ]).to_parquet(archive_dir / "KXHIGH.parquet", index=False)

    monkeypatch.setattr(build_module, "WEATHER_CITIES", [{
        "name": "Phoenix",
        "short": "PHX",
        "series_ticker": "KXHIGHTPHX",
        "type": "high",
    }])
    monkeypatch.setattr("engine.market_archive.ARCHIVE_DIR", tmp_path / "market_archive")

    client = DummyKalshiClient(
        closed_markets=[{"ticker": "KXHIGHTPHX-26MAR20-T55"}],
        market_details={
            "KXHIGHTPHX-26MAR20-T55": {
                "ticker": "KXHIGHTPHX-26MAR20-T55",
                "status": "finalized",
                "result": "yes",
                "close_time": "2026-03-20T16:00:00+00:00",
                "rules_primary": "If the highest temperature is greater than 55°, then the market resolves to Yes.",
            }
        },
    )
    output_path = tmp_path / "weather_calibration.json"

    artifact = build_module.build_calibration(
        client=client,
        snapshot_dir=snapshot_dir,
        output_path=output_path,
    )

    assert artifact is not None
    bucket = artifact["buckets"]["tier1_above_30-50%"]
    assert artifact["matched_markets"] == 1
    assert artifact["global_trade_win_rate"] == 1.0
    assert bucket["raw_trade_win_rate"] == 1.0
    assert bucket["avg_model_prob"] == 0.75
    assert bucket["avg_entry_price_cents"] == 45.0
    assert bucket["avg_edge"] == 0.3


class DummyForecastEngine:
    def get_forecast_temp(self, city, target_date):
        return (57.0, 2.0)

    def _get_cached_ensemble(self, city, target_date, temp_var):
        return {"members": [54.0, 56.0, 57.0, 58.0], "mean": 56.25, "std": 1.7, "n_members": 4}

    def evaluate_strike_ensemble(self, city, target_date, strike_type, strike_value):
        return (0.75, {})


class DummyOpenMarkets:
    def _request(self, method, path, params=None):
        return {
            "markets": [{
                "ticker": "KXHIGHTPHX-26MAR20-T55",
                "rules_primary": "If the highest temperature is greater than 55°, then the market resolves to Yes.",
                "yes_bid_dollars": 0.40,
                "yes_ask_dollars": 0.50,
                "no_bid_dollars": 0.45,
                "no_ask_dollars": 0.55,
            }]
        }


def test_market_analyzer_applies_weather_calibration_gate(monkeypatch, tmp_path):
    calibration = {
        "min_trade_win_rate": 0.42,
        "buckets": {
            "tier1_above_20-30%": {
                "n_markets": 10,
                "sufficient_data": True,
                "shrunk_trade_win_rate": 0.20,
            }
        },
    }

    monkeypatch.setattr("weather.market_analyzer.load_weather_calibration", lambda: calibration)
    monkeypatch.setattr("weather.market_analyzer.get_city_bias", lambda city_short, market_type=None, target_date=None: 0.0)
    monkeypatch.setattr(WeatherMarketAnalyzer, "_save_forecast_snapshot", lambda *args, **kwargs: None)

    analyzer = WeatherMarketAnalyzer(DummyOpenMarkets(), DummyForecastEngine())
    city = {
        "name": "Phoenix",
        "short": "PHX",
        "series_ticker": "KXHIGHTPHX",
        "type": "high",
    }

    opportunities = analyzer.find_best_trades([city], min_edge=0.15)

    assert opportunities == []

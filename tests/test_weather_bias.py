import json

from engine.weather_bias import get_city_bias, load_weather_biases
from scripts import update_weather_bias as update_module


def test_update_weather_bias_uses_settlement_actuals_and_caps_changes(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "weather_snapshots"
    snapshot_dir.mkdir(parents=True)
    snapshot_path = snapshot_dir / "20260320.jsonl"
    snapshot_path.write_text(
        "\n".join(
            json.dumps({
                "city_short": "TEST",
                "market_type": "high",
                "target_date": date_str,
                "lead_days": 1,
                "mean_f": 60.0,
            })
            for date_str in ("2026-03-20", "2026-03-21", "2026-03-22")
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        update_module,
        "_fetch_actuals",
        lambda client: {
            ("TEST", "2026-03-20"): 56.0,
            ("TEST", "2026-03-21"): 56.0,
            ("TEST", "2026-03-22"): 56.0,
        },
    )
    monkeypatch.setattr(
        update_module,
        "_load_current_bias",
        lambda: {"version": "static", "biases": {}, "biases_by_segment": {}},
    )

    output_path = tmp_path / "weather_bias.json"
    payload = update_module.update_bias(client=object(), snapshot_dir=snapshot_dir, output_path=output_path)

    assert payload is not None
    assert payload["biases_by_segment"]["TEST|high|1"] == 1.0
    assert payload["biases"]["TEST"] == 1.0


def test_runtime_weather_bias_prefers_segmented_file(tmp_path, monkeypatch):
    bias_path = tmp_path / "weather_bias.json"
    bias_path.write_text(
        json.dumps({
            "version": "auto_v2",
            "biases": {"TEST": 0.5},
            "biases_by_segment": {"TEST|high|1": 1.25},
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr("engine.weather_bias._BIAS_PATH", bias_path)
    monkeypatch.setattr("engine.weather_bias._CACHE", None)
    monkeypatch.setattr("engine.weather_bias._CACHE_MTIME", None)
    monkeypatch.setattr("engine.weather_bias._lead_days_for_target", lambda target_date, now_utc=None: 1)

    loaded = load_weather_biases(force_reload=True)

    assert loaded["biases"]["TEST"] == 0.5
    assert get_city_bias("TEST", "high", "2026-03-29") == 1.25

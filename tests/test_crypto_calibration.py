from datetime import datetime, timezone

from engine.crypto_calibration import (
    bucket_price_cents,
    build_calibration_artifact,
    lookup_bucket,
    session_tag_for_timestamp,
)


def _event(timestamp: str, side: str, entry_price_cents: int, won: bool, pnl_cents: int, session_tag: str = "us_core"):
    return {
        "timestamp": timestamp,
        "session_tag": session_tag,
        "side": side,
        "entry_price_cents": entry_price_cents,
        "won": won,
        "pnl_cents": pnl_cents,
        "xgb_vote": side.upper(),
        "xgb_conf": 0.64,
        "momentum_vote": side.upper(),
        "momentum_conf": 0.58,
        "label": 2 if side == "up" else 0,
    }


def test_bucket_price_cents_is_deterministic():
    assert bucket_price_cents(0) == 0
    assert bucket_price_cents(2) == 0
    assert bucket_price_cents(3) == 5
    assert bucket_price_cents(7) == 5
    assert bucket_price_cents(8) == 10
    assert bucket_price_cents(47) == 45
    assert bucket_price_cents(48) == 50
    assert bucket_price_cents(97) == 95
    assert bucket_price_cents(99) == 100
    assert bucket_price_cents(47) == bucket_price_cents(47)


def test_build_calibration_artifact_marks_support_threshold_cells():
    events = []
    for i in range(49):
        events.append(_event(f"2026-03-01T00:{i:02d}:00+00:00", "up", 46, True, 54))
    for i in range(50):
        events.append(_event(f"2026-03-01T01:{i:02d}:00+00:00", "down", 48, True, 52))

    artifact = build_calibration_artifact(
        events,
        metadata={
            "data_file": "D:/kalshi-data/binance_history/klines_90d.parquet",
            "model_path": "D:/kalshi-models/latest_model.json",
            "schema_path": "D:/kalshi-models/latest_model.schema.json",
            "source_mode": "bars_90d_replay",
            "price_source": "proxy",
            "forward_bars": 15,
            "threshold_bps": 0.001,
            "signal_rule": "xgb+momentum_conjunction",
            "features_source": "features.honest_features.compute_honest_features",
        },
        generated_at="2026-03-27T00:00:00+00:00",
        min_trades=50,
        ev_buffer_cents=2.0,
        min_net_ev_cents=1.0,
    )

    assert artifact["summary"]["bucket_count"] == 2
    assert artifact["summary"]["tradable_bucket_count"] == 1

    down_cell = artifact["cells"][0]
    up_cell = artifact["cells"][1]

    assert down_cell["side"] == "down"
    assert down_cell["price_bucket"] == 50
    assert down_cell["n_trades"] == 50
    assert down_cell["tradable"] is True

    assert up_cell["side"] == "up"
    assert up_cell["price_bucket"] == 45
    assert up_cell["n_trades"] == 49
    assert up_cell["tradable"] is False

    looked_up = lookup_bucket(artifact, "up", 47)
    assert looked_up is not None
    assert looked_up["price_bucket"] == 45


def test_calibration_artifact_metadata_is_stable_and_complete():
    events = [
        _event("2026-03-01T15:30:00+00:00", "up", 55, True, 45, "us_core"),
        _event("2026-03-02T02:30:00+00:00", "down", 65, False, -65, "weekend"),
    ]

    artifact_1 = build_calibration_artifact(
        events,
        metadata={
            "data_file": "D:/kalshi-data/binance_history/klines_90d.parquet",
            "model_path": "D:/kalshi-models/latest_model.json",
            "schema_path": "D:/kalshi-models/latest_model.schema.json",
            "source_mode": "bars_90d_replay",
            "price_source": "proxy",
            "forward_bars": 15,
            "threshold_bps": 0.001,
            "signal_rule": "xgb+momentum_conjunction",
            "features_source": "features.honest_features.compute_honest_features",
        },
        generated_at="2026-03-27T00:00:00+00:00",
        min_trades=50,
        ev_buffer_cents=2.0,
        min_net_ev_cents=1.0,
    )
    artifact_2 = build_calibration_artifact(
        events,
        metadata={
            "data_file": "D:/kalshi-data/binance_history/klines_90d.parquet",
            "model_path": "D:/kalshi-models/latest_model.json",
            "schema_path": "D:/kalshi-models/latest_model.schema.json",
            "source_mode": "bars_90d_replay",
            "price_source": "proxy",
            "forward_bars": 15,
            "threshold_bps": 0.001,
            "signal_rule": "xgb+momentum_conjunction",
            "features_source": "features.honest_features.compute_honest_features",
        },
        generated_at="2026-03-27T00:00:00+00:00",
        min_trades=50,
        ev_buffer_cents=2.0,
        min_net_ev_cents=1.0,
    )

    assert artifact_1["metadata"]["artifact_type"] == "crypto_conjunction_calibration"
    assert artifact_1["metadata"]["artifact_version"] == 1
    assert artifact_1["metadata"]["bucket_width_cents"] == 5
    assert artifact_1["metadata"]["min_trades"] == 50
    assert artifact_1["metadata"]["ev_buffer_cents"] == 2.0
    assert artifact_1["metadata"]["min_net_ev_cents"] == 1.0
    assert artifact_1["metadata"]["data_file"].endswith("klines_90d.parquet")
    assert artifact_1["metadata"]["model_path"].endswith("latest_model.json")
    assert artifact_1["metadata"]["schema_path"].endswith("latest_model.schema.json")
    assert artifact_1["metadata"]["artifact_id"] == artifact_2["metadata"]["artifact_id"]
    assert artifact_1["cells"] == artifact_2["cells"]
    assert artifact_1["session_stats"]["us_core"]["n_trades"] == 1
    assert artifact_1["session_stats"]["weekend"]["n_trades"] == 1
    assert session_tag_for_timestamp(datetime(2026, 3, 27, 12, 30, tzinfo=timezone.utc)) == "us_off"
    assert session_tag_for_timestamp(datetime(2026, 3, 28, 12, 30, tzinfo=timezone.utc)) == "weekend"

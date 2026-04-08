import json
import tempfile
from pathlib import Path

from engine.crypto_decision import (
    calibration_summary_view,
    evaluate_calibrated_trade,
    gross_ev_cents_per_contract,
    load_crypto_calibration,
)


def _write_artifact(payload: dict) -> Path:
    tmpdir = Path(tempfile.mkdtemp())
    path = tmpdir / "crypto_conjunction_calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_gross_ev_math():
    ev = gross_ev_cents_per_contract(0.485, 45)
    assert round(ev, 2) == 3.50


def test_lookup_and_evaluate_supported_bucket():
    path = _write_artifact(
        {
            "metadata": {
                "artifact_id": "artifact-123",
                "generated_at": "2026-03-27T00:00:00+00:00",
                "bucket_width_cents": 5,
            },
            "summary": {
                "tradable_bucket_count": 1,
                "best_bucket": {"side": "yes", "price_bucket": 45, "gross_ev_cents_per_contract": 3.5},
                "worst_bucket": {"side": "yes", "price_bucket": 45, "gross_ev_cents_per_contract": 3.5},
            },
            "cells": [
                {
                    "side": "yes",
                    "price_bucket": 45,
                    "n_trades": 60,
                    "win_rate": 0.485,
                    "gross_ev_cents_per_contract": 3.5,
                    "net_ev_cents_per_contract": 1.5,
                    "tradable": True,
                }
            ],
        }
    )
    artifact = load_crypto_calibration(path)
    decision = evaluate_calibrated_trade(
        artifact,
        side="yes",
        entry_cents=45,
        min_trades=50,
        ev_buffer_cents=2.0,
        min_net_ev_cents=1.0,
    )

    assert decision["status"] == "trading"
    assert decision["price_bucket"] == "45c"
    assert round(decision["calibrated_p_win"], 3) == 0.485
    assert round(decision["gross_ev_cents_per_contract"], 2) == 3.50
    assert round(decision["net_ev_cents_per_contract"], 2) == 1.50


def test_missing_or_unsupported_bucket_returns_no_calibration():
    path = _write_artifact(
        {
            "metadata": {
                "artifact_id": "artifact-456",
                "generated_at": "2026-03-27T00:00:00+00:00",
                "bucket_width_cents": 5,
            },
            "summary": {},
            "cells": [
                {
                    "side": "no",
                    "price_bucket": 50,
                    "n_trades": 30,
                    "win_rate": 0.60,
                    "gross_ev_cents_per_contract": 10.0,
                    "net_ev_cents_per_contract": 8.0,
                    "tradable": False,
                }
            ],
        }
    )
    artifact = load_crypto_calibration(path)
    decision = evaluate_calibrated_trade(
        artifact,
        side="no",
        entry_cents=48,
        min_trades=50,
        ev_buffer_cents=2.0,
        min_net_ev_cents=1.0,
    )
    summary = calibration_summary_view(artifact)

    assert decision["status"] == "no_calibration"
    assert decision["price_bucket"] == "50c"
    assert summary["exists"] is True
    assert summary["version"] == "artifact-456"


def test_loader_tolerates_kalshi_archive_metadata():
    """Phase 5: loader must accept new top-level metadata fields."""
    path = _write_artifact(
        {
            "version": "kalshi-archive-001",
            "generated_at": "2026-04-08T00:00:00+00:00",
            "bucket_size_cents": 5,
            "data_source": "kalshi_archive",
            "kalshi_archive_range": "2026-03-09 to 2026-04-08",
            "kalshi_series": ["KXBTC15M", "KXETH15M"],
            "summary": {
                "tradable_bucket_count": 1,
                "best_bucket": {"side": "yes", "price_bucket": 45},
                "worst_bucket": {"side": "yes", "price_bucket": 45},
            },
            "rows": [
                {
                    "side": "yes",
                    "price_bucket": 45,
                    "n_trades": 80,
                    "win_rate": 0.50,
                    "gross_ev_cents_per_contract": 5.0,
                    "net_ev_cents_per_contract": 3.0,
                    "tradable": True,
                }
            ],
        }
    )
    artifact = load_crypto_calibration(path)
    assert artifact["exists"] is True
    assert artifact["version"] == "kalshi-archive-001"
    assert artifact["bucket_size_cents"] == 5

    decision = evaluate_calibrated_trade(
        artifact,
        side="yes",
        entry_cents=45,
        min_trades=50,
        ev_buffer_cents=2.0,
        min_net_ev_cents=1.0,
    )
    assert decision["status"] == "trading"
    assert decision["price_bucket"] == "45c"
    assert round(decision["calibrated_p_win"], 3) == 0.500
    assert decision["bucket_trade_count"] == 80

"""Tests for the value-strategy calibration backtest harness."""
from scripts.backtest_value_strategy import (
    run_calibration_backtest,
    synthetic_bars,
    _calibration_metrics,
)
import numpy as np


def test_synthetic_model_is_calibrated_and_beats_coinflip():
    metrics = run_calibration_backtest(synthetic_bars(n_minutes=6000, seed=3))
    assert metrics["n"] > 1000
    # On a GBM path the lognormal model should be well below the 0.25 coin flip.
    assert metrics["brier"] < 0.20
    assert metrics["beats_baseline"] is True
    assert metrics["directional_accuracy"] > 0.6


def test_reliability_buckets_track_diagonal():
    metrics = run_calibration_backtest(synthetic_bars(n_minutes=8000, seed=11))
    # In confident buckets, predicted prob should track realized frequency.
    for row in metrics["reliability"]:
        if row["n"] >= 200:
            assert abs(row["mean_pred"] - row["mean_real"]) < 0.12


def test_calibration_metrics_empty():
    m = _calibration_metrics(np.array([]), np.array([]))
    assert m["n"] == 0


def test_perfect_predictions_have_zero_brier():
    p = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    m = _calibration_metrics(p, y)
    assert m["brier"] == 0.0
    assert m["directional_accuracy"] == 1.0

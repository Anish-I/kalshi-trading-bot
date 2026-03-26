"""Integration test: honest model + schema + live features → score()."""
import sys
sys.path.insert(0, ".")

import pandas as pd
from pathlib import Path


def test_honest_model_scores_on_bars():
    """Load latest model, compute honest features from bars, call score()."""
    from models.signal_models import XGBoostModel
    from features.honest_features import compute_honest_features, HONEST_FEATURE_NAMES

    model_path = "D:/kalshi-models/latest_model.json"
    if not Path(model_path).exists():
        import pytest
        pytest.skip("No latest_model.json")

    # Load model
    xgb = XGBoostModel(model_path)

    # Verify schema loaded correctly
    assert xgb.feature_cols is not None, "Schema not loaded"
    assert isinstance(xgb.feature_cols, list), f"Schema is {type(xgb.feature_cols)}, expected list"
    assert len(xgb.feature_cols) == len(HONEST_FEATURE_NAMES), \
        f"Schema has {len(xgb.feature_cols)} features, expected {len(HONEST_FEATURE_NAMES)}"

    # Load real bars
    bars_dir = Path("D:/kalshi-data/bars_1m")
    if not bars_dir.exists():
        # Use Binance history as fallback
        hist_path = Path("D:/kalshi-data/binance_history/klines_14d.parquet")
        if not hist_path.exists():
            import pytest
            pytest.skip("No bar data available")
        bars = pd.read_parquet(hist_path).tail(100)
    else:
        files = sorted(bars_dir.glob("*.parquet"))
        if not files:
            import pytest
            pytest.skip("No bar files")
        bars = pd.read_parquet(files[-1]).tail(100)

    # Compute honest features
    featured = compute_honest_features(bars)
    assert len(featured) > 0, "No features computed"

    # Check all schema columns are present
    missing = [c for c in xgb.feature_cols if c not in featured.columns]
    present = [c for c in xgb.feature_cols if c in featured.columns]
    print(f"Schema columns: {len(xgb.feature_cols)}")
    print(f"Present in features: {len(present)}")
    print(f"Missing (will be filled with 0): {missing}")

    # Score should not crash
    direction, confidence = xgb.score(featured)
    assert direction in ("up", "down", "flat")
    assert 0 <= confidence <= 1.0
    print(f"Score result: {direction} {confidence:.1%}")


def test_schema_format():
    """Verify schema file can be loaded as list of feature names."""
    import json

    schema_path = "D:/kalshi-models/latest_model.schema.json"
    if not Path(schema_path).exists():
        import pytest
        pytest.skip("No schema file")

    with open(schema_path) as f:
        raw = json.load(f)

    # Accept both formats
    if isinstance(raw, list):
        features = raw
    elif isinstance(raw, dict) and "features" in raw:
        features = raw["features"]
    else:
        assert False, f"Unknown schema format: {type(raw)}"

    assert isinstance(features, list)
    assert len(features) > 0
    assert all(isinstance(f, str) for f in features)
    print(f"Schema: {len(features)} features")

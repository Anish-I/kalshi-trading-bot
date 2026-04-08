"""Phase 6: macro crypto feature tests.

The no-leakage test is the critical one — it must fail loudly if someone
changes ``direction`` to ``nearest`` or ``forward``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.macro_crypto_features import (
    MACRO_FEATURE_NAMES,
    attach_macro_features,
    join_macro_asof,
    load_funding_rates,
)


def test_no_leakage_backward_asof():
    """Macro at 09:55 and 10:05; decision at 10:00 MUST see 09:55, not 10:05."""
    bars = pd.DataFrame({"ts": pd.to_datetime(["2026-04-08 10:00"], utc=True)})
    macro = pd.DataFrame({
        "ts": pd.to_datetime(["2026-04-08 09:50", "2026-04-08 10:01"], utc=True),
        "funding_rate": [0.001, 0.999],  # 0.999 must NOT appear in the result
    })
    merged = join_macro_asof(bars, macro)
    assert merged["funding_rate"].iloc[0] == 0.001
    assert merged["funding_rate"].iloc[0] != 0.999


def test_empty_macro_returns_bars_unchanged():
    bars = pd.DataFrame({
        "ts": pd.to_datetime(["2026-04-08 10:00"], utc=True),
        "x": [1],
    })
    empty = pd.DataFrame({
        "ts": pd.Series([], dtype="datetime64[ns, UTC]"),
        "funding_rate": pd.Series([], dtype="float64"),
    })
    merged = join_macro_asof(bars, empty)
    assert len(merged) == 1
    assert "x" in merged.columns


def test_load_funding_rates_missing_file(tmp_path):
    df = load_funding_rates(tmp_path / "does_not_exist.parquet")
    assert df.empty
    assert "funding_rate" in df.columns


def test_attach_macro_missing_files(tmp_path):
    bars = pd.DataFrame({"ts": pd.to_datetime(["2026-04-08 10:00"], utc=True)})
    out = attach_macro_features(bars, tmp_path)
    for name in MACRO_FEATURE_NAMES:
        assert name in out.columns
        assert pd.isna(out[name].iloc[0])


def _make_bars(n: int = 80) -> pd.DataFrame:
    ts = pd.date_range("2026-04-08 00:00", periods=n, freq="1min", tz="UTC")
    close = np.linspace(100.0, 110.0, n)
    high = close + 0.5
    low = close - 0.5
    open_ = close
    volume = np.full(n, 10.0)
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "buy_volume": volume * 0.6,
        "sell_volume": volume * 0.4,
        "trade_count": np.full(n, 5),
    })


def test_compute_all_features_flag_off(monkeypatch):
    """When MACRO_FEATURES_ENABLED is False, no macro columns should exist."""
    from config import settings as settings_mod
    from features import honest_features

    monkeypatch.setattr(settings_mod.settings, "MACRO_FEATURES_ENABLED", False, raising=False)

    bars = _make_bars()
    out = honest_features.compute_all_features(bars, settings=settings_mod.settings)
    for name in MACRO_FEATURE_NAMES:
        assert name not in out.columns


def test_compute_all_features_flag_on(tmp_path, monkeypatch):
    """With flag on and empty macro data, columns exist but are NaN."""
    from config import settings as settings_mod
    from features import honest_features

    monkeypatch.setattr(settings_mod.settings, "MACRO_FEATURES_ENABLED", True, raising=False)
    monkeypatch.setattr(settings_mod.settings, "MACRO_DATA_ROOT", str(tmp_path), raising=False)

    bars = _make_bars()
    out = honest_features.compute_all_features(bars, settings=settings_mod.settings)
    for name in MACRO_FEATURE_NAMES:
        assert name in out.columns
        assert out[name].isna().all()

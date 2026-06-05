"""Tests for engine.ticker_exposure.TickerExposureTracker."""
from datetime import datetime, timezone

from engine.ticker_exposure import TickerExposureTracker

T = "KXBTC15M-26MAR252045-45"


def _tracker(tmp_path, max_c=10, max_n=1500):
    return TickerExposureTracker(
        state_path=tmp_path / "ticker_exposure.json",
        max_contracts_per_ticker=max_c,
        max_notional_cents_per_ticker=max_n,
    )


def test_allows_under_caps(tmp_path):
    tr = _tracker(tmp_path)
    ok, _ = tr.check(T, contracts=3, notional_cents=135)
    assert ok


def test_blocks_when_contracts_exceed(tmp_path):
    tr = _tracker(tmp_path, max_c=5)
    ok, reason = tr.check(T, contracts=6, notional_cents=10)
    assert not ok and "contracts" in reason


def test_blocks_when_notional_exceeds(tmp_path):
    tr = _tracker(tmp_path, max_n=1500)
    ok, reason = tr.check(T, contracts=1, notional_cents=1600)
    assert not ok and "notional" in reason


def test_accumulates_across_orders_and_blocks(tmp_path):
    """The 26MAR25 scenario: repeated adds on one ticker must hit the cap."""
    tr = _tracker(tmp_path, max_c=10, max_n=1500)
    tr.record(T, contracts=5, notional_cents=900)
    # A second order that would push past the notional cap is rejected.
    ok, reason = tr.check(T, contracts=5, notional_cents=900)
    assert not ok and "notional" in reason
    # A small top-up under the cap is still allowed.
    ok2, _ = tr.check(T, contracts=2, notional_cents=300)
    assert ok2


def test_persists_across_restart(tmp_path):
    tr = _tracker(tmp_path)
    tr.record(T, contracts=4, notional_cents=600)
    # New tracker instance (simulates a process restart) reloads exposure.
    tr2 = _tracker(tmp_path)
    assert tr2.open_for(T)["contracts"] == 4
    assert tr2.open_for(T)["notional_cents"] == 600


def test_stale_day_resets(tmp_path):
    path = tmp_path / "ticker_exposure.json"
    path.write_text('{"utc_date": "2000-01-01", "open": {"X": {"contracts": 9, "notional_cents": 1400}}}')
    tr = TickerExposureTracker(path, 10, 1500)
    # Yesterday's exposure must not carry into today.
    assert tr.open_for("X")["contracts"] == 0


def test_release_clears_ticker(tmp_path):
    tr = _tracker(tmp_path)
    tr.record(T, contracts=4, notional_cents=600)
    tr.release(T)
    assert tr.open_for(T)["contracts"] == 0


def test_unwritable_path_degrades_gracefully(tmp_path):
    # A path whose parent cannot be created should not crash record().
    tr = TickerExposureTracker("/proc/nonexistent/cannot/write.json", 10, 1500)
    tr.record(T, contracts=3, notional_cents=100)  # must not raise
    assert tr.open_for(T)["contracts"] == 3  # still tracked in memory

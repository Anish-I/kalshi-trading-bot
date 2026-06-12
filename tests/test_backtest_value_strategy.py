"""Tests for the value-strategy calibration backtest harness."""
from scripts.backtest_value_strategy import (
    run_calibration_backtest,
    simulate_strategy_pnl,
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


def test_strategy_profits_only_when_market_lags():
    bars = synthetic_bars(n_minutes=8000, seed=5)
    # Efficient market (no lag): the strategy should find ~no edge.
    eff = simulate_strategy_pnl(bars, lag_min=0, half_spread_cents=2, min_edge_cents=2.0)
    # Lagging market: positive expected edge per trade.
    lag = simulate_strategy_pnl(bars, lag_min=4, half_spread_cents=2, min_edge_cents=2.0)
    assert lag["trades"] > 100
    assert lag["avg_pnl_cents"] > 0
    # The lagging market must be strictly more profitable than the efficient one.
    eff_avg = eff.get("avg_pnl_cents", 0.0) if eff.get("trades", 0) else 0.0
    assert lag["avg_pnl_cents"] > eff_avg


def test_perfect_predictions_have_zero_brier():
    p = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    m = _calibration_metrics(p, y)
    assert m["brier"] == 0.0
    assert m["directional_accuracy"] == 1.0


# ---------------- regression: timing + strike alignment fixes ---------------- #

import pandas as pd
from scripts.backtest_value_strategy import _t_remaining, _strike_for_window, WINDOW_MIN


def test_t_remaining_accounts_for_bar_close_timing():
    # Deciding at the close of bar `elapsed` means the price is at minute
    # elapsed+1, so T = 15 - 1 - elapsed. The old code said 15 - elapsed.
    assert _t_remaining(1) == 13
    assert _t_remaining(12) == 2
    assert _t_remaining(WINDOW_MIN - 1) == 0


def test_strike_is_window_open_not_first_bar_close():
    open_ = np.array([100.0, 101.0])
    close = np.array([101.0, 102.0])
    assert _strike_for_window(open_, close, 0) == 100.0   # uses bar open
    assert _strike_for_window(None, close, 0) == 101.0    # fallback w/o open col


def test_synthetic_bars_have_open_column():
    sb = synthetic_bars(n_minutes=100, seed=1)
    assert "open" in sb.columns
    # bar open == previous bar close (no gaps in a continuous GBM path)
    assert np.allclose(sb["open"].to_numpy()[1:], sb["close"].to_numpy()[:-1])


def test_realized_outcome_uses_window_open_strike():
    # 15 flat bars except the window opens at 100 and everything closes at 99.5:
    # vs open-strike (100) the window resolves NO. The old close-of-first-bar
    # strike (99.5) would have called it YES (close >= strike).
    ts = pd.date_range("2026-01-01", periods=15, freq="min", tz="UTC")
    # tiny wiggle so trailing vol is nonzero (all-flat closes -> sigma=0 -> skip)
    closes = [99.5 + 0.01 * (i % 2) for i in range(15)]
    bars = pd.DataFrame({
        "timestamp": ts,
        "open": [100.0] + closes[:-1],
        "close": closes,
    })
    m = run_calibration_backtest(bars, vol_scale=1.0)
    # spot (99.5) < strike (100) with ~zero vol -> model predicts ~0 = realized.
    assert m["n"] > 0
    assert m["brier"] < 0.01


# ------------------- one-entry-per-window honest replay ------------------- #

from scripts.backtest_value_strategy import simulate_one_trade_per_window


def test_one_trade_per_window_books_at_most_one_entry_per_window():
    bars = synthetic_bars(n_minutes=8000, seed=5)
    n_windows = len(bars.set_index("timestamp").resample("15min").first())
    r = simulate_one_trade_per_window(bars, lag_min=4, vol_scale=1.0)
    assert 0 < r["trades"] <= n_windows


def test_one_trade_per_window_efficient_book_finds_nothing():
    bars = synthetic_bars(n_minutes=8000, seed=5)
    r = simulate_one_trade_per_window(bars, lag_min=0, vol_scale=1.0)
    assert r["trades"] == 0


def test_one_trade_per_window_lagging_book_is_profitable():
    bars = synthetic_bars(n_minutes=8000, seed=5)
    r = simulate_one_trade_per_window(bars, lag_min=4, vol_scale=1.0)
    assert r["trades"] > 50
    assert r["avg_pnl_cents"] > 0
    assert "max_drawdown_cents" in r

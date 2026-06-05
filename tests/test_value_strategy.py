"""Tests for engine.value_strategy.evaluate_market (pure decision glue)."""
from datetime import datetime, timezone, timedelta

from engine.value_strategy import evaluate_market

NOW = datetime(2026, 6, 5, 20, 20, tzinfo=timezone.utc)


def _market(close_minutes_from_now=5, floor_strike=60000.0, yes_ask=0.90, no_ask=0.11):
    close = NOW + timedelta(minutes=close_minutes_from_now)
    return {
        "ticker": "KXBTC15M-26JUN051630-30",
        "close_time": close.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "floor_strike": floor_strike,
        "strike_type": "greater_or_equal",
        "yes_ask_dollars": yes_ask,
        "no_ask_dollars": no_ask,
    }


def test_outside_window_no_trade():
    m = _market(close_minutes_from_now=20)  # >13 min
    ev = evaluate_market(m, spot=60300, sigma_per_min=0.001, now=NOW)
    assert ev.decision is None and "window" in ev.reason


def test_no_strike_no_trade():
    m = _market()
    m.pop("floor_strike")
    m["yes_sub_title"] = "Target price: TBD"
    ev = evaluate_market(m, spot=60300, sigma_per_min=0.001, now=NOW)
    assert ev.decision is None and "strike" in ev.reason


def test_value_buy_yes_when_spot_above_strike_and_underpriced():
    # spot well above strike, little time -> fair P(YES) ~0.97 but YES asks 0.90.
    m = _market(close_minutes_from_now=3, floor_strike=60000, yes_ask=0.90, no_ask=0.11)
    ev = evaluate_market(m, spot=60300, sigma_per_min=0.001, now=NOW, min_edge_cents=1.0)
    assert ev.fair_p > 0.9
    assert ev.decision is not None and ev.decision.side == "yes"


def test_value_buy_no_when_spot_below_strike_and_no_underpriced():
    # spot below strike -> fair P(YES) low; NO underpriced -> buy NO.
    m = _market(close_minutes_from_now=3, floor_strike=60000, yes_ask=0.12, no_ask=0.88)
    ev = evaluate_market(m, spot=59700, sigma_per_min=0.001, now=NOW, min_edge_cents=1.0)
    assert ev.fair_p < 0.1
    assert ev.decision is not None and ev.decision.side == "no"


def test_fairly_priced_no_trade():
    # spot == strike -> fair 0.5; symmetric asks 0.50/0.50 -> fee kills edge.
    m = _market(close_minutes_from_now=7, floor_strike=60000, yes_ask=0.50, no_ask=0.50)
    ev = evaluate_market(m, spot=60000, sigma_per_min=0.001, now=NOW, min_edge_cents=1.0)
    # Reached the decision step but no side clears the edge threshold.
    assert ev.decision is not None and ev.decision.side is None


def test_missing_vol_no_trade():
    m = _market(close_minutes_from_now=5)
    ev = evaluate_market(m, spot=60300, sigma_per_min=0.0, now=NOW)
    assert ev.decision is None and "vol" in ev.reason

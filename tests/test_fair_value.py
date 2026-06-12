"""Tests for engine.fair_value and engine.value_decision."""
import math

import pytest

from engine.fair_value import (
    normal_cdf,
    extract_strike,
    sigma_per_min_from_returns,
    prob_yes,
    fair_value_from_market,
)
from engine.value_decision import decide_value_trade


# ----------------------------- fair_value ------------------------------ #

def test_normal_cdf_known_points():
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(1.96) == pytest.approx(0.975, abs=0.001)
    assert normal_cdf(-1.96) == pytest.approx(0.025, abs=0.001)


def test_extract_strike_from_floor_strike():
    assert extract_strike({"floor_strike": 60775.15}) == pytest.approx(60775.15)


def test_extract_strike_from_subtitle():
    m = {"yes_sub_title": "Target Price: $60,775.15"}
    assert extract_strike(m) == pytest.approx(60775.15)


def test_extract_strike_missing():
    assert extract_strike({"yes_sub_title": "Target price: TBD"}) is None


def test_at_the_money_is_half():
    assert prob_yes(60000, 60000, minutes_to_close=10, sigma_per_min=0.001) == pytest.approx(0.5)


def test_above_strike_more_likely_yes():
    # spot above strike with little time/vol left -> high P(YES)
    p = prob_yes(60300, 60000, minutes_to_close=2, sigma_per_min=0.001)
    assert p > 0.9


def test_below_strike_less_likely_yes():
    p = prob_yes(59700, 60000, minutes_to_close=2, sigma_per_min=0.001)
    assert p < 0.1


def test_zero_time_is_deterministic():
    assert prob_yes(60001, 60000, 0, 0.001) == 1.0
    assert prob_yes(59999, 60000, 0, 0.001) == 0.0


def test_zero_vol_is_deterministic():
    assert prob_yes(60001, 60000, 5, 0.0) == 1.0
    assert prob_yes(59999, 60000, 5, 0.0) == 0.0


def test_more_time_pulls_toward_half():
    # Same spot/strike gap, more time + vol -> closer to 0.5 (less certain).
    near = prob_yes(60100, 60000, minutes_to_close=1, sigma_per_min=0.001)
    far = prob_yes(60100, 60000, minutes_to_close=14, sigma_per_min=0.001)
    assert near > far > 0.5


def test_less_or_equal_strike_inverts():
    p_geq = prob_yes(60300, 60000, 2, 0.001, strike_type="greater_or_equal")
    p_leq = prob_yes(60300, 60000, 2, 0.001, strike_type="less_or_equal")
    assert p_leq == pytest.approx(1.0 - p_geq)


def test_sigma_from_returns():
    rets = [0.001, -0.001, 0.002, -0.0015, 0.0008] * 4
    s = sigma_per_min_from_returns(rets, window=15)
    assert s is not None and s > 0


def test_sigma_insufficient_data():
    assert sigma_per_min_from_returns([0.001]) is None


def test_fair_value_from_market():
    market = {"floor_strike": 60000, "strike_type": "greater_or_equal"}
    p = fair_value_from_market(market, spot=60300, minutes_to_close=2, sigma_per_min=0.001)
    assert p is not None and p > 0.9


def test_fair_value_from_market_no_strike():
    assert fair_value_from_market({"yes_sub_title": "TBD"}, 60000, 5, 0.001) is None


# --------------------------- value_decision ---------------------------- #

def test_buys_underpriced_yes():
    # fair P(YES)=0.97 but YES asks only 90c -> strong value buy on YES.
    d = decide_value_trade(fair_p=0.97, yes_ask_cents=90, no_ask_cents=11)
    assert d.side == "yes"
    assert d.net_ev_cents > 1.0


def test_buys_underpriced_no():
    # fair P(YES)=0.05 -> P(NO)=0.95; NO asks 88c -> value buy on NO.
    d = decide_value_trade(fair_p=0.05, yes_ask_cents=13, no_ask_cents=88)
    assert d.side == "no"
    assert d.net_ev_cents > 1.0


def test_passes_when_fairly_priced():
    # fair 0.60, YES asks 60c -> edge ~ -fee, below threshold -> no trade.
    d = decide_value_trade(fair_p=0.60, yes_ask_cents=60, no_ask_cents=41)
    assert d.side is None


def test_respects_min_edge():
    d = decide_value_trade(fair_p=0.55, yes_ask_cents=52, no_ask_cents=49, min_edge_cents=5.0)
    assert d.side is None  # ~1c edge < 5c threshold


def test_respects_max_entry_price():
    # Deep ITM: fair 0.99 but YES asks 98c (above max_entry 95) -> skip YES,
    # NO side has no edge either.
    d = decide_value_trade(fair_p=0.99, yes_ask_cents=98, no_ask_cents=3, max_entry_cents=95)
    assert d.side != "yes"


def test_fee_makes_thin_edge_unprofitable():
    # 2c gross edge but taker fee near mid (~2c) wipes it out.
    d = decide_value_trade(fair_p=0.52, yes_ask_cents=50, no_ask_cents=50, min_edge_cents=1.0)
    assert d.side is None


# ------------------------- ewma / calibrated sigma ------------------------- #

from engine.fair_value import (  # noqa: E402
    ewma_sigma_series,
    ewma_sigma_per_min,
    calibrated_sigma_per_min,
    BTC15M_VOL_SCALE,
)


def test_ewma_sigma_constant_returns_recovers_magnitude():
    rets = [0.001] * 200
    s = ewma_sigma_per_min(rets)
    assert s == pytest.approx(0.001, rel=1e-3)


def test_ewma_sigma_series_no_lookahead_and_decay():
    # A single spike then quiet: sigma jumps at the spike and decays after.
    rets = [0.0005] * 50 + [0.01] + [0.0005] * 50
    arr = ewma_sigma_series(rets)
    spike = arr[50]
    assert spike > arr[49]          # reacts at the spike, not before
    assert arr[60] < spike          # decays afterwards
    assert arr[49] == pytest.approx(0.0005, rel=1e-2)


def test_ewma_sigma_handles_nans():
    rets = [float("nan"), 0.001, -0.001, 0.001, -0.001]
    s = ewma_sigma_per_min(rets)
    assert s is not None and s > 0


def test_calibrated_sigma_applies_scale():
    rets = [0.001] * 200
    raw = ewma_sigma_per_min(rets)
    cal = calibrated_sigma_per_min(rets)
    assert cal == pytest.approx(raw * BTC15M_VOL_SCALE)


def test_ewma_sigma_insufficient_data():
    assert ewma_sigma_per_min([0.001]) is None

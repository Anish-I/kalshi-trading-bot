"""Tests for the Kalshi price-dependent fee model (engine/fees.py)."""
import pytest

from engine.fees import (
    kalshi_fee_cents,
    round_trip_fee_cents,
    pair_fee_cents,
    multiplier_for_family,
    DEFAULT_MULTIPLIER,
    SP500_MULTIPLIER,
)


# Raw taker fee (dollars) = 0.07 * P * (1-P); Kalshi rounds the order total UP to
# the cent, so a single contract always pays at least 1c at any tradeable price.
@pytest.mark.parametrize(
    "price_cents,expected_taker",
    [
        (5, 1.0),   # raw 0.33c -> ceil to 1c
        (45, 2.0),  # raw 1.73c -> ceil to 2c
        (50, 2.0),  # raw 1.75c (peak) -> ceil to 2c
        (95, 1.0),  # raw 0.33c -> ceil to 1c
    ],
)
def test_taker_fee_single_contract(price_cents, expected_taker):
    assert kalshi_fee_cents(price_cents) == expected_taker


def test_maker_is_quarter_of_taker_before_rounding():
    # 10 contracts so per-order rounding doesn't swamp the 25% relationship.
    taker = kalshi_fee_cents(50, contracts=10, maker=False)   # ceil(17.5) = 18
    maker = kalshi_fee_cents(50, contracts=10, maker=True)    # ceil(4.375) = 5
    assert taker == 18.0
    assert maker == 5.0
    assert maker < taker  # maker is materially cheaper


def test_fee_rounds_up_over_whole_order_not_per_contract():
    # 10 @ 50c taker = ceil(0.07*10*0.25 *100) = ceil(17.5) = 18c, i.e. 1.8c/contract,
    # NOT 10 * ceil(1.75) = 20c.
    assert kalshi_fee_cents(50, contracts=10) == 18.0


def test_degenerate_prices_are_free():
    assert kalshi_fee_cents(0) == 0.0
    assert kalshi_fee_cents(100) == 0.0
    assert kalshi_fee_cents(50, contracts=0) == 0.0


def test_sp500_multiplier_is_half():
    full = kalshi_fee_cents(50, contracts=100, multiplier=DEFAULT_MULTIPLIER)  # ceil(175)=175
    half = kalshi_fee_cents(50, contracts=100, multiplier=SP500_MULTIPLIER)    # ceil(87.5)=88
    assert full == 175.0
    assert half == 88.0


def test_round_trip_equals_entry_fee_settlement_is_free():
    assert round_trip_fee_cents(45) == kalshi_fee_cents(45)


def test_pair_fee_defaults_to_maker_both_legs():
    # yes@40 maker: ceil(0.07*0.4*0.6*0.25*100)=ceil(0.42)=1
    # no@50  maker: ceil(0.07*0.5*0.5*0.25*100)=ceil(0.4375)=1
    assert pair_fee_cents(40, 50, contracts=1) == 2.0
    # As takers the same pair costs more per leg.
    taker_pair = pair_fee_cents(40, 50, contracts=1, maker=False)
    assert taker_pair >= 2.0


def test_family_multiplier_lookup():
    assert multiplier_for_family("btc_15m") == DEFAULT_MULTIPLIER  # crypto defaults to 0.07
    assert multiplier_for_family("spx_5m") == SP500_MULTIPLIER
    assert multiplier_for_family("weather") == DEFAULT_MULTIPLIER
    assert multiplier_for_family(None) == DEFAULT_MULTIPLIER

"""Tests for pair pricing calculations with correct orderbook interpretation.

Kalshi orderbook: yes_dollars/no_dollars are BIDS (ascending, best=last).
Implied asks: YES_ask = 100 - best_NO_bid, NO_ask = 100 - best_YES_bid.
"""
import pytest
from engine.pair_pricing import (
    pair_cost_cents,
    pair_gross_profit_cents,
    pair_net_profit_cents,
    is_pair_profitable,
    extract_book_from_orderbook,
    evaluate_pair_opportunity,
)


def test_pair_cost():
    assert pair_cost_cents(73, 29) == 102  # crossing both asks costs > $1
    assert pair_cost_cents(40, 50) == 90


def test_pair_gross_profit():
    assert pair_gross_profit_cents(73, 29) == -2  # taker arb = negative
    assert pair_gross_profit_cents(40, 50) == 10  # only possible as maker


def test_pair_net_profit():
    net = pair_net_profit_cents(40, 50)
    assert net == pytest.approx(10 - 2.14, abs=0.01)


def test_is_profitable():
    assert is_pair_profitable(40, 50, min_net_cents=1.0)  # maker scenario
    assert not is_pair_profitable(49, 50, min_net_cents=1.0)  # 1c gross, negative net


def test_extract_book_bids_are_ascending():
    """Kalshi bids: ascending order, best bid = last element."""
    ob = {
        "orderbook_fp": {
            "yes_dollars": [["0.6700", "100"], ["0.6800", "200"], ["0.7100", "50"]],
            "no_dollars": [["0.2300", "100"], ["0.2500", "200"], ["0.2700", "50"]],
        }
    }
    book = extract_book_from_orderbook(ob)

    # Best bids are LAST (highest)
    assert book["best_yes_bid"] == 71  # 0.71 * 100
    assert book["best_no_bid"] == 27   # 0.27 * 100

    # Implied asks
    assert book["implied_yes_ask"] == 73  # 100 - 27
    assert book["implied_no_ask"] == 29   # 100 - 71

    # Taker pair cost >= 100c (no taker arb)
    assert book["taker_pair_cost"] == 102  # 73 + 29

    # Maker prices: post at best_bid + 1
    assert book["maker_yes_price"] == 72  # 71 + 1
    assert book["maker_no_price"] == 28   # 27 + 1
    assert book["maker_pair_cost"] == 100  # 72 + 28


def test_taker_arb_never_profitable():
    """Crossing both implied asks should never produce profit."""
    ob = {
        "orderbook_fp": {
            "yes_dollars": [["0.6700", "100"], ["0.7100", "50"]],
            "no_dollars": [["0.2300", "100"], ["0.2700", "50"]],
        }
    }
    opp = evaluate_pair_opportunity(ob, pair_cap_cents=96)
    assert opp["taker_arb"] is False
    assert opp["taker_pair_cost"] >= 100


def test_maker_opportunity():
    """Maker opportunity: post limits at best_bid+1 on both sides."""
    ob = {
        "orderbook_fp": {
            # Wide spread: best_yes_bid=50, best_no_bid=40
            "yes_dollars": [["0.4500", "100"], ["0.5000", "50"]],
            "no_dollars": [["0.3500", "100"], ["0.4000", "50"]],
        }
    }
    opp = evaluate_pair_opportunity(ob, pair_cap_cents=96)

    # Maker prices: 51 + 41 = 92c
    assert opp["maker_yes_price"] == 51
    assert opp["maker_no_price"] == 41
    assert opp["maker_pair_cost"] == 92
    assert opp["maker_tradeable"] is True
    assert opp["maker_net"] > 0


def test_min_maker_net_threshold_rejects_marginal():
    """min_maker_net=5.0 should reject opportunities tradable at default 2.5."""
    ob = {
        "orderbook_fp": {
            # best_yes_bid=50, best_no_bid=44 -> maker 51+45=96, gross=4, net~1.86
            "yes_dollars": [["0.4500", "100"], ["0.5000", "50"]],
            "no_dollars": [["0.4000", "100"], ["0.4400", "50"]],
        }
    }
    opp_default = evaluate_pair_opportunity(ob, pair_cap_cents=96)
    opp_strict = evaluate_pair_opportunity(ob, pair_cap_cents=96, min_maker_net=5.0)
    # Default threshold 2.5: not tradable (net ~1.86 < 2.5 anyway). Pick a case
    # where default=True and strict=False.

    ob2 = {
        "orderbook_fp": {
            # best_yes_bid=50, best_no_bid=42 -> maker 51+43=94, gross=6, net~3.86
            "yes_dollars": [["0.4500", "100"], ["0.5000", "50"]],
            "no_dollars": [["0.4000", "100"], ["0.4200", "50"]],
        }
    }
    opp_default2 = evaluate_pair_opportunity(ob2, pair_cap_cents=96)
    opp_strict2 = evaluate_pair_opportunity(ob2, pair_cap_cents=96, min_maker_net=5.0)
    assert opp_default2["maker_tradeable"] is True  # net ~3.86 >= 2.5
    assert opp_strict2["maker_tradeable"] is False  # net ~3.86 < 5.0


def test_tight_market_no_maker_opportunity():
    """Tight market: maker pair cost too high."""
    ob = {
        "orderbook_fp": {
            # Tight: best_yes_bid=49, best_no_bid=49
            "yes_dollars": [["0.4800", "100"], ["0.4900", "50"]],
            "no_dollars": [["0.4800", "100"], ["0.4900", "50"]],
        }
    }
    opp = evaluate_pair_opportunity(ob, pair_cap_cents=96)
    # Maker: 50 + 50 = 100c — too expensive
    assert opp["maker_pair_cost"] == 100
    assert opp["maker_tradeable"] is False

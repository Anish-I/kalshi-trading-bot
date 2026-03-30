"""Tests for pair pricing calculations."""
import pytest
from engine.pair_pricing import (
    pair_cost_cents,
    pair_gross_profit_cents,
    pair_net_profit_cents,
    is_pair_profitable,
    extract_top_of_book,
    evaluate_pair_opportunity,
)


def test_pair_cost():
    assert pair_cost_cents(40, 50) == 90
    assert pair_cost_cents(50, 50) == 100


def test_pair_gross_profit():
    assert pair_gross_profit_cents(40, 50) == 10
    assert pair_gross_profit_cents(50, 50) == 0
    assert pair_gross_profit_cents(30, 60) == 10


def test_pair_net_profit():
    net = pair_net_profit_cents(40, 50)
    assert net == pytest.approx(10 - 2.14, abs=0.01)


def test_is_profitable():
    assert is_pair_profitable(40, 50, min_net_cents=1.0)  # 10c gross, ~7.86c net
    assert not is_pair_profitable(49, 50, min_net_cents=1.0)  # 1c gross, negative net


def test_extract_top_of_book():
    ob = {
        "orderbook_fp": {
            "yes_dollars": [["0.0340", "1400.00"], ["0.0360", "78.00"]],
            "no_dollars": [["0.9530", "1656.90"], ["0.9540", "400.00"]],
        }
    }
    ya, yq, na, nq = extract_top_of_book(ob)
    assert ya == 3  # 0.034 * 100 = 3.4 → 3
    assert yq == 1400
    assert na == 95  # 0.953 * 100 = 95.3 → 95
    assert nq == 1656


def test_evaluate_opportunity():
    ob = {
        "orderbook_fp": {
            "yes_dollars": [["0.40", "100"], ["0.42", "50"]],
            "no_dollars": [["0.50", "200"], ["0.52", "100"]],
        }
    }
    opp = evaluate_pair_opportunity(ob, pair_cap_cents=96)
    assert opp["yes_ask_cents"] == 40
    assert opp["no_ask_cents"] == 50
    assert opp["pair_cost_cents"] == 90
    assert opp["gross_profit_cents"] == 10
    assert opp["tradeable"] is True
    assert opp["max_pairs"] == 100  # min(100, 200)


def test_evaluate_not_tradeable():
    ob = {
        "orderbook_fp": {
            "yes_dollars": [["0.55", "100"]],
            "no_dollars": [["0.50", "100"]],
        }
    }
    opp = evaluate_pair_opportunity(ob, pair_cap_cents=96)
    assert opp["pair_cost_cents"] == 105
    assert opp["tradeable"] is False

"""Tests for orphan-leg flatten logic in crypto_combined_trader.

We don't import the trader module (it has side effects: ProcessLock,
argparse). Instead we exercise the flatten algorithm as a small
standalone helper that mirrors the logic in the trader, and verify:

1. Flatten fires a sell-market order when slippage is within cap.
2. Slippage exceeding cap triggers hold-to-settle fallback.
3. Realized loss uses actual proceeds, not cost basis.
"""
from unittest.mock import MagicMock

from engine.pair_pricing import extract_book_from_orderbook

FLATTEN_MAX_SLIPPAGE_CENTS = 5


def flatten_orphan(client, ticker, filled_side, filled_qty, filled_price):
    """Mirror of the trader's orphan flatten block, extracted for testing."""
    ob = client.get_orderbook(ticker, depth=1)
    book = extract_book_from_orderbook(ob)
    exit_bid = book["best_yes_bid"] if filled_side == "yes" else book["best_no_bid"]

    if exit_bid <= 0:
        return {"action": "hold_to_settle", "reason": "no_exit_liquidity"}

    slippage = filled_price - exit_bid
    if slippage > FLATTEN_MAX_SLIPPAGE_CENTS:
        return {
            "action": "hold_to_settle",
            "reason": "flatten_slippage_too_wide",
            "entry": filled_price,
            "exit": exit_bid,
        }

    client.place_order(
        ticker=ticker, side=filled_side, action="sell",
        count=filled_qty, order_type="market",
    )
    proceeds = exit_bid * filled_qty
    realized_loss = (filled_price * filled_qty) - proceeds
    return {
        "action": "flattened",
        "entry": filled_price,
        "exit": exit_bid,
        "proceeds_cents": proceeds,
        "realized_loss_cents": realized_loss,
    }


def _book(best_yes_bid_c, best_no_bid_c):
    return {
        "orderbook_fp": {
            "yes_dollars": [[f"{best_yes_bid_c/100:.4f}", "100"]],
            "no_dollars": [[f"{best_no_bid_c/100:.4f}", "100"]],
        }
    }


def test_flatten_within_slippage_cap_issues_market_sell():
    client = MagicMock()
    # YES filled at 50c, best YES bid now 48c -> slippage 2c, within 5c cap
    client.get_orderbook.return_value = _book(48, 50)
    client.place_order.return_value = {"order": {"order_id": "X1"}}

    result = flatten_orphan(client, "KXBTC-25-T60000", "yes", filled_qty=10, filled_price=50)

    client.place_order.assert_called_once_with(
        ticker="KXBTC-25-T60000", side="yes", action="sell",
        count=10, order_type="market",
    )
    assert result["action"] == "flattened"
    # Cost basis: 50 * 10 = 500; proceeds: 48 * 10 = 480; realized loss: 20c
    assert result["proceeds_cents"] == 480
    assert result["realized_loss_cents"] == 20


def test_flatten_slippage_too_wide_holds_to_settle():
    client = MagicMock()
    # YES filled at 55c, best YES bid now 45c -> slippage 10c > cap 5c
    client.get_orderbook.return_value = _book(45, 50)

    result = flatten_orphan(client, "KXETH", "yes", filled_qty=5, filled_price=55)

    client.place_order.assert_not_called()
    assert result["action"] == "hold_to_settle"
    assert result["reason"] == "flatten_slippage_too_wide"


def test_flatten_realized_loss_is_not_cost_basis():
    """Realized loss must reflect actual proceeds, not the full cost basis."""
    client = MagicMock()
    # NO filled at 40c, best NO bid now 38c -> slippage 2c
    client.get_orderbook.return_value = _book(55, 38)
    client.place_order.return_value = {"order": {"order_id": "X2"}}

    result = flatten_orphan(client, "KXSOL", "no", filled_qty=20, filled_price=40)

    cost_basis = 40 * 20  # 800
    assert result["realized_loss_cents"] < cost_basis
    # Actually realized = (40-38)*20 = 40c
    assert result["realized_loss_cents"] == 40
    assert result["proceeds_cents"] == 760


def test_flatten_no_exit_liquidity_holds():
    client = MagicMock()
    client.get_orderbook.return_value = {
        "orderbook_fp": {"yes_dollars": [], "no_dollars": [["0.4000", "10"]]}
    }
    result = flatten_orphan(client, "KXXRP", "yes", filled_qty=3, filled_price=50)
    client.place_order.assert_not_called()
    assert result["action"] == "hold_to_settle"

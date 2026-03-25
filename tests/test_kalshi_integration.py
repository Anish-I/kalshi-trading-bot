"""
Integration tests against the live Kalshi API.

These tests verify real order lifecycle behavior:
- Auth works
- Markets can be fetched
- Orders placed at unfillable prices stay resting
- Resting orders can be canceled
- Settlements can be read
- Positions reflect reality

Run: pytest tests/test_kalshi_integration.py -v -s
(Uses real API — will place and cancel a 1-contract order at 1c)
"""
import sys
import time

import pytest

sys.path.insert(0, ".")

from kalshi.client import KalshiClient
from kalshi.market_discovery import KXBTCMarketTracker


@pytest.fixture(scope="module")
def client():
    return KalshiClient()


@pytest.fixture(scope="module")
def tracker(client):
    return KXBTCMarketTracker(client)


class TestAuth:
    def test_balance_returns_float(self, client):
        bal = client.get_balance()
        assert isinstance(bal, float)
        assert bal >= 0

    def test_positions_returns_list(self, client):
        result = client._request("GET", "/portfolio/positions", params={"limit": 5})
        assert "market_positions" in result or "positions" in result


class TestMarketDiscovery:
    def test_active_markets_found(self, tracker):
        markets = tracker.get_active_markets()
        # BTC 15-min markets run 24/7 so there should always be at least one
        assert len(markets) >= 1

    def test_market_has_required_fields(self, tracker):
        markets = tracker.get_active_markets()
        m = markets[0]
        assert "ticker" in m
        assert "yes_bid_dollars" in m or "yes_ask_dollars" in m
        assert "close_time" in m

    def test_time_remaining_positive(self, tracker):
        nxt = tracker.get_next_market()
        if nxt:
            remaining = tracker.get_market_time_remaining(nxt)
            assert remaining > 0


class TestOrderLifecycle:
    """Place a 1c limit order (won't fill), verify resting, then cancel."""

    def test_place_resting_cancel(self, client, tracker):
        nxt = tracker.get_next_market()
        if not nxt:
            pytest.skip("No active market")

        ticker = nxt["ticker"]
        remaining = tracker.get_market_time_remaining(nxt)
        if remaining < 60:
            pytest.skip("Market closing too soon")

        # Place YES at 1c — guaranteed not to fill
        order = client.place_order(
            ticker=ticker, side="yes", action="buy",
            count=1, order_type="limit", yes_price=1,
        )
        order_data = order.get("order", order)
        status = order_data.get("status", "")
        order_id = order_data.get("order_id", "")

        assert status == "resting", f"Expected resting, got {status}"
        assert order_id, "No order_id returned"

        # Verify order exists via GET
        time.sleep(1)
        check = client._request("GET", f"/portfolio/orders/{order_id}")
        check_data = check.get("order", check)
        assert check_data.get("status") == "resting"

        # Cancel it
        client.cancel_order(order_id)
        time.sleep(1)

        # Verify canceled
        check2 = client._request("GET", f"/portfolio/orders/{order_id}")
        check2_data = check2.get("order", check2)
        assert check2_data.get("status") in ("canceled", "cancelled")


class TestSettlements:
    def test_settlements_readable(self, client):
        result = client._request("GET", "/portfolio/settlements", params={"limit": 5})
        settlements = result.get("settlements", [])
        # We've traded before so there should be settlements
        assert isinstance(settlements, list)
        if settlements:
            s = settlements[0]
            assert "ticker" in s
            assert "market_result" in s


class TestOrderbook:
    def test_orderbook_returns_levels(self, client, tracker):
        nxt = tracker.get_next_market()
        if not nxt:
            pytest.skip("No active market")

        ob = client.get_orderbook(nxt["ticker"], depth=5)
        ob_data = ob.get("orderbook_fp", ob.get("orderbook", {}))

        yes_levels = ob_data.get("yes_dollars", ob_data.get("yes", []))
        assert isinstance(yes_levels, list)
        # Should have at least 1 level
        if yes_levels:
            assert len(yes_levels[0]) == 2  # [price, qty]

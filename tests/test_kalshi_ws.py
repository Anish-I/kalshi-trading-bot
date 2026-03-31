from engine.kalshi_ws import KalshiWebSocketClient, derive_ws_url


def test_derive_ws_url_matches_demo_host():
    assert derive_ws_url("https://demo-api.kalshi.co/trade-api/v2") == "wss://demo-api.kalshi.co/trade-api/ws/v2"


def test_normalize_fill_message_converts_price_and_count():
    payload = {
        "type": "fill",
        "msg": {
            "order_id": "ord_yes",
            "market_ticker": "KXBTC-TEST",
            "side": "yes",
            "action": "buy",
            "yes_price_dollars": "0.35",
            "count_fp": "1.00",
            "trade_id": "trade-1",
        },
    }

    normalized = KalshiWebSocketClient.normalize_fill_message(payload)

    assert normalized["order_id"] == "ord_yes"
    assert normalized["ticker"] == "KXBTC-TEST"
    assert normalized["side"] == "yes"
    assert normalized["price_cents"] == 35
    assert normalized["count"] == 1


def test_normalize_user_order_message_marks_executed_fill():
    payload = {
        "type": "user_order",
        "msg": {
            "order_id": "ord_no",
            "market_ticker": "KXBTC-TEST",
            "side": "no",
            "status": "executed",
            "no_price_dollars": "0.61",
            "fill_count_fp": "0.00",
            "remaining_count_fp": "0.00",
            "initial_count_fp": "1.00",
        },
    }

    normalized = KalshiWebSocketClient.normalize_user_order_message(payload)

    assert normalized["order_id"] == "ord_no"
    assert normalized["ticker"] == "KXBTC-TEST"
    assert normalized["side"] == "no"
    assert normalized["status"] == "executed"
    assert normalized["price_cents"] == 61
    assert normalized["fill_count"] == 1
    assert normalized["remaining_count"] == 0
    assert normalized["initial_count"] == 1

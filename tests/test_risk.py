"""Tests for the RiskManager."""

from engine.risk import RiskManager


def test_can_trade_initially():
    rm = RiskManager()
    allowed, reason = rm.can_trade()
    assert allowed is True
    assert reason == "ok"


def test_daily_loss_limit():
    rm = RiskManager(daily_loss_limit_cents=1000)
    # Record losses totalling exactly the limit
    for _ in range(10):
        rm.record_trade(-100, {"ticker": "TEST"})
    allowed, reason = rm.can_trade()
    assert allowed is False
    assert "daily loss limit" in reason


def test_consecutive_loss_halt():
    rm = RiskManager(consecutive_loss_halt=3)
    for _ in range(3):
        rm.record_trade(-50, {"ticker": "TEST"})
    allowed, reason = rm.can_trade()
    assert allowed is False
    assert "consecutive loss halt" in reason


def test_consecutive_losses_reset_on_win():
    rm = RiskManager(consecutive_loss_halt=3)
    rm.record_trade(-50, {"ticker": "TEST"})
    rm.record_trade(-50, {"ticker": "TEST"})
    # A win should reset the streak
    rm.record_trade(100, {"ticker": "TEST"})
    allowed, _ = rm.can_trade()
    assert allowed is True
    assert rm.consecutive_losses == 0


def test_kill_switch():
    rm = RiskManager()
    rm.kill()
    allowed, reason = rm.can_trade()
    assert allowed is False
    assert "kill switch" in reason

    rm.resume()
    allowed, reason = rm.can_trade()
    assert allowed is True
    assert reason == "ok"


def test_order_size_check():
    rm = RiskManager(max_contracts=10)
    allowed, reason = rm.check_order_size(10)
    assert allowed is True

    allowed, reason = rm.check_order_size(11)
    assert allowed is False
    assert "exceeds max" in reason


def test_order_size_under_limit():
    rm = RiskManager(max_contracts=10)
    allowed, reason = rm.check_order_size(5)
    assert allowed is True
    assert reason == "ok"

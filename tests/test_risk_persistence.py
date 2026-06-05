"""Tests for RiskManager restart-safe persistence (engine/risk.py)."""
import json

from engine.risk import RiskManager


def test_in_memory_when_no_state_path():
    rm = RiskManager(daily_loss_limit_cents=5000)
    rm.record_trade(-100, {"t": "x"})
    assert rm.daily_pnl_cents == -100
    # No file is written when state_path is None (purely in-memory).


def test_daily_loss_survives_restart(tmp_path):
    path = tmp_path / "risk_state.json"
    rm = RiskManager(daily_loss_limit_cents=500, state_path=path)
    rm.record_trade(-300, {"t": "loss1"})
    rm.record_trade(-300, {"t": "loss2"})
    assert rm.daily_pnl_cents == -600
    # Restart: a fresh manager must reload the daily loss and stay halted.
    rm2 = RiskManager(daily_loss_limit_cents=500, state_path=path)
    assert rm2.daily_pnl_cents == -600
    allowed, reason = rm2.can_trade()
    assert allowed is False and "daily loss" in reason.lower()


def test_consecutive_losses_survive_restart(tmp_path):
    path = tmp_path / "risk_state.json"
    rm = RiskManager(consecutive_loss_halt=3, state_path=path)
    rm.record_trade(-10, {})
    rm.record_trade(-10, {})
    rm2 = RiskManager(consecutive_loss_halt=3, state_path=path)
    assert rm2.consecutive_losses == 2
    rm2.record_trade(-10, {})
    allowed, reason = rm2.can_trade()
    assert allowed is False and "consecutive" in reason.lower()


def test_stale_day_state_is_discarded(tmp_path):
    path = tmp_path / "risk_state.json"
    path.write_text(json.dumps({"utc_date": "2000-01-01", "daily_pnl_cents": -9999, "consecutive_losses": 9}))
    rm = RiskManager(daily_loss_limit_cents=500, state_path=path)
    # Yesterday's loss must not carry into today.
    assert rm.daily_pnl_cents == 0
    assert rm.consecutive_losses == 0


def test_win_resets_consecutive_and_persists(tmp_path):
    path = tmp_path / "risk_state.json"
    rm = RiskManager(state_path=path)
    rm.record_trade(-10, {})
    rm.record_trade(+50, {})  # a win resets the streak
    assert rm.consecutive_losses == 0
    rm2 = RiskManager(state_path=path)
    assert rm2.consecutive_losses == 0

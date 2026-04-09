"""Tests for engine.gate_risk_adapter."""
from engine.gate_risk_adapter import RiskManagerAdapter, FamilyLimitsAdapter


def test_risk_adapter_ok():
    a = RiskManagerAdapter(lambda: (True, "ok"), max_contracts=10)
    assert a.can_trade() == (True, "ok")
    ok, detail = a.check_order_size(5)
    assert ok and detail == "ok"


def test_risk_adapter_can_trade_false():
    a = RiskManagerAdapter(lambda: (False, "trailing_stop:halted"), max_contracts=10)
    ok, detail = a.can_trade()
    assert not ok
    assert "trailing_stop" in detail


def test_risk_adapter_oversize():
    a = RiskManagerAdapter(lambda: (True, "ok"), max_contracts=3)
    ok, detail = a.check_order_size(5)
    assert not ok
    assert "5" in detail and "3" in detail


def test_risk_adapter_nonpositive_size():
    a = RiskManagerAdapter(lambda: (True, "ok"), max_contracts=3)
    ok, detail = a.check_order_size(0)
    assert not ok and "<= 0" in detail


def test_risk_adapter_fails_closed_on_exception():
    def _boom():
        raise RuntimeError("x")

    a = RiskManagerAdapter(_boom, max_contracts=10)
    ok, detail = a.can_trade()
    assert not ok
    assert "risk_adapter_error" in detail


def test_family_adapter_ok():
    a = FamilyLimitsAdapter(lambda _f: 100, lambda _f: 1000)
    ok, detail = a.can_enter("fam", 200)
    assert ok and detail == "ok"


def test_family_adapter_over_budget():
    a = FamilyLimitsAdapter(lambda _f: 800, lambda _f: 1000)
    ok, detail = a.can_enter("fam", 300)
    assert not ok
    assert "family_budget" in detail


def test_family_adapter_fails_closed_on_exception():
    def _boom(_f):
        raise RuntimeError("x")

    a = FamilyLimitsAdapter(_boom, lambda _f: 1000)
    ok, detail = a.can_enter("fam", 100)
    assert not ok
    assert "family_adapter_error" in detail

"""Tests for engine.pre_trade_gate.PreTradeGate composition."""

import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, ".")

from engine.pre_trade_gate import GateContext, GateDecision, PreTradeGate


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _make_ctx(**overrides) -> GateContext:
    base = dict(
        ticker="KXBTC15M-26APR0800-T70000",
        family="btc_15m",
        side="yes",
        entry_cents=45,
        contracts=5,
        yes_ask=0.45,
        no_ask=0.56,
        yes_bid=0.44,
        no_bid=0.55,
        model_prob=0.65,  # raw edge 0.20 -> well above min_edge_after_fees_pct
        quote_age_s=1.0,
        max_stale_s=120.0,
        session_tag="us_core",
        strategy_tag="ml",
        calibration_artifact=None,
        min_trades=30,
        ev_buffer_cents=0.0,
        min_net_ev_cents=0.0,
    )
    base.update(overrides)
    return GateContext(**base)


def _ok_risk():
    m = MagicMock()
    m.can_trade.return_value = (True, "ok")
    m.check_order_size.return_value = (True, "ok")
    return m


def _ok_family():
    m = MagicMock()
    m.can_enter.return_value = (True, "ok")
    return m


def _good_calibration_artifact(p_win: float = 0.65, tradable: bool = True, n: int = 60) -> dict:
    """Build a calibration dict shaped exactly like load_crypto_calibration() output."""
    row = {
        "side": "yes",
        "price_bucket": "45c",
        "win_rate": p_win,
        "n_trades": n,
        "tradable": tradable,
    }
    return {
        "exists": True,
        "version": "test-artifact",
        "bucket_size_cents": 5,
        "rows": [row],
        "rows_by_key": {("yes", "45c"): row, ("yes", 45): row},
        "metadata": {},
        "summary": {},
    }


# ---------------------------------------------------------------------- #
# Tests — one per reason_code
# ---------------------------------------------------------------------- #


def test_all_pass_happy_path():
    gate = PreTradeGate(_ok_risk(), _ok_family())
    ctx = _make_ctx(calibration_artifact=_good_calibration_artifact())
    decision = gate.evaluate(ctx)
    assert isinstance(decision, GateDecision)
    assert decision.allowed is True
    assert decision.reason_code == "ok"
    assert decision.contracts_adjusted == ctx.contracts


def test_risk_halt_short_circuits():
    risk = _ok_risk()
    risk.can_trade.return_value = (False, "daily_loss_limit")
    family = _ok_family()
    gate = PreTradeGate(risk, family)
    decision = gate.evaluate(_make_ctx())

    assert decision.allowed is False
    assert decision.reason_code == "risk_halt"
    assert "risk_halt" in decision.checks
    # Later checks must NOT have been attempted.
    assert "order_size" not in decision.checks
    assert "family_budget" not in decision.checks
    family.can_enter.assert_not_called()
    risk.check_order_size.assert_not_called()


def test_order_size_blocks():
    risk = _ok_risk()
    risk.check_order_size.return_value = (False, "too big")
    gate = PreTradeGate(risk, _ok_family())
    decision = gate.evaluate(_make_ctx(contracts=999))
    assert decision.allowed is False
    assert decision.reason_code == "order_size"


def test_family_budget_blocks():
    family = _ok_family()
    family.can_enter.return_value = (False, "over_budget")
    gate = PreTradeGate(_ok_risk(), family)
    decision = gate.evaluate(_make_ctx())
    assert decision.allowed is False
    assert decision.reason_code == "family_budget"
    assert "family_budget" in decision.checks
    assert decision.checks["family_budget"]["passed"] is False


def test_stale_quote_blocks():
    gate = PreTradeGate(_ok_risk(), _ok_family())
    decision = gate.evaluate(
        _make_ctx(quote_age_s=999.0, max_stale_s=120.0)
    )
    assert decision.allowed is False
    assert decision.reason_code == "quote_quality"
    assert "stale" in decision.checks["quote_quality"]["detail"].lower()


def test_session_blocked():
    gate = PreTradeGate(_ok_risk(), _ok_family(), allowed_sessions={"us_core"})
    decision = gate.evaluate(_make_ctx(session_tag="overnight"))
    assert decision.allowed is False
    assert decision.reason_code == "session_blocked"


def test_bucket_not_tradable():
    gate = PreTradeGate(_ok_risk(), _ok_family())
    artifact = _good_calibration_artifact(tradable=False)
    decision = gate.evaluate(_make_ctx(calibration_artifact=artifact))
    assert decision.allowed is False
    assert decision.reason_code == "bucket_not_tradable"


def test_negative_ev_blocks():
    gate = PreTradeGate(_ok_risk(), _ok_family())
    # p_win = 0.10 at entry 45c -> gross EV very negative
    artifact = _good_calibration_artifact(p_win=0.10, tradable=True)
    decision = gate.evaluate(_make_ctx(calibration_artifact=artifact))
    assert decision.allowed is False
    assert decision.reason_code == "negative_ev"


def test_checks_dict_complete_on_pass():
    gate = PreTradeGate(_ok_risk(), _ok_family())
    decision = gate.evaluate(_make_ctx(calibration_artifact=_good_calibration_artifact()))
    assert decision.allowed is True
    expected = {
        "risk_halt",
        "order_size",
        "family_budget",
        "quote_quality",
        "session",
        "calibration",
        "family_scorecard",
    }
    assert expected.issubset(set(decision.checks.keys()))
    for name in expected:
        entry = decision.checks[name]
        assert "passed" in entry and "detail" in entry and "values" in entry
        assert entry["passed"] is True


def test_to_json_roundtrip():
    gate = PreTradeGate(_ok_risk(), _ok_family())
    decision = gate.evaluate(_make_ctx(calibration_artifact=_good_calibration_artifact()))
    js = decision.to_json()
    assert js["allowed"] is True
    assert js["reason_code"] == "ok"
    assert "checks" in js and isinstance(js["checks"], dict)

"""Regression: the 26MAR25 BTC_15M catastrophe cannot recur.

On 2026-03-26 two BTC_15M NO positions lost -7374c and -3686c (-$110 combined),
dominating a -$102 week. The -7374c loss at a ~45c entry implies ~164 contracts
accumulated on a single ticker — far beyond any sane per-trade size — because
risk state was in-memory (reset on restart) and nothing capped per-ticker
exposure.

This test drives the REAL TickerExposureTracker through the REAL PreTradeGate and
asserts a single ticker can never accumulate near the catastrophic notional.
"""
from unittest.mock import MagicMock

from engine.pre_trade_gate import GateContext, PreTradeGate
from engine.ticker_exposure import TickerExposureTracker

TICKER = "KXBTC15M-26MAR252045-45"
ENTRY_CENTS = 45
PER_ORDER_CONTRACTS = 10  # ML_MAX_CONTRACTS in the combined trader

# Production caps (config/settings.py defaults).
MAX_CONTRACTS_PER_TICKER = 10
MAX_NOTIONAL_CENTS_PER_TICKER = 1500  # $15


def _permissive_gate(tracker):
    risk = MagicMock()
    risk.can_trade.return_value = (True, "ok")
    # order_size only caps a SINGLE order; the danger is accumulation, so let it pass.
    risk.check_order_size.return_value = (True, "ok")
    family = MagicMock()
    family.can_enter.return_value = (True, "ok")
    return PreTradeGate(risk, family, ticker_cap=tracker)


def _ctx(contracts):
    return GateContext(
        ticker=TICKER, family="btc_15m", side="no",
        entry_cents=ENTRY_CENTS, contracts=contracts,
        yes_ask=0.55, no_ask=0.45, yes_bid=0.54, no_bid=0.44,
        # model_prob low -> big NO-side edge, so quote_quality passes and the
        # ONLY control that can block accumulation is the per-ticker cap.
        model_prob=0.30, quote_age_s=1.0, max_stale_s=60.0,
        session_tag="us_core", strategy_tag="combined_ml",
        calibration_artifact=None,  # sim: not required
    )


def test_single_ticker_exposure_is_hard_capped(tmp_path):
    tracker = TickerExposureTracker(
        tmp_path / "exp.json", MAX_CONTRACTS_PER_TICKER, MAX_NOTIONAL_CENTS_PER_TICKER
    )
    gate = _permissive_gate(tracker)

    placed_contracts = 0
    placed_notional = 0
    blocked = False
    # Try to place 200 contracts worth (the catastrophe was ~164) in 10-lots.
    for _ in range(20):
        decision = gate.evaluate(_ctx(PER_ORDER_CONTRACTS))
        if not decision.allowed:
            assert decision.reason_code == "ticker_cap"
            blocked = True
            break
        # Mirror the trader: record exposure only after a successful gate pass.
        tracker.record(TICKER, PER_ORDER_CONTRACTS, ENTRY_CENTS * PER_ORDER_CONTRACTS)
        placed_contracts += PER_ORDER_CONTRACTS
        placed_notional += ENTRY_CENTS * PER_ORDER_CONTRACTS

    assert blocked, "ticker cap never engaged — exposure was unbounded"
    # The realized exposure stays a tiny fraction of the -7374c catastrophe.
    assert placed_notional <= MAX_NOTIONAL_CENTS_PER_TICKER
    assert placed_contracts <= MAX_CONTRACTS_PER_TICKER
    assert placed_notional < 1500  # << 7374c that actually occurred


def test_cap_survives_simulated_restart(tmp_path):
    """Even if the process restarts mid-day, accumulated exposure is remembered."""
    path = tmp_path / "exp.json"
    t1 = TickerExposureTracker(path, MAX_CONTRACTS_PER_TICKER, MAX_NOTIONAL_CENTS_PER_TICKER)
    t1.record(TICKER, 10, ENTRY_CENTS * 10)  # 450c used

    # Restart: new tracker reloads, gate must keep enforcing against prior exposure.
    t2 = TickerExposureTracker(path, MAX_CONTRACTS_PER_TICKER, MAX_NOTIONAL_CENTS_PER_TICKER)
    gate = _permissive_gate(t2)
    # 10 already open (= contract cap). Any further order is blocked.
    decision = gate.evaluate(_ctx(PER_ORDER_CONTRACTS))
    assert decision.allowed is False
    assert decision.reason_code == "ticker_cap"

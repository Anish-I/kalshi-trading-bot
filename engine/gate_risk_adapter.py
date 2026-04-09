"""Adapters that let PreTradeGate see the trader's actual live risk state.

The 3 crypto traders use different risk objects:
  - combined_trader: PairRiskManager + TrailingStopLoss + inline daily_pnl tracking
  - ml_trader: inline daily_pnl globals
  - pair_trader: PairRiskManager

Rather than unify them tonight (larger refactor), wrap each in a thin adapter
that exposes RiskManager + FamilyLimits interfaces.
"""
from __future__ import annotations
from typing import Callable


class RiskManagerAdapter:
    """Adapter exposing RiskManager's public interface over arbitrary live state."""

    def __init__(
        self,
        can_trade_fn: Callable[[], tuple[bool, str]],
        max_contracts: int,
    ):
        self._can_trade_fn = can_trade_fn
        self._max_contracts = int(max_contracts)

    def can_trade(self) -> tuple[bool, str]:
        try:
            return self._can_trade_fn()
        except Exception as e:
            # Fail closed: if we can't compute risk, refuse the trade.
            return False, f"risk_adapter_error:{type(e).__name__}"

    def check_order_size(self, contracts: int) -> tuple[bool, str]:
        if int(contracts) > self._max_contracts:
            return False, f"size {contracts} > cap {self._max_contracts}"
        if int(contracts) <= 0:
            return False, "size <= 0"
        return True, "ok"


class FamilyLimitsAdapter:
    """Adapter exposing FamilyLimits.can_enter over arbitrary exposure state."""

    def __init__(
        self,
        exposure_fn: Callable[[str], int],  # (family) -> current cents
        budget_fn: Callable[[str], int],    # (family) -> budget cents
    ):
        self._exposure_fn = exposure_fn
        self._budget_fn = budget_fn

    def can_enter(self, family_name: str, proposed_cents: int) -> tuple[bool, str]:
        try:
            exposure = int(self._exposure_fn(family_name))
            budget = int(self._budget_fn(family_name))
        except Exception as e:
            return False, f"family_adapter_error:{type(e).__name__}"
        if exposure + int(proposed_cents) > budget:
            return (
                False,
                f"family_budget exposure={exposure}+proposed={proposed_cents} > budget={budget}",
            )
        return True, "ok"

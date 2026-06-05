"""Glue: turn a live market dict + spot + vol into a value decision.

Pure (no network) so it can be unit-tested. The trader script handles I/O and
funnels the result through PreTradeGate for risk/ticker-cap/quote/session checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from engine.fair_value import extract_strike, prob_yes
from engine.value_decision import decide_value_trade, ValueDecision
from engine.fees import multiplier_for_family

# Same window as the ML trader: trade only with 2-13 minutes to close.
MIN_REMAINING_S = 120
MAX_REMAINING_S = 780


@dataclass
class MarketEval:
    decision: ValueDecision | None
    remaining_s: float
    fair_p: float | None
    strike: float | None
    reason: str


def _parse_close_time(market: dict) -> datetime | None:
    raw = market.get("close_time")
    if not raw:
        return None
    if isinstance(raw, str) and raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _ask_cents(market: dict, side: str) -> int:
    key_d = f"{side}_ask_dollars"
    key = f"{side}_ask"
    val = market.get(key_d, market.get(key, 0)) or 0
    return int(round(float(val) * 100))


def evaluate_market(
    market: dict,
    spot: float,
    sigma_per_min: float,
    *,
    now: datetime | None = None,
    min_edge_cents: float = 1.0,
    max_entry_cents: int = 95,
    family: str = "btc_15m",
    min_remaining_s: int = MIN_REMAINING_S,
    max_remaining_s: int = MAX_REMAINING_S,
) -> MarketEval:
    """Decide whether (and which side) to value-bet on a single market."""
    now = now or datetime.now(timezone.utc)
    close_dt = _parse_close_time(market)
    if close_dt is None:
        return MarketEval(None, 0.0, None, None, "no/invalid close_time")
    remaining_s = (close_dt - now).total_seconds()
    if remaining_s < min_remaining_s or remaining_s > max_remaining_s:
        return MarketEval(None, remaining_s, None, None, "outside trade window")

    strike = extract_strike(market)
    if strike is None:
        return MarketEval(None, remaining_s, None, None, "no strike on market")
    if spot is None or spot <= 0 or sigma_per_min is None or sigma_per_min <= 0:
        return MarketEval(None, remaining_s, None, strike, "missing spot/vol")

    minutes = remaining_s / 60.0
    fair_p = prob_yes(
        spot, strike, minutes, sigma_per_min,
        strike_type=str(market.get("strike_type", "greater_or_equal")),
    )
    yes_c = _ask_cents(market, "yes")
    no_c = _ask_cents(market, "no")
    decision = decide_value_trade(
        fair_p, yes_c, no_c,
        min_edge_cents=min_edge_cents,
        multiplier=multiplier_for_family(family),
        max_entry_cents=max_entry_cents,
    )
    return MarketEval(decision, remaining_s, fair_p, strike, decision.reason)

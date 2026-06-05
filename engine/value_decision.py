"""Value decision: bet whichever side is cheap relative to fair value.

This is the one-sided version of the user's "buy the cheap side" instinct. Given
a fair P(YES) and the current asks, compute the net expected value (after real
Kalshi taker fees) of buying YES vs NO, and take the better side if its edge
clears a threshold. Never buys both sides (taker pair cost is always >= $1).
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.fees import kalshi_fee_cents, DEFAULT_MULTIPLIER


@dataclass
class ValueDecision:
    side: str | None            # "yes" | "no" | None (no trade)
    entry_cents: int            # ask price of the chosen side
    fair_p: float               # model P(YES)
    net_ev_cents: float         # expected value per contract after fees
    reason: str


def _net_ev_cents(win_prob: float, ask_cents: int, multiplier: float) -> float:
    """EV per contract of buying a side at ``ask_cents`` with P(win)=win_prob.

    Payout is 100c on win, 0 on loss; the only fee is the taker entry
    (settlement is fee-free on Kalshi).
    """
    if ask_cents <= 0 or ask_cents >= 100:
        return float("-inf")
    fee = kalshi_fee_cents(ask_cents, contracts=1, maker=False, multiplier=multiplier)
    return win_prob * (100 - ask_cents) - (1.0 - win_prob) * ask_cents - fee


def decide_value_trade(
    fair_p: float,
    yes_ask_cents: int,
    no_ask_cents: int,
    *,
    min_edge_cents: float = 1.0,
    multiplier: float = DEFAULT_MULTIPLIER,
    max_entry_cents: int = 95,
) -> ValueDecision:
    """Pick the side with the higher net EV, if it clears ``min_edge_cents``.

    Args:
        fair_p: model probability that YES resolves.
        yes_ask_cents / no_ask_cents: current asks in cents.
        min_edge_cents: minimum net EV (cents/contract) required to trade.
        multiplier: fee multiplier for the market family (engine.fees).
        max_entry_cents: never pay above this (avoid thin deep-ITM longshots).
    """
    yes_ev = _net_ev_cents(fair_p, yes_ask_cents, multiplier) if 0 < yes_ask_cents <= max_entry_cents else float("-inf")
    no_ev = _net_ev_cents(1.0 - fair_p, no_ask_cents, multiplier) if 0 < no_ask_cents <= max_entry_cents else float("-inf")

    if yes_ev == float("-inf") and no_ev == float("-inf"):
        return ValueDecision(None, 0, fair_p, 0.0, "no usable quotes")

    if yes_ev >= no_ev:
        side, ask, ev, win_p = "yes", yes_ask_cents, yes_ev, fair_p
    else:
        side, ask, ev, win_p = "no", no_ask_cents, no_ev, 1.0 - fair_p

    if ev < min_edge_cents:
        return ValueDecision(
            None, int(ask), fair_p, ev,
            f"best side {side} net EV {ev:.2f}c < min {min_edge_cents:.2f}c",
        )
    return ValueDecision(
        side, int(ask), fair_p, ev,
        f"{side} cheap vs fair: P(win)={win_p:.3f} ask={ask}c net_ev={ev:.2f}c",
    )

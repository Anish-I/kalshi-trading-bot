"""Pair pricing — implied asks, fee calculation, pair cost evaluation.

On Kalshi, the orderbook shows YES bids and NO bids.
A YES ask at price X means someone is offering to sell YES at X cents.
A NO ask at price Y means someone is offering to sell NO at Y cents.
Buying both: total cost = yes_ask + no_ask. Payout = $1.00 always.
Profit = 100c - yes_ask_cents - no_ask_cents - fees.
"""

import logging

logger = logging.getLogger(__name__)

# Kalshi fees (approximate, per contract per side)
FEE_PER_CONTRACT_PER_SIDE_CENTS = 1.07
ROUND_TRIP_FEE_CENTS = FEE_PER_CONTRACT_PER_SIDE_CENTS * 2  # ~2.14c

# For pairs: we pay fee on YES buy + NO buy = 2 sides
PAIR_FEE_CENTS = FEE_PER_CONTRACT_PER_SIDE_CENTS * 2  # ~2.14c total


def pair_cost_cents(yes_ask_cents: int, no_ask_cents: int) -> int:
    """Total cost to buy one YES + one NO contract."""
    return yes_ask_cents + no_ask_cents


def pair_gross_profit_cents(yes_ask_cents: int, no_ask_cents: int) -> int:
    """Gross profit per pair (before fees). Payout is always 100c."""
    return 100 - yes_ask_cents - no_ask_cents


def pair_net_profit_cents(yes_ask_cents: int, no_ask_cents: int) -> float:
    """Net profit per pair after fees."""
    return pair_gross_profit_cents(yes_ask_cents, no_ask_cents) - PAIR_FEE_CENTS


def is_pair_profitable(yes_ask_cents: int, no_ask_cents: int, min_net_cents: float = 1.0) -> bool:
    """Check if buying both sides is profitable after fees."""
    return pair_net_profit_cents(yes_ask_cents, no_ask_cents) >= min_net_cents


def extract_top_of_book(orderbook_response: dict) -> tuple[int, int, int, int]:
    """Extract best YES ask and NO ask from Kalshi orderbook response.

    Returns (yes_ask_cents, yes_qty, no_ask_cents, no_qty).
    """
    ob = orderbook_response.get("orderbook_fp", orderbook_response.get("orderbook", {}))
    yes_levels = ob.get("yes_dollars", ob.get("yes", []))
    no_levels = ob.get("no_dollars", ob.get("no", []))

    yes_ask_cents = int(float(yes_levels[0][0]) * 100) if yes_levels else 0
    yes_qty = int(float(yes_levels[0][1])) if yes_levels else 0
    no_ask_cents = int(float(no_levels[0][0]) * 100) if no_levels else 0
    no_qty = int(float(no_levels[0][1])) if no_levels else 0

    return yes_ask_cents, yes_qty, no_ask_cents, no_qty


def evaluate_pair_opportunity(orderbook_response: dict, pair_cap_cents: int = 96) -> dict:
    """Evaluate a pair opportunity from an orderbook snapshot.

    Args:
        orderbook_response: Raw Kalshi orderbook response.
        pair_cap_cents: Maximum total cost for the pair.

    Returns:
        Dict with opportunity details.
    """
    yes_cents, yes_qty, no_cents, no_qty = extract_top_of_book(orderbook_response)

    total = yes_cents + no_cents
    gross = 100 - total
    net = gross - PAIR_FEE_CENTS
    tradeable = total > 0 and total <= pair_cap_cents and net > 0
    qty = min(yes_qty, no_qty) if tradeable else 0

    return {
        "yes_ask_cents": yes_cents,
        "yes_qty": yes_qty,
        "no_ask_cents": no_cents,
        "no_qty": no_qty,
        "pair_cost_cents": total,
        "gross_profit_cents": gross,
        "net_profit_cents": round(net, 2),
        "tradeable": tradeable,
        "max_pairs": qty,
        "pair_cap_cents": pair_cap_cents,
    }

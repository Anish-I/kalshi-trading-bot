"""Pair pricing — implied asks from bids, fee calculation, pair cost evaluation.

Kalshi orderbook structure:
  yes_dollars: list of [price, qty] — YES BIDS (resting buy orders), ascending
  no_dollars: list of [price, qty] — NO BIDS (resting buy orders), ascending

Best bid = LAST element (highest price).
Implied ask: YES_ask = 100 - best_NO_bid, NO_ask = 100 - best_YES_bid.

Crossing both implied asks as taker always costs >= $1 (no taker arb).
Pair strategy requires MAKER on at least one leg.
"""

import logging

logger = logging.getLogger(__name__)

# Kalshi fees (approximate — actual fees are variable by price/size)
FEE_PER_CONTRACT_PER_SIDE_CENTS = 1.07
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


def extract_book_from_orderbook(orderbook_response: dict) -> dict:
    """Extract bids and implied asks from Kalshi orderbook response.

    Kalshi's yes_dollars/no_dollars are BIDS (ascending order, best=last).
    Implied asks: YES_ask = 100 - best_NO_bid, NO_ask = 100 - best_YES_bid.

    Returns dict with all book info.
    """
    ob = orderbook_response.get("orderbook_fp", orderbook_response.get("orderbook", {}))
    yes_bids = ob.get("yes_dollars", ob.get("yes", []))
    no_bids = ob.get("no_dollars", ob.get("no", []))

    # Best bids are LAST element (ascending sort, highest = best)
    best_yes_bid_cents = int(float(yes_bids[-1][0]) * 100) if yes_bids else 0
    best_yes_bid_qty = int(float(yes_bids[-1][1])) if yes_bids else 0
    best_no_bid_cents = int(float(no_bids[-1][0]) * 100) if no_bids else 0
    best_no_bid_qty = int(float(no_bids[-1][1])) if no_bids else 0

    # Implied asks (what it costs to BUY)
    # Only valid when BOTH sides have bids; one-sided books can't produce valid asks
    if best_no_bid_cents > 0 and best_yes_bid_cents > 0:
        implied_yes_ask = 100 - best_no_bid_cents
        implied_no_ask = 100 - best_yes_bid_cents
    else:
        implied_yes_ask = 0
        implied_no_ask = 0

    return {
        "best_yes_bid": best_yes_bid_cents,
        "best_yes_bid_qty": best_yes_bid_qty,
        "best_no_bid": best_no_bid_cents,
        "best_no_bid_qty": best_no_bid_qty,
        "implied_yes_ask": implied_yes_ask,
        "implied_no_ask": implied_no_ask,
        # Taker pair cost (always >= 100c — no taker arb)
        "taker_pair_cost": implied_yes_ask + implied_no_ask,
        # Maker opportunity: post limit below implied ask
        # YES maker at best_yes_bid+1, NO maker at best_no_bid+1
        "maker_yes_price": best_yes_bid_cents + 1,
        "maker_no_price": best_no_bid_cents + 1,
        "maker_pair_cost": (best_yes_bid_cents + 1) + (best_no_bid_cents + 1),
        # Spread (should be >= 0, = 0 means tight market)
        "spread_cents": (implied_yes_ask + implied_no_ask) - 100,
        "yes_bids_depth": len(yes_bids),
        "no_bids_depth": len(no_bids),
    }


def evaluate_pair_opportunity(orderbook_response: dict, pair_cap_cents: int = 95) -> dict:
    """Evaluate a pair opportunity from an orderbook snapshot.

    Since taker pair cost is always >= 100c, we evaluate MAKER opportunities:
    post limits at best_bid + 1 on both sides.

    Args:
        orderbook_response: Raw Kalshi orderbook response.
        pair_cap_cents: Maximum total cost for the pair.

    Returns:
        Dict with opportunity details.
    """
    book = extract_book_from_orderbook(orderbook_response)

    taker_cost = book["taker_pair_cost"]
    maker_cost = book["maker_pair_cost"]
    maker_gross = 100 - maker_cost
    maker_net = maker_gross - PAIR_FEE_CENTS

    # Taker arb (crossing both asks) — should never be profitable
    taker_gross = 100 - taker_cost
    taker_arb = taker_cost > 0 and taker_cost <= pair_cap_cents and taker_gross > PAIR_FEE_CENTS

    # Maker opportunity (posting limits)
    maker_tradeable = maker_cost > 0 and maker_cost <= pair_cap_cents and maker_net >= 2.5

    return {
        # Taker (crossing asks)
        "taker_pair_cost": taker_cost,
        "taker_gross": taker_gross,
        "taker_arb": taker_arb,
        # Maker (posting limits)
        "maker_yes_price": book["maker_yes_price"],
        "maker_no_price": book["maker_no_price"],
        "maker_pair_cost": maker_cost,
        "maker_gross": maker_gross,
        "maker_net": round(maker_net, 2),
        "maker_tradeable": maker_tradeable,
        # Book info
        "best_yes_bid": book["best_yes_bid"],
        "best_no_bid": book["best_no_bid"],
        "implied_yes_ask": book["implied_yes_ask"],
        "implied_no_ask": book["implied_no_ask"],
        "spread_cents": book["spread_cents"],
        "pair_cap_cents": pair_cap_cents,
        # Legacy compat
        "yes_ask_cents": book["implied_yes_ask"],
        "no_ask_cents": book["implied_no_ask"],
        "pair_cost_cents": taker_cost,
        "gross_profit_cents": taker_gross,
        "net_profit_cents": round(taker_gross - PAIR_FEE_CENTS, 2),
        "tradeable": maker_tradeable,
        "max_pairs": min(book["best_yes_bid_qty"], book["best_no_bid_qty"]),
    }

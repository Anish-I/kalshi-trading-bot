"""Quote quality guards — stale check, min edge after fees, max spread.

Shared across all market families to prevent trading on bad quotes.
"""

import logging
import time

logger = logging.getLogger(__name__)

# Kalshi fee: ~1.07c per contract per side (varies, using conservative estimate)
KALSHI_FEE_CENTS_PER_SIDE = 1.07
ROUND_TRIP_FEE_CENTS = KALSHI_FEE_CENTS_PER_SIDE * 2


def check_quote_quality(
    yes_ask: float,
    no_ask: float,
    yes_bid: float = 0.0,
    no_bid: float = 0.0,
    max_spread_cents: int = 20,
    min_edge_after_fees_pct: float = 0.02,
    model_prob: float = 0.5,
    side: str = "yes",
    quote_age_s: float = 0.0,
    max_stale_s: float = 120.0,
) -> tuple[bool, str]:
    """Check if a quote is fresh and tradeable.

    Args:
        yes_ask, no_ask: Current ask prices (as probability 0-1).
        yes_bid, no_bid: Current bid prices.
        max_spread_cents: Maximum acceptable spread in cents.
        min_edge_after_fees_pct: Minimum edge after fees as fraction.
        model_prob: Our model's probability estimate for YES.
        side: Which side we'd trade ("yes" or "no").
        quote_age_s: How old the quote is in seconds.
        max_stale_s: Maximum acceptable quote age.

    Returns:
        (tradeable, reason)
    """
    # Stale check
    if quote_age_s > max_stale_s:
        return False, f"stale quote: {quote_age_s:.0f}s > {max_stale_s:.0f}s"

    # No quotes available
    if yes_ask <= 0 and no_ask <= 0:
        return False, "no quotes"

    # Spread check
    if yes_ask > 0 and yes_bid > 0:
        spread = int((yes_ask - yes_bid) * 100)
        if spread > max_spread_cents:
            return False, f"spread too wide: {spread}c > {max_spread_cents}c"

    # Edge after fees
    if side == "yes" and yes_ask > 0:
        raw_edge = model_prob - yes_ask
        entry_cents = yes_ask * 100
        fee_drag = ROUND_TRIP_FEE_CENTS / entry_cents if entry_cents > 0 else 0
        net_edge = raw_edge - fee_drag
        if net_edge < min_edge_after_fees_pct:
            return False, f"edge after fees too small: {net_edge:.1%} < {min_edge_after_fees_pct:.1%}"
    elif side == "no" and no_ask > 0:
        raw_edge = (1.0 - model_prob) - no_ask
        entry_cents = no_ask * 100
        fee_drag = ROUND_TRIP_FEE_CENTS / entry_cents if entry_cents > 0 else 0
        net_edge = raw_edge - fee_drag
        if net_edge < min_edge_after_fees_pct:
            return False, f"edge after fees too small: {net_edge:.1%} < {min_edge_after_fees_pct:.1%}"

    return True, "ok"

"""Kalshi trading-fee model — price-dependent, maker/taker aware.

Replaces the old flat ``1.07c per contract per side`` placeholder that lived in
``pair_pricing.py`` and ``quote_guard.py``. That constant was wrong in both
directions: it overestimated maker fees at the extremes and *underestimated*
taker fees near 50c (real taker fee at 50c is ~1.75c, not 1.07c).

Kalshi fee schedule (effective Feb 2026, verified against published formula):

    taker fee for an order of C contracts at price P (dollars, 0.01-0.99):
        fee_dollars = ceil( multiplier * C * P * (1 - P) , 2 decimals )   # round UP to the cent
    maker fee = 25% of the taker fee.

Per-category multiplier:
    * default / most categories ........ 0.07   (peak 1.75c per contract at 50c)
    * S&P 500 / Nasdaq-100 ............. 0.035  (half-rate)
    * crypto (BTC / ETH) ............... 0.07   (see CRYPTO_MULTIPLIER note)

NOTE (crypto multiplier): Kalshi documents crypto as a "premium" category, but
the exact multiplier is not published in the machine-readable schedule and the
official PDF rate-limits automated fetches (HTTP 429). We default crypto to the
standard 0.07, which is the conservative (higher) end consistent with the quoted
"~1.75% at 50c" peak — overestimating fees is the safe direction for EV gating.
If a higher crypto multiplier is later confirmed, change CRYPTO_MULTIPLIER only.

Settlement is fee-free (winners and losers both settle at 0 fee); fees are only
charged on the executing trade, which is what these helpers model.
"""

from __future__ import annotations

import math

# Per-category fee multipliers. Change in ONE place if Kalshi updates the schedule.
DEFAULT_MULTIPLIER = 0.07
SP500_MULTIPLIER = 0.035
NASDAQ_MULTIPLIER = 0.035
CRYPTO_MULTIPLIER = 0.07  # TODO confirm vs official PDF when reachable (see module docstring)

# Maker fees are exactly 25% of the taker fee.
MAKER_FEE_FRACTION = 0.25

# Map a market "family"/series to its multiplier. Families come from
# engine/family_limits.py (btc_15m, eth_5m, spx_5m, weather, fed_cut, ...).
_FAMILY_MULTIPLIERS = {
    "btc_15m": CRYPTO_MULTIPLIER,
    "eth_5m": CRYPTO_MULTIPLIER,
    "spx_5m": SP500_MULTIPLIER,
}


def multiplier_for_family(family: str | None) -> float:
    """Return the fee multiplier for a market family, defaulting to 0.07."""
    if not family:
        return DEFAULT_MULTIPLIER
    return _FAMILY_MULTIPLIERS.get(family.lower(), DEFAULT_MULTIPLIER)


def kalshi_fee_cents(
    price_cents: int | float,
    contracts: int = 1,
    *,
    maker: bool = False,
    multiplier: float = DEFAULT_MULTIPLIER,
) -> float:
    """Kalshi trading fee, in cents, for one side of a trade.

    Args:
        price_cents: Execution price of the contract, 1-99 cents.
        contracts: Number of contracts in the order (fee rounds up over the order).
        maker: True for a resting/limit fill (25% of taker), False for taker.
        multiplier: Category multiplier (use ``multiplier_for_family``).

    Returns:
        Fee in cents, rounded UP to the whole cent (Kalshi rounds the order's
        total fee up). Returns 0.0 for degenerate prices (<=0 or >=100).
    """
    p = price_cents / 100.0
    if p <= 0.0 or p >= 1.0 or contracts <= 0:
        return 0.0

    rate = multiplier * (MAKER_FEE_FRACTION if maker else 1.0)
    fee_dollars = rate * contracts * p * (1.0 - p)
    # Round UP to the nearest cent over the whole order.
    fee_cents = math.ceil(round(fee_dollars, 6) * 100)
    return float(fee_cents)


def round_trip_fee_cents(
    entry_price_cents: int | float,
    contracts: int = 1,
    *,
    maker: bool = False,
    multiplier: float = DEFAULT_MULTIPLIER,
) -> float:
    """Total fee to enter and settle a one-sided position.

    Settlement (resolution to 0 or 100) is fee-free on Kalshi, so the round-trip
    fee for a directional bet is just the entry fee. Exposed as a named helper so
    callers don't accidentally double-count a settlement fee.
    """
    return kalshi_fee_cents(
        entry_price_cents, contracts, maker=maker, multiplier=multiplier
    )


def pair_fee_cents(
    yes_price_cents: int | float,
    no_price_cents: int | float,
    contracts: int = 1,
    *,
    maker: bool = True,
    multiplier: float = DEFAULT_MULTIPLIER,
) -> float:
    """Total fee to buy one YES + one NO leg (the 'both sides' pair).

    Both legs default to ``maker=True`` because the only viable pair strategy
    posts resting limit orders (taker pair cost is always >= $1; see
    pair_pricing.py). Fee is the sum of the two legs' fees.
    """
    return kalshi_fee_cents(
        yes_price_cents, contracts, maker=maker, multiplier=multiplier
    ) + kalshi_fee_cents(
        no_price_cents, contracts, maker=maker, multiplier=multiplier
    )

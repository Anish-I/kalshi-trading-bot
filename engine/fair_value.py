"""Fair-value model for Kalshi 15-minute crypto markets.

These markets ("BTC price up in next 15 mins?") resolve YES iff the settlement
price at close is >= the window's opening reference price. That reference is the
market's ``floor_strike`` (strike_type ``greater_or_equal``). So fair value is a
direct, computable probability — not a direction *forecast*:

    P(YES) = P(S_close >= K)
           = Phi( (ln(S/K) + drift*T) / (sigma * sqrt(T)) )

where
    S      = current spot,
    K      = floor_strike (the window's open price),
    T      = minutes to close,
    sigma  = per-minute log-return volatility,
    drift  = per-minute log drift (default 0 over a 15-minute horizon).

This is the edge the old ML bot missed: it bet "up/down" from momentum and never
compared spot to the actual strike. Most of the exploitable mispricing lives in
the final minutes, where S-vs-K is nearly decided but the book still lags.
"""

from __future__ import annotations

import math
import re

import numpy as np


def normal_cdf(x: float) -> float:
    """Standard normal CDF via erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def extract_strike(market: dict) -> float | None:
    """Get the strike (open reference price) from a Kalshi market dict.

    Prefers the numeric ``floor_strike`` field; falls back to parsing the
    ``yes_sub_title`` ("Target Price: $60,775.15").
    """
    fs = market.get("floor_strike")
    if fs is not None:
        try:
            return float(fs)
        except (TypeError, ValueError):
            pass
    sub = market.get("yes_sub_title") or market.get("subtitle") or ""
    m = re.search(r"\$?([\d,]+(?:\.\d+)?)", str(sub))
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def sigma_per_min_from_returns(ret_1m: "np.ndarray | list[float]", window: int = 15) -> float | None:
    """Per-minute log-return volatility = std of the last ``window`` 1m returns.

    Mirrors features.honest_features ``volatility_15m`` (std of ret_1m). Returns
    None if there isn't enough data.
    """
    arr = np.asarray([r for r in ret_1m if r is not None and not np.isnan(r)], dtype=float)
    if arr.size < 2:
        return None
    tail = arr[-window:] if arr.size >= window else arr
    if tail.size < 2:
        return None
    sigma = float(np.std(tail, ddof=1))
    return sigma if sigma > 0 else None


def prob_yes(
    spot: float,
    strike: float,
    minutes_to_close: float,
    sigma_per_min: float,
    *,
    drift_per_min: float = 0.0,
    strike_type: str = "greater_or_equal",
) -> float:
    """Fair P(YES) = P(settlement satisfies the strike) under a lognormal walk.

    Clamps to [0, 1]. Degenerate inputs (no time or no vol) collapse to the
    deterministic outcome implied by spot-vs-strike.
    """
    if spot <= 0 or strike <= 0:
        return 0.5  # cannot evaluate — neutral

    at_or_above = spot >= strike
    if minutes_to_close <= 0 or sigma_per_min <= 0:
        p_geq = 1.0 if at_or_above else 0.0
    else:
        z = (math.log(spot / strike) + drift_per_min * minutes_to_close) / (
            sigma_per_min * math.sqrt(minutes_to_close)
        )
        p_geq = normal_cdf(z)

    if strike_type in ("less", "less_or_equal", "less_than"):
        p = 1.0 - p_geq
    else:  # greater_or_equal (default for these markets)
        p = p_geq
    return max(0.0, min(1.0, p))


def fair_value_from_market(
    market: dict,
    spot: float,
    minutes_to_close: float,
    sigma_per_min: float,
    *,
    drift_per_min: float = 0.0,
) -> float | None:
    """Convenience: pull the strike from a market dict and compute P(YES).

    Returns None if the strike can't be determined.
    """
    strike = extract_strike(market)
    if strike is None:
        return None
    return prob_yes(
        spot,
        strike,
        minutes_to_close,
        sigma_per_min,
        drift_per_min=drift_per_min,
        strike_type=str(market.get("strike_type", "greater_or_equal")),
    )

"""Fed cut signal generator.

Compares Kalshi market-implied rate probabilities against our model
(initially just FRED current rate + threshold logic).

Strategy: Kalshi KXFED markets are threshold markets on the Fed funds
rate upper bound. E.g., KXFED-27APR-T4.25 asks "will the rate be above 4.25%?"

If the current rate IS 4.25%, then for the next meeting:
- T4.25 YES = rate stays or goes up (no cut)
- T4.25 NO = rate goes below 4.25% (at least one cut)
- T4.00 YES = rate stays above 4.00% (at most one cut)
- T4.00 NO = two or more cuts

The edge is: compare Kalshi implied prob vs CME FedWatch implied prob
(or our own estimate based on rate distance + meeting proximity).
"""

import logging
from datetime import date

from .data_source import get_fed_signal, CURRENT_RATE_UPPER

logger = logging.getLogger(__name__)


def evaluate_fed_markets(markets: list[dict], signal: dict) -> list[dict]:
    """Evaluate Fed rate markets for trading opportunities.

    Args:
        markets: List of Kalshi market dicts for KXFED series.
        signal: Output of get_fed_signal().

    Returns:
        List of opportunity dicts with edge, side, confidence.
    """
    current_rate = signal.get("current_rate", CURRENT_RATE_UPPER)
    days_to_fomc = signal.get("days_to_fomc")
    opportunities = []

    for mkt in markets:
        ticker = mkt.get("ticker", "")
        title = mkt.get("title", "")

        # Parse threshold from ticker: KXFED-27APR-T4.25 → 4.25
        threshold = _parse_threshold(ticker)
        if threshold is None:
            continue

        # Parse settlement date from ticker: KXFED-27APR → April 2027
        close_time = mkt.get("close_time", "")
        if close_time:
            try:
                from datetime import datetime as _dt
                settle_date = _dt.fromisoformat(close_time.replace("Z", "+00:00"))
                months_out = (settle_date.year - _dt.now().year) * 12 + (settle_date.month - _dt.now().month)
            except Exception:
                months_out = 0
        else:
            months_out = 0

        yes_ask = float(mkt.get("yes_ask", mkt.get("yes_ask_dollars", 0)) or 0)
        no_ask = float(mkt.get("no_ask", mkt.get("no_ask_dollars", 0)) or 0)

        if yes_ask <= 0 and no_ask <= 0:
            continue  # no quotes

        # Simple model: probability that rate stays above threshold
        model_prob_above = _model_prob_above_threshold(
            current_rate, threshold, days_to_fomc
        )

        # Edge calculation
        side = None
        edge = 0.0

        if model_prob_above > yes_ask and yes_ask > 0:
            side = "YES"
            edge = model_prob_above - yes_ask
        elif (1.0 - model_prob_above) > no_ask and no_ask > 0:
            side = "NO"
            edge = (1.0 - model_prob_above) - no_ask

        if side and edge > 0.03:  # minimum 3% edge
            opportunities.append({
                "ticker": ticker,
                "threshold": threshold,
                "current_rate": current_rate,
                "model_prob_above": round(model_prob_above, 4),
                "market_yes_ask": yes_ask,
                "market_no_ask": no_ask,
                "side": side,
                "edge": round(edge, 4),
                "days_to_fomc": days_to_fomc,
                "months_out": months_out,
                "confidence": _confidence_level(edge, days_to_fomc),
            })

    opportunities.sort(key=lambda x: x["edge"], reverse=True)
    return opportunities


def _parse_threshold(ticker: str) -> float | None:
    """Extract rate threshold from ticker like KXFED-27APR-T4.25."""
    try:
        parts = ticker.split("-")
        for part in parts:
            if part.startswith("T"):
                return float(part[1:])
    except (ValueError, IndexError):
        pass
    return None


def _model_prob_above_threshold(
    current_rate: float,
    threshold: float,
    days_to_fomc: int | None,
) -> float:
    """Estimate P(rate > threshold) at next FOMC meeting.

    Simple model based on rate distance. Will be replaced by
    CME FedWatch data when available.

    Logic:
    - If threshold <= current_rate - 0.50: very likely above (95%+)
    - If threshold == current_rate: ~70% (slight hold bias)
    - If threshold == current_rate + 0.25: ~15% (unlikely hike)
    - If threshold >= current_rate + 0.50: ~5% (very unlikely)
    """
    diff = current_rate - threshold  # positive = threshold is below current

    if diff >= 0.75:
        return 0.97
    elif diff >= 0.50:
        return 0.92
    elif diff >= 0.25:
        return 0.82
    elif diff >= 0.0:
        return 0.70  # at current rate, slight hold bias
    elif diff >= -0.25:
        return 0.15  # one hike needed
    elif diff >= -0.50:
        return 0.05
    else:
        return 0.02

    # TODO: replace with CME FedWatch probabilities when feed is available


def _confidence_level(edge: float, days_to_fomc: int | None) -> str:
    """Classify confidence level for logging."""
    if edge > 0.15:
        return "high"
    elif edge > 0.08:
        return "medium"
    else:
        return "low"

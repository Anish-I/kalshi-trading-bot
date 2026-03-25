"""Weather market analyzer for finding edge on Kalshi temperature markets."""

import logging
import math
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class WeatherMarketAnalyzer:
    """Scans Kalshi weather markets and identifies trading opportunities
    by comparing model probabilities against market prices."""

    def __init__(self, kalshi_client, forecast_engine):
        self.kalshi = kalshi_client
        self.engine = forecast_engine

    # ------------------------------------------------------------------ #
    #  Ticker parsing
    # ------------------------------------------------------------------ #

    def parse_strike(self, ticker: str, rules: str = "") -> tuple[str, float, float]:
        """Parse a Kalshi weather ticker into strike type and bounds.

        Parameters
        ----------
        ticker : str
            Market ticker, e.g. ``"KXHIGHNY-26MAR25-T58"`` or
            ``"KXHIGHNY-26MAR25-B55.5"``.
        rules : str
            Market rules text used to disambiguate above/below for T-strikes.

        Returns
        -------
        tuple[str, float, float]
            ``(strike_type, low, high)`` where:
            - ``"above"``  -> ``(threshold, threshold)``
            - ``"below"``  -> ``(threshold, threshold)``
            - ``"between"`` -> ``(floor, ceil)``
            - ``"unknown"`` -> ``(0, 0)``
        """
        # --- Bracket market: -B<number> ---
        bracket_match = re.search(r"-B(\d+(?:\.\d+)?)", ticker)
        if bracket_match:
            midpoint = float(bracket_match.group(1))
            low = math.floor(midpoint)
            high = math.ceil(midpoint)
            # If midpoint is an integer (e.g. B55), treat as 54.5-55.5 range
            if low == high:
                low = midpoint - 0.5
                high = midpoint + 0.5
            return ("between", float(low), float(high))

        # --- Threshold market: -T<number> ---
        threshold_match = re.search(r"-T(\d+(?:\.\d+)?)", ticker)
        if threshold_match:
            value = float(threshold_match.group(1))
            rules_lower = rules.lower()
            if "less than" in rules_lower or "below" in rules_lower:
                return ("below", value, value)
            else:
                # Default to above
                return ("above", value, value)

        logger.warning("Could not parse strike from ticker %r", ticker)
        return ("unknown", 0.0, 0.0)

    # ------------------------------------------------------------------ #
    #  Single-city scan
    # ------------------------------------------------------------------ #

    def scan_city(self, city: dict, target_date: str | None = None) -> list[dict]:
        """Scan all open weather markets for a city and score opportunities.

        Parameters
        ----------
        city : dict
            Must contain ``series_ticker``, ``lat``, ``lon``, ``name``.
        target_date : str or None
            ISO date ``"YYYY-MM-DD"``.  Defaults to today (UTC).

        Returns
        -------
        list[dict]
            Opportunities sorted by edge descending.
        """
        if target_date is None:
            target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        city_name = city["name"]

        # --- Forecast ---
        try:
            mean, std = self.engine.get_forecast_temp(city, target_date)
        except Exception:
            logger.error("Forecast failed for %s on %s, skipping", city_name, target_date, exc_info=True)
            return []

        # --- Fetch open markets ---
        try:
            resp = self.kalshi._request(
                "GET",
                "/markets",
                params={
                    "series_ticker": city["series_ticker"],
                    "status": "open",
                    "limit": 20,
                },
            )
            markets = resp.get("markets", [])
        except Exception:
            logger.error("Failed to fetch markets for %s", city["series_ticker"], exc_info=True)
            return []

        if not markets:
            logger.info("No open markets for %s (%s)", city_name, city["series_ticker"])
            return []

        logger.info("Found %d open markets for %s", len(markets), city_name)

        # --- Evaluate each market ---
        opportunities: list[dict] = []

        for mkt in markets:
            ticker = mkt.get("ticker", "")
            rules = mkt.get("rules_primary", "") or mkt.get("rules", "")

            strike_type, strike_low, strike_high = self.parse_strike(ticker, rules)
            if strike_type == "unknown":
                logger.debug("Skipping unparseable ticker %s", ticker)
                continue

            # Model probability for YES outcome
            if strike_type == "between":
                strike_value = (strike_low + strike_high) / 2.0
            else:
                strike_value = strike_low
            model_prob = self.engine.evaluate_strike(mean, std, strike_type, strike_value)

            # Market prices (dollars as probability, e.g. 0.0500 = 5%)
            yes_bid = float(mkt.get("yes_bid_dollars", 0) or 0)
            yes_ask = float(mkt.get("yes_ask_dollars", 0) or 0)
            no_bid = float(mkt.get("no_bid_dollars", 0) or 0)
            no_ask = float(mkt.get("no_ask_dollars", 0) or 0)

            # Skip markets with no valid quotes
            if yes_bid == 0 and yes_ask == 0:
                logger.debug("No quotes for %s, skipping", ticker)
                continue

            # Mid price as market-implied probability
            market_mid = (yes_bid + yes_ask) / 2.0 if yes_ask > 0 else yes_bid

            # Edge calculation
            yes_edge = model_prob - market_mid
            no_edge = market_mid - model_prob  # (1 - model_prob) - (1 - market_mid)

            if model_prob > market_mid:
                side = "YES"
                edge = yes_edge
                # Suggested limit: don't pay more than our model probability,
                # but also don't exceed the current ask
                suggested_price = min(model_prob, yes_ask) if yes_ask > 0 else model_prob
            else:
                side = "NO"
                edge = no_edge
                # For NO side, price in terms of NO cost = (1 - yes_price)
                suggested_price = min(1.0 - model_prob, no_ask) if no_ask > 0 else (1.0 - model_prob)

            suggested_price_cents = max(1, min(99, round(suggested_price * 100)))

            opportunities.append({
                "ticker": ticker,
                "city": city_name,
                "strike_type": strike_type,
                "strike_low": strike_low,
                "strike_high": strike_high,
                "model_prob": round(model_prob, 4),
                "market_mid": round(market_mid, 4),
                "yes_bid": round(yes_bid, 4),
                "yes_ask": round(yes_ask, 4),
                "edge": round(edge, 4),
                "side": side,
                "suggested_price_cents": suggested_price_cents,
                "forecast_mean": round(mean, 1),
                "forecast_std": round(std, 1),
            })

        # Sort by edge descending
        opportunities.sort(key=lambda x: x["edge"], reverse=True)
        return opportunities

    # ------------------------------------------------------------------ #
    #  Multi-city scan
    # ------------------------------------------------------------------ #

    def find_best_trades(self, cities: list[dict], min_edge: float = 0.15) -> list[dict]:
        """Scan all cities and return opportunities exceeding ``min_edge``.

        Parameters
        ----------
        cities : list[dict]
            List of city configs (see ``config.settings.WEATHER_CITIES``).
        min_edge : float
            Minimum edge to include (default 0.15 = 15%).

        Returns
        -------
        list[dict]
            Filtered and sorted opportunities.
        """
        all_opps: list[dict] = []

        for city in cities:
            opps = self.scan_city(city)
            all_opps.extend(opps)

        # Filter by minimum edge
        filtered = [o for o in all_opps if o["edge"] >= min_edge]
        filtered.sort(key=lambda x: x["edge"], reverse=True)

        logger.info(
            "Found %d total opportunities across %d cities, %d with edge >= %.0f%%",
            len(all_opps), len(cities), len(filtered), min_edge * 100,
        )

        if filtered:
            self.print_scan_report(filtered)

        return filtered

    # ------------------------------------------------------------------ #
    #  Reporting
    # ------------------------------------------------------------------ #

    def print_scan_report(self, opportunities: list[dict]) -> None:
        """Pretty-print a table of weather trading opportunities."""
        header = (
            f"{'City':<10} {'Strike':<18} {'Model%':>7} {'Mkt%':>7} "
            f"{'Edge':>7} {'Side':>4} {'Price':>6}"
        )
        separator = "-" * len(header)

        lines = [
            "",
            "=== Weather Market Scan Report ===",
            separator,
            header,
            separator,
        ]

        for opp in opportunities:
            # Format strike description
            if opp["strike_type"] == "between":
                strike_str = f"B {opp['strike_low']:.0f}-{opp['strike_high']:.0f}F"
            elif opp["strike_type"] == "above":
                strike_str = f"> {opp['strike_low']:.0f}F"
            elif opp["strike_type"] == "below":
                strike_str = f"< {opp['strike_low']:.0f}F"
            else:
                strike_str = "???"

            line = (
                f"{opp['city']:<10} {strike_str:<18} "
                f"{opp['model_prob']*100:6.1f}% {opp['market_mid']*100:6.1f}% "
                f"{opp['edge']*100:+6.1f}% {opp['side']:>4} "
                f"{opp['suggested_price_cents']:>4}c"
            )
            lines.append(line)

        lines.append(separator)
        lines.append(f"Forecast: mean={opportunities[0]['forecast_mean']:.1f}F, "
                     f"std={opportunities[0]['forecast_std']:.1f}F")
        lines.append("")

        report = "\n".join(lines)
        logger.info(report)
        print(report)

"""Backtest weather trading strategy on settled Kalshi markets."""

import logging
import math
import re
from collections import defaultdict

from scipy.stats import norm

logger = logging.getLogger(__name__)


class WeatherBacktest:
    """Run a backtest of the Gaussian-forecast weather trading strategy
    against historically settled Kalshi temperature markets."""

    def __init__(self, kalshi_client) -> None:
        self.client = kalshi_client

    # ------------------------------------------------------------------ #
    #  Data fetching
    # ------------------------------------------------------------------ #

    def fetch_settled_markets(
        self,
        series_ticker: str,
        limit: int = 100,
    ) -> list[dict]:
        """Fetch settled markets for a given series ticker.

        Parameters
        ----------
        series_ticker : str
            Kalshi series ticker, e.g. ``"KXHIGHNY"``.
        limit : int
            Maximum number of markets to return.

        Returns
        -------
        list[dict]
            List of settled market dicts from the Kalshi API.
        """
        try:
            resp = self.client._request(
                "GET",
                "/markets",
                params={
                    "series_ticker": series_ticker,
                    "status": "settled",
                    "limit": limit,
                },
            )
            markets = resp.get("markets", [])
            logger.info(
                "Fetched %d settled markets for %s", len(markets), series_ticker,
            )
            return markets
        except Exception:
            logger.error(
                "Failed to fetch settled markets for %s", series_ticker, exc_info=True,
            )
            return []

    # ------------------------------------------------------------------ #
    #  Ticker / strike parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_strike(ticker: str, rules: str = "") -> tuple[str, float]:
        """Parse strike type and value from a ticker.

        Returns
        -------
        tuple[str, float]
            ``(strike_type, strike_value)`` where strike_type is
            ``"above"``, ``"below"``, ``"between"``, or ``"unknown"``.
        """
        # Bracket market: -B<number>
        bracket = re.search(r"-B(\d+(?:\.\d+)?)", ticker)
        if bracket:
            return ("between", float(bracket.group(1)))

        # Threshold market: -T<number>
        threshold = re.search(r"-T(\d+(?:\.\d+)?)", ticker)
        if threshold:
            value = float(threshold.group(1))
            rules_lower = rules.lower()
            if "less than" in rules_lower or "below" in rules_lower:
                return ("below", value)
            return ("above", value)

        return ("unknown", 0.0)

    @staticmethod
    def parse_date_from_ticker(ticker: str) -> str | None:
        """Extract the date portion from a ticker like KXHIGHNY-26MAR25-T58.

        Returns ``"2025-03-26"`` format or ``None``.
        """
        match = re.search(r"-(\d{1,2})([A-Z]{3})(\d{2})-", ticker.upper())
        if not match:
            return None

        day = int(match.group(1))
        month_str = match.group(2)
        year = 2000 + int(match.group(3))

        months = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
            "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        }
        month = months.get(month_str)
        if month is None:
            return None

        return f"{year:04d}-{month:02d}-{day:02d}"

    # ------------------------------------------------------------------ #
    #  Probability model (same as forecast_engine)
    # ------------------------------------------------------------------ #

    @staticmethod
    def model_prob_yes(
        strike_type: str,
        strike_value: float,
        actual_temp: float,
        forecast_std: float,
    ) -> float:
        """Compute P(YES) for a strike given a Gaussian centred on actual_temp.

        This simulates what our model would have predicted if the forecast
        was centred on *actual_temp* with *forecast_std* noise.
        """
        if strike_type == "above":
            return float(1.0 - norm.cdf(strike_value, loc=actual_temp, scale=forecast_std))
        elif strike_type == "below":
            return float(norm.cdf(strike_value, loc=actual_temp, scale=forecast_std))
        elif strike_type == "between":
            low = strike_value - 0.5
            high = strike_value + 0.5
            return float(
                norm.cdf(high, loc=actual_temp, scale=forecast_std)
                - norm.cdf(low, loc=actual_temp, scale=forecast_std)
            )
        return 0.5

    # ------------------------------------------------------------------ #
    #  Backtest logic
    # ------------------------------------------------------------------ #

    def run_backtest(
        self,
        series_ticker: str,
        forecast_std: float = 2.5,
        min_edge: float = 0.10,
    ) -> dict:
        """Run a backtest on settled markets for the given series.

        Strategy
        --------
        1. Group settled markets by date.
        2. For each date, find the bracket/threshold that settled YES to infer
           the approximate actual temperature.
        3. For every market on that date, compute what our model probability
           would have been (Gaussian centred on inferred actual temp with
           ``forecast_std``).
        4. Compare model probability vs the market's last traded price.
           If edge > ``min_edge``, count as a simulated trade.  Check if the
           side we would have chosen matches the settlement outcome.

        Parameters
        ----------
        series_ticker : str
            Kalshi series ticker.
        forecast_std : float
            Standard deviation for the Gaussian forecast model.
        min_edge : float
            Minimum edge threshold to trigger a simulated trade.

        Returns
        -------
        dict
            Backtest results summary.
        """
        markets = self.fetch_settled_markets(series_ticker)
        if not markets:
            return {
                "series_ticker": series_ticker,
                "n_markets": 0,
                "n_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl_cents": 0,
                "avg_edge": 0.0,
            }

        # --- Group markets by date ---
        by_date: dict[str, list[dict]] = defaultdict(list)
        for mkt in markets:
            ticker = mkt.get("ticker", "")
            date_str = self.parse_date_from_ticker(ticker)
            if date_str:
                by_date[date_str].append(mkt)

        # --- Infer actual temp per date ---
        n_markets = len(markets)
        n_trades = 0
        wins = 0
        losses = 0
        total_pnl_cents = 0
        edges: list[float] = []

        for date_str, date_markets in by_date.items():
            actual_temp = self._infer_actual_temp(date_markets)
            if actual_temp is None:
                logger.debug("Could not infer actual temp for %s, skipping", date_str)
                continue

            for mkt in date_markets:
                ticker = mkt.get("ticker", "")
                rules = mkt.get("rules_primary", "") or mkt.get("rules", "")
                result = mkt.get("result", "").lower()

                strike_type, strike_value = self.parse_strike(ticker, rules)
                if strike_type == "unknown":
                    continue

                # Model probability
                model_p = self.model_prob_yes(
                    strike_type, strike_value, actual_temp, forecast_std,
                )

                # Market price as probability
                last_price = mkt.get("last_price", 0) or 0
                market_p = float(last_price) / 100.0

                if market_p <= 0 or market_p >= 1:
                    continue

                # Determine trade side
                yes_edge = model_p - market_p
                no_edge = market_p - model_p

                if yes_edge >= min_edge:
                    side = "yes"
                    edge = yes_edge
                    entry_price_cents = int(market_p * 100)
                elif no_edge >= min_edge:
                    side = "no"
                    edge = no_edge
                    entry_price_cents = int((1.0 - market_p) * 100)
                else:
                    continue

                # Would we have won?
                settled_yes = result == "yes"
                if side == "yes":
                    pnl = (100 - entry_price_cents) if settled_yes else -entry_price_cents
                else:
                    pnl = (100 - entry_price_cents) if not settled_yes else -entry_price_cents

                n_trades += 1
                edges.append(edge)
                total_pnl_cents += pnl

                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

                logger.debug(
                    "BT  %s  side=%s  entry=%dc  result=%s  pnl=%+dc  edge=%.1f%%",
                    ticker, side, entry_price_cents, result, pnl, edge * 100,
                )

        win_rate = wins / n_trades if n_trades > 0 else 0.0
        avg_edge = sum(edges) / len(edges) if edges else 0.0

        results = {
            "series_ticker": series_ticker,
            "n_markets": n_markets,
            "n_dates": len(by_date),
            "n_trades": n_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "total_pnl_cents": total_pnl_cents,
            "avg_edge": round(avg_edge, 4),
            "forecast_std": forecast_std,
            "min_edge": min_edge,
        }

        logger.info(
            "Backtest %s: %d markets, %d trades, %d W / %d L (%.1f%%), "
            "P&L %+dc, avg edge %.1f%%",
            series_ticker, n_markets, n_trades, wins, losses,
            win_rate * 100, total_pnl_cents, avg_edge * 100,
        )

        return results

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _infer_actual_temp(self, date_markets: list[dict]) -> float | None:
        """Infer the actual temperature from a group of markets on the same date.

        Heuristic: find the threshold-type market that settled YES with the
        highest strike value (for "above" type) — the actual temp was above
        that strike.  Or find a bracket market that settled YES — the actual
        temp was in that bracket.
        """
        # Try bracket markets first (most precise)
        for mkt in date_markets:
            ticker = mkt.get("ticker", "")
            result = mkt.get("result", "").lower()
            if result != "yes":
                continue

            strike_type, strike_value = self.parse_strike(
                ticker, mkt.get("rules_primary", "") or mkt.get("rules", ""),
            )
            if strike_type == "between":
                # Bracket settled YES — actual temp was in this range
                return strike_value  # midpoint

        # Fall back to threshold markets
        above_yes: list[float] = []
        below_yes: list[float] = []

        for mkt in date_markets:
            ticker = mkt.get("ticker", "")
            result = mkt.get("result", "").lower()
            rules = mkt.get("rules_primary", "") or mkt.get("rules", "")
            strike_type, strike_value = self.parse_strike(ticker, rules)

            if result == "yes" and strike_type == "above":
                above_yes.append(strike_value)
            elif result == "yes" and strike_type == "below":
                below_yes.append(strike_value)

        if above_yes and below_yes:
            # Actual temp was above max(above_yes thresholds) and below min(below_yes thresholds)
            return (max(above_yes) + min(below_yes)) / 2.0
        elif above_yes:
            # Actual was above the highest threshold that settled YES
            return max(above_yes) + 1.0
        elif below_yes:
            # Actual was below the lowest threshold that settled YES
            return min(below_yes) - 1.0

        return None

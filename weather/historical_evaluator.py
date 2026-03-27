"""Honest historical weather evaluation using archived forecast snapshots."""

from __future__ import annotations

from collections import defaultdict
import logging
import math
import re

logger = logging.getLogger(__name__)


class HistoricalWeatherEvaluator:
    """Evaluate archived prior-run forecasts against settled Kalshi weather markets.

    This path is intentionally separate from the legacy backtests. It uses
    archived forecast snapshots from Open-Meteo's Previous Runs API and never
    derives the forecast from the realized outcome.
    """

    def __init__(self, kalshi_client, meteo_client) -> None:
        self.client = kalshi_client
        self.meteo = meteo_client

    def fetch_settled_markets(self, series_ticker: str, max_pages: int = 20, page_limit: int = 200) -> list[dict]:
        """Fetch all settled markets for a given Kalshi series."""
        markets: list[dict] = []
        cursor: str | None = None

        for _ in range(max_pages):
            params = {
                "series_ticker": series_ticker,
                "status": "settled",
                "limit": page_limit,
            }
            if cursor:
                params["cursor"] = cursor

            resp = self.client._request("GET", "/markets", params=params)
            page = resp.get("markets", [])
            if not page:
                break

            markets.extend(page)
            cursor = resp.get("cursor")
            if not cursor:
                break

        logger.info("Fetched %d settled markets for %s", len(markets), series_ticker)
        return markets

    @staticmethod
    def parse_strike(ticker: str, rules: str = "") -> tuple[str, float]:
        """Parse the strike encoded in a weather market ticker."""
        bracket = re.search(r"-B(\d+(?:\.\d+)?)", ticker)
        if bracket:
            return ("between", float(bracket.group(1)))

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
        """Extract an ISO date from a weather market ticker."""
        match = re.search(r"-(\d{2})([A-Z]{3})(\d{1,2})-", ticker.upper())
        if not match:
            return None

        year = 2000 + int(match.group(1))
        month_str = match.group(2)
        day = int(match.group(3))

        months = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
            "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        }
        month = months.get(month_str)
        if month is None:
            return None

        return f"{year:04d}-{month:02d}-{day:02d}"

    @classmethod
    def group_markets_by_date(cls, markets: list[dict]) -> dict[str, list[dict]]:
        """Group settled markets by their target date."""
        by_date: dict[str, list[dict]] = defaultdict(list)
        for market in markets:
            date_str = cls.parse_date_from_ticker(market.get("ticker", ""))
            if date_str:
                by_date[date_str].append(market)
        return by_date

    @classmethod
    def infer_actual_temp(cls, date_markets: list[dict]) -> float | None:
        """Infer the realized temperature from settled bracket/threshold outcomes."""
        for market in date_markets:
            if market.get("result", "").lower() != "yes":
                continue
            strike_type, strike_value = cls.parse_strike(
                market.get("ticker", ""),
                market.get("rules_primary", "") or market.get("rules", ""),
            )
            if strike_type == "between":
                return strike_value

        above_yes: list[float] = []
        below_yes: list[float] = []

        for market in date_markets:
            if market.get("result", "").lower() != "yes":
                continue
            strike_type, strike_value = cls.parse_strike(
                market.get("ticker", ""),
                market.get("rules_primary", "") or market.get("rules", ""),
            )
            if strike_type == "above":
                above_yes.append(strike_value)
            elif strike_type == "below":
                below_yes.append(strike_value)

        if above_yes and below_yes:
            return (max(above_yes) + min(below_yes)) / 2.0
        if above_yes:
            return max(above_yes) + 1.0
        if below_yes:
            return min(below_yes) - 1.0
        return None

    @staticmethod
    def forecast_implies_yes(strike_type: str, strike_value: float, forecast_temp: float) -> bool:
        """Map a deterministic forecast temperature to a market-side prediction."""
        if strike_type == "above":
            return forecast_temp > strike_value
        if strike_type == "below":
            return forecast_temp < strike_value
        if strike_type == "between":
            return (strike_value - 0.5) <= forecast_temp <= (strike_value + 0.5)
        return False

    def evaluate_city(
        self,
        city: dict,
        lead_days: int = 1,
        start_date: str | None = None,
        end_date: str | None = None,
        max_dates: int | None = None,
    ) -> dict:
        """Evaluate one city's settled markets against archived previous-run forecasts."""
        markets = self.fetch_settled_markets(city["series_ticker"])
        by_date = self.group_markets_by_date(markets)
        available_dates = sorted(by_date)

        if start_date:
            available_dates = [d for d in available_dates if d >= start_date]
        if end_date:
            available_dates = [d for d in available_dates if d <= end_date]
        if max_dates is not None:
            available_dates = available_dates[-max_dates:]

        if not available_dates:
            return {
                "city": city["name"],
                "short": city["short"],
                "series_ticker": city["series_ticker"],
                "temp_type": city.get("type", "high"),
                "lead_days": lead_days,
                "n_total_dates": 0,
                "n_covered_dates": 0,
                "skipped_no_forecast": 0,
                "skipped_no_actual": 0,
                "mae_deg_f": 0.0,
                "rmse_deg_f": 0.0,
                "bias_deg_f": 0.0,
                "market_accuracy": 0.0,
                "market_correct": 0,
                "market_total": 0,
                "daily_rows": [],
                "_errors": [],
            }

        forecast_rows = self.meteo.get_previous_run_daily_forecast(
            city["lat"],
            city["lon"],
            available_dates[0],
            available_dates[-1],
            lead_days=lead_days,
        )
        forecast_by_date = {row["date"]: row for row in forecast_rows}
        temp_key = "low_f" if city.get("type") == "low" else "high_f"

        errors: list[float] = []
        daily_rows: list[dict] = []
        market_total = 0
        market_correct = 0
        skipped_no_forecast = 0
        skipped_no_actual = 0

        for date_str in available_dates:
            forecast_row = forecast_by_date.get(date_str)
            if forecast_row is None:
                skipped_no_forecast += 1
                continue

            actual_temp = self.infer_actual_temp(by_date[date_str])
            if actual_temp is None:
                skipped_no_actual += 1
                continue

            forecast_temp = float(forecast_row[temp_key])
            error = forecast_temp - actual_temp
            errors.append(error)

            day_market_total = 0
            day_market_correct = 0
            for market in by_date[date_str]:
                strike_type, strike_value = self.parse_strike(
                    market.get("ticker", ""),
                    market.get("rules_primary", "") or market.get("rules", ""),
                )
                result = market.get("result", "").lower()
                if strike_type == "unknown" or result not in {"yes", "no"}:
                    continue

                predicted_yes = self.forecast_implies_yes(strike_type, strike_value, forecast_temp)
                correct = predicted_yes == (result == "yes")
                day_market_total += 1
                market_total += 1
                if correct:
                    day_market_correct += 1
                    market_correct += 1

            daily_rows.append({
                "date": date_str,
                "forecast_temp_f": round(forecast_temp, 2),
                "actual_temp_f": round(actual_temp, 2),
                "error_f": round(error, 2),
                "market_correct": day_market_correct,
                "market_total": day_market_total,
            })

        mae = sum(abs(err) for err in errors) / len(errors) if errors else 0.0
        rmse = math.sqrt(sum(err * err for err in errors) / len(errors)) if errors else 0.0
        bias = sum(errors) / len(errors) if errors else 0.0
        market_accuracy = market_correct / market_total if market_total else 0.0

        return {
            "city": city["name"],
            "short": city["short"],
            "series_ticker": city["series_ticker"],
            "temp_type": city.get("type", "high"),
            "lead_days": lead_days,
            "n_total_dates": len(available_dates),
            "n_covered_dates": len(daily_rows),
            "skipped_no_forecast": skipped_no_forecast,
            "skipped_no_actual": skipped_no_actual,
            "mae_deg_f": round(mae, 4),
            "rmse_deg_f": round(rmse, 4),
            "bias_deg_f": round(bias, 4),
            "market_accuracy": round(market_accuracy, 4),
            "market_correct": market_correct,
            "market_total": market_total,
            "daily_rows": daily_rows,
            "_errors": errors,
        }

    def evaluate(
        self,
        cities: list[dict],
        lead_days: int = 1,
        start_date: str | None = None,
        end_date: str | None = None,
        max_dates: int | None = None,
    ) -> dict:
        """Evaluate a list of cities and return aggregate metrics."""
        city_results: list[dict] = []
        all_errors: list[float] = []
        total_market_correct = 0
        total_market_total = 0
        total_covered_dates = 0
        total_requested_dates = 0

        for city in cities:
            result = self.evaluate_city(
                city,
                lead_days=lead_days,
                start_date=start_date,
                end_date=end_date,
                max_dates=max_dates,
            )
            all_errors.extend(result["_errors"])
            total_market_correct += result["market_correct"]
            total_market_total += result["market_total"]
            total_covered_dates += result["n_covered_dates"]
            total_requested_dates += result["n_total_dates"]

            city_results.append({
                key: value
                for key, value in result.items()
                if key != "_errors"
            })

        mae = sum(abs(err) for err in all_errors) / len(all_errors) if all_errors else 0.0
        rmse = math.sqrt(sum(err * err for err in all_errors) / len(all_errors)) if all_errors else 0.0
        bias = sum(all_errors) / len(all_errors) if all_errors else 0.0
        market_accuracy = total_market_correct / total_market_total if total_market_total else 0.0

        return {
            "lead_days": lead_days,
            "start_date": start_date,
            "end_date": end_date,
            "max_dates": max_dates,
            "n_cities": len(cities),
            "n_requested_dates": total_requested_dates,
            "n_covered_dates": total_covered_dates,
            "mae_deg_f": round(mae, 4),
            "rmse_deg_f": round(rmse, 4),
            "bias_deg_f": round(bias, 4),
            "market_accuracy": round(market_accuracy, 4),
            "market_correct": total_market_correct,
            "market_total": total_market_total,
            "cities": city_results,
        }

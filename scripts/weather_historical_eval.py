"""Evaluate archived weather forecasts against settled Kalshi weather markets."""

from __future__ import annotations

import argparse
import json
import logging
import sys

sys.path.insert(0, ".")

from config.settings import WEATHER_CITIES
from kalshi.client import KalshiClient
from weather.historical_evaluator import HistoricalWeatherEvaluator
from weather.open_meteo_client import OpenMeteoClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Honest weather evaluation using archived Open-Meteo previous-run forecasts.",
    )
    parser.add_argument(
        "--cities",
        help="Comma-separated city shorts, names, or series tickers. Default: all configured weather cities.",
    )
    parser.add_argument(
        "--lead-days",
        type=int,
        default=1,
        help="Archived forecast lead in days. 1 = yesterday's forecast for today.",
    )
    parser.add_argument("--start-date", help="Optional start date filter (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Optional end date filter (YYYY-MM-DD).")
    parser.add_argument(
        "--max-dates",
        type=int,
        help="Optional limit to the most recent N settled dates per city.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON output instead of a text table.",
    )
    return parser.parse_args()


def select_cities(selector: str | None) -> list[dict]:
    if not selector:
        return WEATHER_CITIES

    wanted = {token.strip().upper() for token in selector.split(",") if token.strip()}
    selected = [
        city for city in WEATHER_CITIES
        if city["short"].upper() in wanted
        or city["series_ticker"].upper() in wanted
        or city["name"].upper() in wanted
    ]

    if not selected:
        raise SystemExit(f"No configured weather cities matched: {selector}")

    return selected


def print_text_report(results: dict) -> None:
    sep = "=" * 84
    print(f"\n{sep}")
    print("  HONEST WEATHER EVAL")
    print("  Archived previous-run forecasts vs settled Kalshi weather outcomes")
    print(sep)
    print(
        f"Lead days: {results['lead_days']} | Cities: {results['n_cities']} | "
        f"Covered dates: {results['n_covered_dates']} / {results['n_requested_dates']}"
    )
    print(
        f"Overall MAE: {results['mae_deg_f']:.2f}F | RMSE: {results['rmse_deg_f']:.2f}F | "
        f"Bias: {results['bias_deg_f']:+.2f}F | "
        f"Market accuracy: {results['market_accuracy'] * 100:.1f}% "
        f"({results['market_correct']}/{results['market_total']})"
    )

    print(
        f"\n{'City':<8} {'Type':<5} {'Dates':>7} {'MAE':>7} {'RMSE':>7} "
        f"{'Bias':>7} {'MktAcc':>8}"
    )
    print("-" * 60)
    for city in results["cities"]:
        print(
            f"{city['short']:<8} {city['temp_type']:<5} "
            f"{city['n_covered_dates']:>7d} {city['mae_deg_f']:>7.2f} "
            f"{city['rmse_deg_f']:>7.2f} {city['bias_deg_f']:>+7.2f} "
            f"{city['market_accuracy'] * 100:>7.1f}%"
        )

    print("\nNote: actual temperature is inferred from settled Kalshi market outcomes.")
    print("This measures honest forecast error, not tradable P&L or edge versus historical prices.")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cities = select_cities(args.cities)
    evaluator = HistoricalWeatherEvaluator(KalshiClient(), OpenMeteoClient())
    results = evaluator.evaluate(
        cities,
        lead_days=args.lead_days,
        start_date=args.start_date,
        end_date=args.end_date,
        max_dates=args.max_dates,
    )

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print_text_report(results)


if __name__ == "__main__":
    main()

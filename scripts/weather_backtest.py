#!/usr/bin/env python3
"""Backtest weather trading strategy on settled Kalshi markets.

Usage:
    python scripts/weather_backtest.py
    python scripts/weather_backtest.py --std 3.0 --edge 0.12
"""

import argparse
import logging
import sys

sys.path.insert(0, ".")

from kalshi.client import KalshiClient
from weather.backtest import WeatherBacktest
from config.settings import WEATHER_CITIES

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("weather_backtest")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest weather trading strategy")
    parser.add_argument(
        "--std",
        type=float,
        default=2.5,
        help="Forecast standard deviation in degF (default 2.5)",
    )
    parser.add_argument(
        "--edge",
        type=float,
        default=0.10,
        help="Minimum edge to trigger a trade (default 0.10 = 10%%)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max settled markets to fetch per city (default 100)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Weather Strategy Backtest")
    logger.info("  Forecast std: %.1f degF", args.std)
    logger.info("  Min edge: %.0f%%", args.edge * 100)
    logger.info("  Market limit: %d per city", args.limit)
    logger.info("=" * 60)

    try:
        client = KalshiClient()
    except Exception:
        logger.critical("Failed to initialise KalshiClient", exc_info=True)
        sys.exit(1)

    bt = WeatherBacktest(client)

    results: list[dict] = []

    for city in WEATHER_CITIES:
        city_name = city["name"]
        series = city["series_ticker"]

        logger.info("--- Backtesting %s (%s) ---", city_name, series)

        try:
            r = bt.run_backtest(
                series_ticker=series,
                forecast_std=args.std,
                min_edge=args.edge,
            )
            r["city"] = city_name
            results.append(r)
        except Exception:
            logger.error("Backtest failed for %s", city_name, exc_info=True)
            results.append({
                "city": city_name,
                "series_ticker": series,
                "n_markets": 0,
                "n_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl_cents": 0,
                "avg_edge": 0.0,
                "error": True,
            })

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("  BACKTEST RESULTS SUMMARY")
    print("=" * 80)

    header = (
        f"{'City':<12} {'Series':<14} {'Markets':>8} {'Trades':>7} "
        f"{'Wins':>5} {'Losses':>7} {'WinRate':>8} {'P&L':>9} {'AvgEdge':>8}"
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_pnl = 0

    for r in results:
        win_pct = f"{r['win_rate'] * 100:.1f}%" if r["n_trades"] > 0 else "N/A"
        avg_edge_pct = f"{r['avg_edge'] * 100:.1f}%" if r["n_trades"] > 0 else "N/A"

        print(
            f"{r.get('city', '?'):<12} {r['series_ticker']:<14} "
            f"{r['n_markets']:>8} {r['n_trades']:>7} "
            f"{r['wins']:>5} {r['losses']:>7} "
            f"{win_pct:>8} {r['total_pnl_cents']:>+8}c {avg_edge_pct:>8}"
        )

        total_trades += r["n_trades"]
        total_wins += r["wins"]
        total_losses += r["losses"]
        total_pnl += r["total_pnl_cents"]

    print(separator)

    overall_wr = f"{total_wins / total_trades * 100:.1f}%" if total_trades > 0 else "N/A"
    print(
        f"{'TOTAL':<12} {'':14} {'':>8} {total_trades:>7} "
        f"{total_wins:>5} {total_losses:>7} "
        f"{overall_wr:>8} {total_pnl:>+8}c"
    )

    print(separator)
    print()
    print(f"  Parameters: forecast_std={args.std} degF, min_edge={args.edge * 100:.0f}%")
    print()


if __name__ == "__main__":
    main()

"""
Full historical backtest of weather trading strategy.
Uses settled Kalshi markets + NWS-quality forecast simulation.

Tests: If we had used a Gaussian model (forecast +/- 2.5F std) against
Kalshi market prices, what would our win rate and P&L have been?
"""
import sys
import re
import logging
from collections import defaultdict

import numpy as np
from scipy.stats import norm

sys.path.insert(0, ".")

from kalshi.client import KalshiClient
from config.settings import WEATHER_CITIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backtest")


def parse_strike(ticker, rules=""):
    b_match = re.search(r"-B(\d+\.?\d*)", ticker)
    if b_match:
        mid = float(b_match.group(1))
        return "between", mid - 0.5, mid + 0.5
    t_match = re.search(r"-T(\d+\.?\d*)", ticker)
    if t_match:
        val = float(t_match.group(1))
        if "less than" in rules.lower() or "below" in rules.lower():
            return "below", val, val
        return "above", val, val
    return "unknown", 0, 0


def infer_actual_temp(markets):
    """Infer actual temperature from settled market results."""
    # Best: find the bracket that settled YES
    for m in markets:
        stype, low, high = parse_strike(m["ticker"], m.get("rules_primary", ""))
        if stype == "between" and m.get("result") == "yes":
            return (low + high) / 2

    # Fallback: use threshold markets
    above_thresholds_yes = []
    above_thresholds_no = []
    below_thresholds_yes = []

    for m in markets:
        stype, val, _ = parse_strike(m["ticker"], m.get("rules_primary", ""))
        if stype == "above":
            if m.get("result") == "yes":
                above_thresholds_yes.append(val)
            else:
                above_thresholds_no.append(val)
        elif stype == "below":
            if m.get("result") == "yes":
                below_thresholds_yes.append(val)

    # Actual temp is above the highest "above" threshold that settled YES,
    # and below the lowest "above" threshold that settled NO
    if above_thresholds_yes and above_thresholds_no:
        return (max(above_thresholds_yes) + min(above_thresholds_no)) / 2
    if above_thresholds_yes:
        return max(above_thresholds_yes) + 1
    if below_thresholds_yes:
        return min(below_thresholds_yes) - 1
    return None


def run_backtest():
    c = KalshiClient()

    all_trades = []
    city_stats = {}

    for city in WEATHER_CITIES:
        series = city["series_ticker"]
        log.info("Fetching settled markets for %s (%s)...", city["name"], series)

        # Paginate to get all settled markets
        all_markets = []
        cursor = None
        for _ in range(20):
            params = {"series_ticker": series, "status": "settled", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            result = c._request("GET", "/markets", params=params)
            markets = result.get("markets", [])
            all_markets.extend(markets)
            cursor = result.get("cursor")
            if not cursor or not markets:
                break

        # Group by date
        by_date = defaultdict(list)
        for m in all_markets:
            date_match = re.search(r"-(\d{2}[A-Z]{3}\d{2})-", m["ticker"])
            if date_match:
                by_date[date_match.group(1)].append(m)

        log.info("  %d markets, %d days", len(all_markets), len(by_date))

        city_trades = 0
        city_wins = 0
        city_pnl = 0.0

        for date_key in sorted(by_date.keys()):
            day_markets = by_date[date_key]
            actual = infer_actual_temp(day_markets)
            if actual is None:
                continue

            # Simulate NWS-quality forecast: actual + noise
            # Use deterministic seed per date for reproducibility
            rng = np.random.RandomState(hash(date_key + city["short"]) % 2**31)
            forecast_error = rng.normal(0, 2.5)
            forecast = actual + forecast_error
            forecast_std = 2.5

            for m in day_markets:
                stype, low, high = parse_strike(m["ticker"], m.get("rules_primary", ""))
                if stype == "unknown":
                    continue

                result_val = m.get("result", "")

                # Use previous_price as proxy for early market price
                # If not available, use yes_bid/ask from when market was active
                prev_price = float(m.get("previous_price_dollars") or 0)
                yes_bid = float(m.get("previous_yes_bid_dollars") or m.get("yes_bid_dollars") or 0)
                yes_ask = float(m.get("previous_yes_ask_dollars") or m.get("yes_ask_dollars") or 0)

                # Use mid of previous bid/ask if available, else previous_price
                if yes_bid > 0 and yes_ask > 0:
                    market_price = (yes_bid + yes_ask) / 2
                elif prev_price > 0.02:
                    market_price = prev_price
                else:
                    continue  # No usable price

                # Skip already-settled extremes
                if market_price <= 0.03 or market_price >= 0.97:
                    continue

                # Model probability
                if stype == "above":
                    model_p = 1 - norm.cdf(low, loc=forecast, scale=forecast_std)
                elif stype == "below":
                    model_p = norm.cdf(low, loc=forecast, scale=forecast_std)
                elif stype == "between":
                    model_p = norm.cdf(high, loc=forecast, scale=forecast_std) - \
                              norm.cdf(low, loc=forecast, scale=forecast_std)
                else:
                    continue

                # Edge calculation
                edge_yes = model_p - market_price
                edge_no = market_price - model_p
                best_edge = max(edge_yes, edge_no)

                # Only trade with significant edge
                if best_edge < 0.15:
                    continue

                # Determine trade
                if edge_yes > edge_no:
                    side = "yes"
                    entry = market_price
                    correct = result_val == "yes"
                else:
                    side = "no"
                    entry = 1 - market_price
                    correct = result_val == "no"

                pnl = (1.0 - entry) if correct else -entry

                city_trades += 1
                city_wins += int(correct)
                city_pnl += pnl

                all_trades.append({
                    "city": city["short"],
                    "date": date_key,
                    "ticker": m["ticker"],
                    "actual": actual,
                    "forecast": forecast,
                    "strike": f"{stype} {low}" + (f"-{high}" if stype == "between" else ""),
                    "model_p": model_p,
                    "market_p": market_price,
                    "edge": best_edge,
                    "side": side,
                    "entry": entry,
                    "correct": correct,
                    "pnl": pnl,
                })

        wr = city_wins / city_trades * 100 if city_trades > 0 else 0
        city_stats[city["short"]] = {
            "trades": city_trades, "wins": city_wins,
            "losses": city_trades - city_wins, "win_rate": wr, "pnl": city_pnl,
        }
        log.info("  %s: %d trades, %d wins (%.1f%%), P&L=$%.2f/contract",
                 city["short"], city_trades, city_wins, wr, city_pnl)

    # Overall summary
    sep = "=" * 65
    print(f"\n{sep}")
    print("  WEATHER BACKTEST — FULL HISTORY")
    print(f"  NWS-quality forecast (2.5F std), 15% min edge")
    print(sep)

    total_trades = sum(s["trades"] for s in city_stats.values())
    total_wins = sum(s["wins"] for s in city_stats.values())
    total_pnl = sum(s["pnl"] for s in city_stats.values())

    print(f"\n{'City':>6s} {'Trades':>7s} {'Wins':>6s} {'Losses':>7s} {'Win%':>6s} {'P&L/ct':>9s}")
    print("-" * 45)
    for city_short, s in city_stats.items():
        print(f"{city_short:>6s} {s['trades']:>7d} {s['wins']:>6d} {s['losses']:>7d} "
              f"{s['win_rate']:>5.1f}% ${s['pnl']:>7.2f}")
    print("-" * 45)
    overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    print(f"{'TOTAL':>6s} {total_trades:>7d} {total_wins:>6d} {total_trades-total_wins:>7d} "
          f"{overall_wr:>5.1f}% ${total_pnl:>7.2f}")

    if total_trades > 0:
        avg_pnl = total_pnl / total_trades
        print(f"\nAvg P&L per trade: ${avg_pnl:.3f}/contract")
        print(f"At 10 contracts/trade: ${total_pnl * 10:.2f} total")

        # Edge breakdown
        print(f"\n{'Edge Range':>12s} {'Trades':>7s} {'Win%':>6s} {'Avg P&L':>9s}")
        print("-" * 38)
        for lo, hi, label in [(0.15, 0.25, "15-25%"), (0.25, 0.40, "25-40%"), (0.40, 1.0, "40%+")]:
            bucket = [t for t in all_trades if lo <= t["edge"] < hi]
            if bucket:
                bw = sum(1 for t in bucket if t["correct"])
                bp = sum(t["pnl"] for t in bucket)
                print(f"{label:>12s} {len(bucket):>7d} {bw/len(bucket)*100:>5.1f}% ${bp/len(bucket):>7.3f}")

        # Show some example trades
        print(f"\nSample winning trades:")
        wins = [t for t in all_trades if t["correct"]][:10]
        for t in wins:
            print(f"  {t['city']} {t['date']} {t['strike']:20s} {t['side']:3s} "
                  f"fc={t['forecast']:.0f} actual={t['actual']:.0f} "
                  f"model={t['model_p']:.0%} mkt={t['market_p']:.0%} edge={t['edge']:.0%} pnl={t['pnl']:+.2f}")

        print(f"\nSample losing trades:")
        losses = [t for t in all_trades if not t["correct"]][:10]
        for t in losses:
            print(f"  {t['city']} {t['date']} {t['strike']:20s} {t['side']:3s} "
                  f"fc={t['forecast']:.0f} actual={t['actual']:.0f} "
                  f"model={t['model_p']:.0%} mkt={t['market_p']:.0%} edge={t['edge']:.0%} pnl={t['pnl']:+.2f}")


if __name__ == "__main__":
    run_backtest()

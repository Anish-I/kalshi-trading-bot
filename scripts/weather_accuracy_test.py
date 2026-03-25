"""Backtest weather strategy accuracy on settled Kalshi markets."""
import sys
import re
sys.path.insert(0, ".")

from scipy.stats import norm
from kalshi.client import KalshiClient
from config.settings import WEATHER_CITIES

c = KalshiClient()
SEP = "=" * 60


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
    for m in markets:
        stype, low, high = parse_strike(m["ticker"], m.get("rules_primary", ""))
        if stype == "between" and m.get("result") == "yes":
            return (low + high) / 2
    for m in markets:
        stype, val, _ = parse_strike(m["ticker"], m.get("rules_primary", ""))
        if stype == "above" and m.get("result") == "yes":
            return val + 1
        if stype == "below" and m.get("result") == "yes":
            return val - 1
    return None


total_trades = 0
total_correct = 0
total_pnl = 0.0

for city in WEATHER_CITIES:
    series = city["series_ticker"]
    print(f"\n{SEP}")
    print(f"  {city['name']} ({series})")
    print(SEP)

    result = c._request("GET", "/markets", params={
        "series_ticker": series, "status": "settled", "limit": 100,
    })
    markets = result.get("markets", [])
    print(f"Settled markets: {len(markets)}")

    by_date = {}
    for m in markets:
        date_match = re.search(r"-(\d{2}[A-Z]{3}\d{2})-", m["ticker"])
        if date_match:
            by_date.setdefault(date_match.group(1), []).append(m)

    print(f"Trading days: {len(by_date)}")

    for date_key in sorted(by_date.keys()):
        day_markets = by_date[date_key]
        actual = infer_actual_temp(day_markets)
        if actual is None:
            continue

        forecast_std = 2.5
        day_trades = 0
        day_correct = 0
        day_pnl = 0.0

        for m in day_markets:
            stype, low, high = parse_strike(m["ticker"], m.get("rules_primary", ""))
            if stype == "unknown":
                continue

            result_val = m.get("result", "")
            last_price = float(m.get("last_price_dollars", "0") or "0")

            if last_price <= 0.01 or last_price >= 0.99:
                continue

            if stype == "above":
                model_p = 1 - norm.cdf(low, loc=actual, scale=forecast_std)
            elif stype == "below":
                model_p = norm.cdf(low, loc=actual, scale=forecast_std)
            elif stype == "between":
                model_p = norm.cdf(high, loc=actual, scale=forecast_std) - norm.cdf(low, loc=actual, scale=forecast_std)
            else:
                continue

            edge_yes = model_p - last_price
            edge_no = last_price - model_p

            if max(edge_yes, edge_no) < 0.10:
                continue

            if edge_yes > edge_no:
                side = "yes"
                entry = last_price
                correct = result_val == "yes"
            else:
                side = "no"
                entry = 1 - last_price
                correct = result_val == "no"

            pnl = (1.0 - entry) if correct else -entry

            day_trades += 1
            day_correct += int(correct)
            day_pnl += pnl
            total_trades += 1
            total_correct += int(correct)
            total_pnl += pnl

        if day_trades > 0:
            wr = day_correct / day_trades * 100
            print(f"  {date_key}: actual={actual:.0f}F | trades={day_trades} wins={day_correct} ({wr:.0f}%) pnl=${day_pnl:.2f}")

print(f"\n{SEP}")
print(f"  BACKTEST SUMMARY (forecast_std=2.5F, min_edge=10%)")
print(SEP)
print(f"Total trades: {total_trades}")
print(f"Wins: {total_correct} | Losses: {total_trades - total_correct}")
if total_trades > 0:
    print(f"Win rate: {total_correct / total_trades * 100:.1f}%")
    print(f"Total P&L: ${total_pnl:.2f} per contract")
    print(f"Avg P&L/trade: ${total_pnl / total_trades:.3f} per contract")
    print(f"At 10 contracts/trade: ${total_pnl * 10:.2f} total")
else:
    print("No trades found.")

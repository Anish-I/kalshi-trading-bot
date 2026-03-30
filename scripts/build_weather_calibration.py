"""
Build weather calibration artifact from settled markets + archived forecast snapshots.

This script measures actual trade-quality for the live weather model:
  - reconstruct model probabilities from the archived ensemble snapshot
  - reconstruct contemporaneous entry quotes from the market archive
  - compute bucketed trade win rates by city tier x strike type x edge bucket

Output: D:/kalshi-models/weather_calibration.json
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

from config.settings import CITY_TIERS, WEATHER_CITIES, settings
from engine.market_archive import load_archive
from engine.weather_calibration import edge_bucket_label
from kalshi.client import KalshiClient
from weather.historical_evaluator import HistoricalWeatherEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("weather_calibration")

SNAPSHOT_DIR = Path(settings.DATA_DIR) / "weather_snapshots"
OUTPUT_PATH = Path(settings.MODEL_DIR) / "weather_calibration.json"
MIN_BUCKET_SIZE = 5
SHRINKAGE_WEIGHT = 0.3
MIN_TRADE_WIN_RATE = 0.42
MAX_ARCHIVE_DELTA_S = 3600


def _get_city_tier(city_short: str) -> int:
    for tier, cities in CITY_TIERS.items():
        if city_short in cities:
            return tier
    return 3


def _parse_iso_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _load_snapshots(snapshot_dir: Path | None = None) -> dict[tuple[str, str], list[dict]]:
    snapshot_dir = snapshot_dir or SNAPSHOT_DIR
    snapshots: dict[tuple[str, str], list[dict]] = defaultdict(list)
    if not snapshot_dir.exists():
        return snapshots

    for path in sorted(snapshot_dir.glob("*.jsonl")):
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except Exception:
                continue
            key = (record.get("city_short", ""), record.get("target_date", ""))
            if not key[0] or not key[1]:
                continue
            snapshots[key].append(record)

    for key in snapshots:
        snapshots[key].sort(key=lambda rec: rec.get("issue_time_utc", ""))

    log.info("Loaded %d snapshot groups", len(snapshots))
    return snapshots


def _select_snapshot(candidates: list[dict], close_time: str) -> dict | None:
    if not candidates:
        return None

    close_dt = _parse_iso_ts(close_time)
    if close_dt is None:
        return candidates[-1]

    eligible = [
        rec for rec in candidates
        if (_parse_iso_ts(rec.get("issue_time_utc", "")) or close_dt) <= close_dt
    ]
    return eligible[-1] if eligible else candidates[-1]


def _parse_strike(ticker: str, rules: str = "") -> tuple[str, float, float]:
    strike_type, strike_value = HistoricalWeatherEvaluator.parse_strike(ticker, rules)
    if strike_type == "between":
        low = math.floor(strike_value)
        high = math.ceil(strike_value)
        if low == high:
            low = strike_value - 0.5
            high = strike_value + 0.5
        return ("between", float(low), float(high))
    if strike_type in {"above", "below"}:
        return (strike_type, strike_value, strike_value)
    return ("unknown", 0.0, 0.0)


def _normal_cdf(value: float, mean: float, std: float) -> float:
    std = max(std, 0.25)
    z = (value - mean) / (std * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def _model_probability_from_snapshot(snapshot: dict, strike_type: str, strike_low: float, strike_high: float) -> float | None:
    members = [float(value) for value in (snapshot.get("member_values_f") or []) if value is not None]
    if members:
        if strike_type == "above":
            return sum(1 for value in members if value > strike_low) / len(members)
        if strike_type == "below":
            return sum(1 for value in members if value < strike_low) / len(members)
        if strike_type == "between":
            return sum(1 for value in members if strike_low <= value <= strike_high) / len(members)
        return None

    mean = snapshot.get("bias_adjusted_mean_f", snapshot.get("mean_f"))
    std = snapshot.get("std_f", 0.0)
    try:
        mean = float(mean)
        std = float(std)
    except Exception:
        return None

    if strike_type == "above":
        return 1.0 - _normal_cdf(strike_low, mean, std)
    if strike_type == "below":
        return _normal_cdf(strike_low, mean, std)
    if strike_type == "between":
        return max(0.0, _normal_cdf(strike_high, mean, std) - _normal_cdf(strike_low, mean, std))
    return None


def _archive_series_for_market_type(market_type: str) -> str:
    return "KXLOW" if market_type == "low" else "KXHIGH"


def _lookup_archive_quote(
    archive_cache: dict[tuple[str, date], object],
    archive_series: str,
    ticker: str,
    issue_dt: datetime,
) -> dict | None:
    cache_key = (archive_series, issue_dt.date())
    df = archive_cache.get(cache_key)
    if df is None:
        df = load_archive(archive_series, start_date=issue_dt.date(), end_date=issue_dt.date())
        if not df.empty and "_timestamp_utc" not in df.columns:
            df = df.copy()
            df["_timestamp_utc"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        archive_cache[cache_key] = df

    if getattr(df, "empty", True):
        return None

    subset = df[df["ticker"] == ticker]
    if subset.empty:
        return None

    subset = subset.dropna(subset=["_timestamp_utc"]).copy()
    if subset.empty:
        return None

    deltas = (subset["_timestamp_utc"] - issue_dt).abs()
    nearest_idx = deltas.idxmin()
    nearest_delta_s = deltas.loc[nearest_idx].total_seconds()
    if nearest_delta_s > MAX_ARCHIVE_DELTA_S:
        return None

    return subset.loc[nearest_idx].to_dict()


def _fetch_settled_weather_markets(client: KalshiClient) -> list[dict]:
    settled = []
    checked = 0
    for city in WEATHER_CITIES:
        series = city["series_ticker"]
        try:
            data = client._request(
                "GET",
                "/markets",
                params={"series_ticker": series, "status": "closed", "limit": 200},
            )
            closed_markets = data.get("markets", [])
            for market in closed_markets:
                ticker = market.get("ticker", "")
                try:
                    detail = client.get_market(ticker)
                except Exception:
                    continue
                checked += 1
                if detail.get("status") not in ("finalized", "settled") or not detail.get("result"):
                    continue
                detail["_city_short"] = city["short"]
                detail["_city_tier"] = _get_city_tier(city["short"])
                detail["_market_type"] = city["type"]
                settled.append(detail)
        except Exception:
            log.warning("Failed to fetch markets for %s", series, exc_info=True)

    log.info("Checked %d closed weather markets, found %d finalized", checked, len(settled))
    return settled


def build_calibration(
    client: KalshiClient | None = None,
    snapshot_dir: Path | None = None,
    output_path: Path | None = None,
) -> dict | None:
    client = client or KalshiClient()
    snapshot_dir = snapshot_dir or SNAPSHOT_DIR
    output_path = output_path or OUTPUT_PATH

    snapshots = _load_snapshots(snapshot_dir)
    settled = _fetch_settled_weather_markets(client)

    if not settled:
        log.warning("No settled weather markets found. Cannot build calibration.")
        return None

    buckets: dict[str, list[dict]] = defaultdict(list)
    archive_cache: dict[tuple[str, date], object] = {}
    matched = 0
    unmatched = defaultdict(int)
    all_trade_results: list[bool] = []

    for market in settled:
        ticker = market.get("ticker", "")
        close_time = market.get("close_time", "")
        close_dt = _parse_iso_ts(close_time)
        if close_dt is None:
            unmatched["missing_close_time"] += 1
            continue

        target_date = (
            HistoricalWeatherEvaluator.parse_date_from_ticker(ticker)
            or close_time[:10]
        )
        city_short = market.get("_city_short", "")
        tier = market.get("_city_tier", 3)
        market_type = market.get("_market_type", "high")
        result = market.get("result", "").lower()
        if result not in {"yes", "no"}:
            unmatched["missing_result"] += 1
            continue

        snapshot = _select_snapshot(snapshots.get((city_short, target_date), []), close_time)
        if snapshot is None:
            unmatched["no_snapshot"] += 1
            continue

        issue_dt = _parse_iso_ts(snapshot.get("issue_time_utc", ""))
        if issue_dt is None:
            unmatched["bad_issue_time"] += 1
            continue

        strike_type, strike_low, strike_high = _parse_strike(
            ticker,
            market.get("rules_primary", "") or market.get("rules", ""),
        )
        if strike_type == "unknown":
            unmatched["unknown_strike"] += 1
            continue

        model_prob = _model_probability_from_snapshot(snapshot, strike_type, strike_low, strike_high)
        if model_prob is None:
            unmatched["no_model_prob"] += 1
            continue

        quote = _lookup_archive_quote(
            archive_cache,
            _archive_series_for_market_type(market_type),
            ticker,
            issue_dt,
        )
        if quote is None:
            unmatched["no_archive_quote"] += 1
            continue

        yes_ask = float(quote.get("yes_ask", 0) or 0)
        no_ask = float(quote.get("no_ask", 0) or 0)

        side = None
        entry_price_cents = 0
        edge = 0.0
        if model_prob > yes_ask and yes_ask > 0:
            side = "YES"
            entry_price_cents = int(round(yes_ask * 100))
            edge = model_prob - yes_ask
        elif (1.0 - model_prob) > no_ask and no_ask > 0:
            side = "NO"
            entry_price_cents = int(round(no_ask * 100))
            edge = (1.0 - model_prob) - no_ask
        else:
            unmatched["no_edge"] += 1
            continue

        edge_bucket = edge_bucket_label(edge)
        trade_won = (side == "YES" and result == "yes") or (side == "NO" and result == "no")
        bucket_key = f"tier{tier}_{strike_type}_{edge_bucket}"
        buckets[bucket_key].append({
            "trade_won": trade_won,
            "model_prob": round(model_prob, 6),
            "entry_price_cents": entry_price_cents,
            "edge": round(edge, 6),
            "side": side,
        })
        all_trade_results.append(trade_won)
        matched += 1

    if not all_trade_results:
        log.warning("No priced trade signals matched archive data. Cannot build calibration.")
        return None

    global_trade_win_rate = sum(1 for won in all_trade_results if won) / len(all_trade_results)

    calibration = {
        "version": "weather_v2",
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_markets": len(settled),
        "matched_markets": matched,
        "unmatched_markets": dict(unmatched),
        "global_trade_win_rate": round(global_trade_win_rate, 4),
        "min_bucket_size": MIN_BUCKET_SIZE,
        "shrinkage_weight": SHRINKAGE_WEIGHT,
        "min_trade_win_rate": MIN_TRADE_WIN_RATE,
        "buckets": {},
    }

    for bucket_key, records in sorted(buckets.items()):
        n = len(records)
        raw_trade_win_rate = sum(1 for rec in records if rec["trade_won"]) / n
        shrunk_trade_win_rate = (
            (1 - SHRINKAGE_WEIGHT) * raw_trade_win_rate + SHRINKAGE_WEIGHT * global_trade_win_rate
            if n >= MIN_BUCKET_SIZE
            else global_trade_win_rate
        )

        calibration["buckets"][bucket_key] = {
            "n_markets": n,
            "raw_trade_win_rate": round(raw_trade_win_rate, 4),
            "shrunk_trade_win_rate": round(shrunk_trade_win_rate, 4),
            "avg_model_prob": round(sum(rec["model_prob"] for rec in records) / n, 4),
            "avg_entry_price_cents": round(sum(rec["entry_price_cents"] for rec in records) / n, 2),
            "avg_edge": round(sum(rec["edge"] for rec in records) / n, 4),
            "sufficient_data": n >= MIN_BUCKET_SIZE,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    log.info(
        "Weather calibration saved to %s (%d matched trades, %d buckets)",
        output_path,
        matched,
        len(calibration["buckets"]),
    )
    return calibration


if __name__ == "__main__":
    build_calibration()

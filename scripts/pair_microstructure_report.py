"""
Phase 8: Offline pair microstructure report (read-only).

Aggregates pair-trade lifecycle evidence from:
  - OrderLedger parquet (D:/kalshi-data/order_ledger.parquet)
  - logs/crypto_combined.log (PAIR LIVE / PAIR SIM / orphan events)
  - D:/kalshi-data/combined_trader_state.json

Writes D:/kalshi-data/reports/pair_microstructure_{YYYY-MM-DD}.parquet with
per-(series, spread_bucket, hour_of_day) aggregates. Prints a stdout summary
of the worst (series, spread_bucket) combinations by net P&L.

Pure read-only — never mutates source files or trader configs.

Usage:
    python scripts/pair_microstructure_report.py --days 30 [--series KXBTC15M]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger("pair_microstructure_report")

DEFAULT_LEDGER_PATH = Path("D:/kalshi-data/order_ledger.parquet")
DEFAULT_LOG_PATH = Path("logs/crypto_combined.log")
DEFAULT_STATE_PATH = Path("D:/kalshi-data/combined_trader_state.json")
DEFAULT_REPORTS_DIR = Path("D:/kalshi-data/reports")

SPREAD_BUCKET_LABELS = ["<3c", "3-5c", "5-7c", "7-10c", ">10c"]

OUTPUT_COLUMNS = [
    "series",
    "spread_bucket",
    "hour_of_day",
    "trade_count",
    "fill_ratio",
    "orphan_rate",
    "avg_orphan_loss_cents",
    "avg_pair_pnl_cents",
    "net_pnl_cents",
    "sample_count",
]


def bucket_spread(spread_cents: float) -> str:
    """Bucket a spread (in cents) into one of the labeled buckets."""
    try:
        s = float(spread_cents)
    except (TypeError, ValueError):
        return "<3c"
    if s < 3:
        return "<3c"
    if s < 5:
        return "3-5c"
    if s < 7:
        return "5-7c"
    if s < 10:
        return "7-10c"
    return ">10c"


def extract_series(ticker: str) -> str:
    """Extract series prefix from a Kalshi ticker, e.g. 'KXBTC15M-...' -> 'KXBTC15M'."""
    if not isinstance(ticker, str) or not ticker:
        return ""
    return ticker.split("-", 1)[0]


def load_ledger(ledger_path: Path) -> pd.DataFrame:
    """Load the OrderLedger parquet. Returns empty DataFrame if missing."""
    if not ledger_path.exists():
        logger.warning("Ledger parquet not found at %s", ledger_path)
        return pd.DataFrame()
    try:
        df = pd.read_parquet(ledger_path)
    except Exception as exc:
        logger.warning("Failed to read ledger parquet: %s", exc)
        return pd.DataFrame()
    return df


def filter_pair_rows(df: pd.DataFrame, days: int, series_filter: str | None) -> pd.DataFrame:
    """Return only pair-strategy rows within the last N days (and optional series)."""
    if df.empty:
        return df
    if "signal_ref" not in df.columns and "strategy" not in df.columns:
        return df.iloc[0:0]

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    def _is_pair(row) -> bool:
        strategy = str(row.get("strategy", "") or "")
        signal = str(row.get("signal_ref", "") or "")
        market = str(row.get("market_type", "") or "")
        return (
            "pair" in strategy.lower()
            or "pair" in signal.lower()
            or "pair" in market.lower()
        )

    mask = df.apply(_is_pair, axis=1)
    out = df[mask].copy()

    if "created_at" in out.columns:
        out["_created_dt"] = pd.to_datetime(out["created_at"], errors="coerce", utc=True)
        out = out[out["_created_dt"].notna() & (out["_created_dt"] >= cutoff)]

    if series_filter:
        out["_series"] = out["ticker"].apply(extract_series)
        out = out[out["_series"] == series_filter]

    return out


# Log parsing patterns — tolerant substring matches.
_PAIR_EVENT_PATTERNS = [
    ("PAIR LIVE", "pair_live"),
    ("PAIR SIM", "pair_sim"),
    ("PAIR_LIVE_ORPHAN", "orphan"),
    ("PAIR ORPHANED", "orphan"),
]

_SPREAD_RE = re.compile(r"spread[=:\s]+(\d+(?:\.\d+)?)\s*c?", re.IGNORECASE)
_TICKER_RE = re.compile(r"([A-Z]{2,}[A-Z0-9]*)-\S+")
_PNL_RE = re.compile(r"pnl[=:\s]+(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_log_events(log_path: Path) -> list[dict]:
    """Parse pair lifecycle events from the combined log. Tolerant of malformed lines."""
    if not log_path.exists():
        logger.info("Log file not found at %s — skipping log enrichment", log_path)
        return []

    events: list[dict] = []
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, start=1):
                try:
                    kind = None
                    for needle, tag in _PAIR_EVENT_PATTERNS:
                        if needle in line:
                            kind = tag
                            break
                    if kind is None:
                        continue

                    spread_m = _SPREAD_RE.search(line)
                    ticker_m = _TICKER_RE.search(line)
                    pnl_m = _PNL_RE.search(line)

                    events.append(
                        {
                            "kind": kind,
                            "spread_cents": float(spread_m.group(1)) if spread_m else None,
                            "ticker": ticker_m.group(0) if ticker_m else "",
                            "series": ticker_m.group(1) if ticker_m else "",
                            "pnl_cents": float(pnl_m.group(1)) if pnl_m else None,
                            "raw": line.strip()[:200],
                        }
                    )
                except Exception as exc:
                    logger.warning("Skipping malformed log line %d: %s", lineno, exc)
                    continue
    except Exception as exc:
        logger.warning("Failed to read log file %s: %s", log_path, exc)
        return []

    return events


def load_state_snapshot(state_path: Path) -> dict:
    """Load combined_trader_state.json snapshot for orphan counters."""
    if not state_path.exists():
        return {}
    try:
        with state_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Failed to read state json %s: %s", state_path, exc)
        return {}


def build_report(
    pair_df: pd.DataFrame,
    log_events: list[dict],
) -> pd.DataFrame:
    """Aggregate pair rows into the microstructure report DataFrame."""
    if pair_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pair_df.copy()
    df["series"] = df["ticker"].apply(extract_series)

    # Derive spread: try bid/ask columns if present, else fall back to slippage-derived estimate,
    # else bucket as <3c by default.
    def _row_spread(row) -> float:
        for key in ("spread_cents", "mid_spread_cents", "book_spread_cents"):
            if key in row and pd.notna(row[key]):
                try:
                    return float(row[key])
                except (TypeError, ValueError):
                    pass
        slip = row.get("slippage_cents", 0)
        try:
            return abs(float(slip)) * 2 if slip else 0.0
        except (TypeError, ValueError):
            return 0.0

    df["_spread_cents"] = df.apply(_row_spread, axis=1)
    df["spread_bucket"] = df["_spread_cents"].apply(bucket_spread)

    if "_created_dt" in df.columns:
        df["hour_of_day"] = df["_created_dt"].dt.hour.fillna(0).astype(int)
    else:
        dt = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True)
        df["hour_of_day"] = dt.dt.hour.fillna(0).astype(int)

    df["_filled"] = df["status"].isin(["filled", "settled", "simulated"]).astype(int)
    df["_pnl"] = pd.to_numeric(df.get("pnl_cents", 0), errors="coerce").fillna(0)

    # Orphan detection: look for "orphan" in signal_ref or settlement_result == "orphan".
    def _is_orphan(row) -> bool:
        sig = str(row.get("signal_ref", "") or "").lower()
        res = str(row.get("settlement_result", "") or "").lower()
        return "orphan" in sig or "orphan" in res

    df["_orphan"] = df.apply(_is_orphan, axis=1).astype(int)

    # Enrich orphan counts from log events per series.
    orphan_by_series: dict[str, int] = {}
    for ev in log_events:
        if ev["kind"] == "orphan" and ev["series"]:
            orphan_by_series[ev["series"]] = orphan_by_series.get(ev["series"], 0) + 1

    grouped = (
        df.groupby(["series", "spread_bucket", "hour_of_day"], dropna=False)
        .agg(
            trade_count=("ticker", "count"),
            filled_sum=("_filled", "sum"),
            orphan_sum=("_orphan", "sum"),
            orphan_loss_sum=("_pnl", lambda s: float(s[df.loc[s.index, "_orphan"] == 1].sum())),
            pnl_sum=("_pnl", "sum"),
        )
        .reset_index()
    )

    grouped["fill_ratio"] = grouped.apply(
        lambda r: (r["filled_sum"] / r["trade_count"]) if r["trade_count"] else 0.0, axis=1
    )
    grouped["orphan_rate"] = grouped.apply(
        lambda r: (r["orphan_sum"] / r["trade_count"]) if r["trade_count"] else 0.0, axis=1
    )
    grouped["avg_orphan_loss_cents"] = grouped.apply(
        lambda r: (r["orphan_loss_sum"] / r["orphan_sum"]) if r["orphan_sum"] else 0.0, axis=1
    )
    grouped["avg_pair_pnl_cents"] = grouped.apply(
        lambda r: (r["pnl_sum"] / r["trade_count"]) if r["trade_count"] else 0.0, axis=1
    )
    grouped["net_pnl_cents"] = grouped["pnl_sum"]
    grouped["sample_count"] = grouped["trade_count"]

    return grouped[OUTPUT_COLUMNS]


def print_summary(report: pd.DataFrame) -> None:
    """Print top-10 worst (series, spread_bucket) combos by net P&L per contract."""
    if report.empty:
        print("No pair data found — nothing to summarize.")
        return

    agg = (
        report.groupby(["series", "spread_bucket"], dropna=False)
        .agg(
            trades=("sample_count", "sum"),
            net_pnl_cents=("net_pnl_cents", "sum"),
            orphan_rate=("orphan_rate", "mean"),
            fill_ratio=("fill_ratio", "mean"),
        )
        .reset_index()
    )
    agg["net_pnl_per_contract"] = agg.apply(
        lambda r: (r["net_pnl_cents"] / r["trades"]) if r["trades"] else 0.0, axis=1
    )
    worst = agg.sort_values("net_pnl_per_contract").head(10)

    print("\n=== Top 10 worst (series, spread_bucket) by net P&L per contract ===")
    print(worst.to_string(index=False))
    print()


def run(
    days: int,
    series_filter: str | None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> int:
    """Main entrypoint. Returns an exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = load_ledger(ledger_path)
    pair_df = filter_pair_rows(df, days=days, series_filter=series_filter)

    if pair_df.empty:
        print("No pair data found — nothing to analyze")
        return 0

    log_events = parse_log_events(log_path)
    _ = load_state_snapshot(state_path)  # snapshot loaded for future enrichment

    report = build_report(pair_df, log_events)

    total_pairs = int(len(pair_df))
    filled = int(pair_df["status"].isin(["filled", "settled", "simulated"]).sum())
    completion_rate = (filled / total_pairs) if total_pairs else 0.0
    orphan_rows = int(
        pair_df.apply(
            lambda r: "orphan" in str(r.get("signal_ref", "") or "").lower()
            or "orphan" in str(r.get("settlement_result", "") or "").lower(),
            axis=1,
        ).sum()
    )
    orphan_rate = (orphan_rows / total_pairs) if total_pairs else 0.0
    total_pnl = int(pd.to_numeric(pair_df.get("pnl_cents", 0), errors="coerce").fillna(0).sum())

    logger.info(
        "PAIR MICROSTRUCTURE: pairs=%d completion_rate=%.3f orphan_rate=%.3f total_pnl_cents=%d",
        total_pairs,
        completion_rate,
        orphan_rate,
        total_pnl,
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = reports_dir / f"pair_microstructure_{date_str}.parquet"
    report.to_parquet(out_path, index=False)
    logger.info("Wrote report to %s (%d rows)", out_path, len(report))

    print_summary(report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline pair microstructure report (read-only)")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days (default 30)")
    parser.add_argument("--series", type=str, default=None, help="Optional series filter, e.g. KXBTC15M")
    args = parser.parse_args(argv)
    return run(days=args.days, series_filter=args.series)


if __name__ == "__main__":
    sys.exit(main())

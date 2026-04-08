"""
Kalshi Trade History Backfill — paginates /markets/trades and archives results.

CLI:
    python scripts/backfill_kalshi_trades.py [--series KXBTC15M] [--days 2]

Writes to D:/kalshi-data/market_archive/YYYY-MM-DD/{series}_trades.parquet
(separate from quote snapshots in {series}.parquet).

Append-only: existing rows are merged on trade_id, dedup on trade_id.
Cursor state lives in D:/kalshi-data/trade_archiver_state.json.

Safe to re-run: dedup is by trade_id. Run manually — not a long-running daemon.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd

from config.settings import settings
from kalshi.client import KalshiClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("kalshi_trade_backfill.log"),
    ],
)
log = logging.getLogger("kalshi_trade_backfill")

ARCHIVE_DIR = Path(settings.DATA_DIR) / "market_archive"
STATE_FILE = Path(settings.DATA_DIR) / "trade_archiver_state.json"

DEFAULT_SERIES = [
    "KXBTC15M",
    "KXETH15M",
    "KXSOL15M",
    "KXXRP15M",
    "KXHYPE15M",
    "KXDOGE15M",
]

PAGE_LIMIT = 1000
SLEEP_BETWEEN_PAGES_S = 0.1
SLEEP_BETWEEN_SERIES_S = 0.5


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _list_tickers_for_series(client: KalshiClient, series: str) -> list[str]:
    """Expand a series ticker into individual market tickers (open + settled)."""
    tickers: set[str] = set()
    for status in ("open", "settled", "closed"):
        cursor: str | None = None
        while True:
            params = {
                "series_ticker": series,
                "status": status,
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            try:
                data = client._request("GET", "/markets", params=params)
            except Exception:
                log.warning("Failed to list %s markets (status=%s)", series, status, exc_info=True)
                break
            for mkt in data.get("markets", []) or []:
                t = mkt.get("ticker")
                if t:
                    tickers.add(t)
            cursor = data.get("cursor") or None
            if not cursor:
                break
            time.sleep(SLEEP_BETWEEN_PAGES_S)
    return sorted(tickers)


def _fetch_trades_for_ticker(
    client: KalshiClient,
    ticker: str,
    min_ts: int | None,
) -> list[dict]:
    """Paginate /markets/trades for one ticker."""
    out: list[dict] = []
    cursor: str | None = None
    while True:
        params: dict = {"ticker": ticker, "limit": PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor
        if min_ts is not None:
            params["min_ts"] = min_ts
        try:
            data = client._request("GET", "/markets/trades", params=params)
        except Exception:
            log.warning("Failed to fetch trades for %s", ticker, exc_info=True)
            break
        trades = data.get("trades") or []
        if not trades:
            break
        out.extend(trades)
        cursor = data.get("cursor") or None
        if not cursor:
            break
        time.sleep(SLEEP_BETWEEN_PAGES_S)
    return out


REQUIRED_COLS = [
    "trade_id",
    "ticker",
    "created_time",
    "yes_price_cents",
    "no_price_cents",
    "count",
    "taker_side",
]


def _normalize_trade(raw: dict) -> dict:
    """Normalize a raw Kalshi trade to include our canonical columns.

    Preserves all original fields as-is, plus adds canonical
    yes_price_cents / no_price_cents columns (Kalshi returns yes_price / no_price
    already in integer cents).
    """
    rec = dict(raw)
    if "yes_price_cents" not in rec:
        rec["yes_price_cents"] = rec.get("yes_price")
    if "no_price_cents" not in rec:
        rec["no_price_cents"] = rec.get("no_price")
    return rec


def _write_trades(series: str, trades: list[dict]) -> int:
    """Group trades by date and append to per-day parquet files. Returns rows written."""
    if not trades:
        return 0

    rows: list[dict] = []
    for t in trades:
        rows.append(_normalize_trade(t))

    df = pd.DataFrame(rows)

    # Parse created_time for per-day bucketing
    if "created_time" not in df.columns:
        log.error("Trade response missing created_time — unexpected schema. Sample: %s", trades[0])
        return 0

    df["_dt"] = pd.to_datetime(df["created_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["_dt"])
    df["_date"] = df["_dt"].dt.date.astype(str)

    total_written = 0
    for day, group in df.groupby("_date"):
        out_dir = ARCHIVE_DIR / day
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{series}_trades.parquet"

        to_write = group.drop(columns=["_dt", "_date"])
        if path.exists():
            try:
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, to_write], ignore_index=True)
                if "trade_id" in combined.columns:
                    combined = combined.drop_duplicates(subset=["trade_id"], keep="last")
                to_write = combined
            except Exception:
                log.warning("Could not read existing %s; overwriting", path, exc_info=True)

        tmp = path.with_suffix(".tmp")
        to_write.to_parquet(tmp, index=False)
        tmp.replace(path)
        total_written += len(group)

    return total_written


def backfill_series(
    client: KalshiClient,
    series: str,
    days: int,
    state: dict,
) -> int:
    log.info("Backfilling series %s (%d days)", series, days)

    tickers = _list_tickers_for_series(client, series)
    if not tickers:
        log.warning("No tickers found for series %s — skipping", series)
        return 0
    log.info("  found %d tickers", len(tickers))

    min_ts = int(time.time()) - days * 86400
    total_new = 0
    series_state = state.setdefault(series, {})

    for i, ticker in enumerate(tickers, 1):
        trades = _fetch_trades_for_ticker(client, ticker, min_ts=min_ts)
        if trades:
            written = _write_trades(series, trades)
            total_new += written
            # Cursor: remember the most recent trade_id we saw
            last_id = trades[0].get("trade_id")
            if last_id:
                series_state["last_trade_id"] = last_id
            series_state["last_run"] = datetime.now(timezone.utc).isoformat()
        log.info("  [%d/%d] %s: %d trades", i, len(tickers), ticker, len(trades))
        time.sleep(SLEEP_BETWEEN_PAGES_S)

    log.info("Series %s done: %d rows written", series, total_new)
    return total_new


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill Kalshi trade history")
    parser.add_argument("--series", action="append", help="Series to backfill (repeatable)")
    parser.add_argument("--days", type=int, default=2, help="Days of history (default 2)")
    args = parser.parse_args()

    series_list = args.series or DEFAULT_SERIES

    log.info("=" * 60)
    log.info("  Kalshi Trade History Backfill")
    log.info("  Series: %s", ",".join(series_list))
    log.info("  Days:   %d", args.days)
    log.info("  Output: %s", ARCHIVE_DIR)
    log.info("=" * 60)

    client = KalshiClient()
    state = _load_state()
    grand_total = 0

    for series in series_list:
        try:
            grand_total += backfill_series(client, series, args.days, state)
        except Exception:
            log.error("Series %s failed", series, exc_info=True)
        finally:
            _save_state(state)
            time.sleep(SLEEP_BETWEEN_SERIES_S)

    log.info("Backfill complete: %d total rows written", grand_total)
    return 0


if __name__ == "__main__":
    sys.exit(main())

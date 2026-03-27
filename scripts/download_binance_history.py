#!/usr/bin/env python3
"""Download 14 days of BTCUSDT 1-minute klines from Binance public data (data.binance.vision)."""
import sys
sys.path.insert(0, ".")

import io
import zipfile
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m"
OUTPUT_DIR = Path("D:/kalshi-data/binance_history")
OUTPUT_FILE = OUTPUT_DIR / "klines_90d.parquet"
DAYS = 90

CSV_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def download_day(client: httpx.Client, date_str: str) -> pd.DataFrame | None:
    """Download and parse one day's kline ZIP from Binance."""
    filename = f"BTCUSDT-1m-{date_str}.zip"
    url = f"{BASE_URL}/{filename}"
    log.info(f"Downloading {url}")

    try:
        resp = client.get(url, follow_redirects=True, timeout=60)
        if resp.status_code == 404:
            log.warning(f"  {date_str}: not found (404), skipping")
            return None
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        log.warning(f"  {date_str}: HTTP {e.response.status_code}, skipping")
        return None
    except httpx.RequestError as e:
        log.warning(f"  {date_str}: request error ({e}), skipping")
        return None

    # Extract CSV from ZIP
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            log.warning(f"  {date_str}: no CSV in ZIP, skipping")
            return None
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f, header=None, names=CSV_COLUMNS)

    log.info(f"  {date_str}: {len(df)} rows")
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate date range: last 14 complete days (excluding today)
    today = datetime.now(timezone.utc).date()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, DAYS + 1)]
    dates.reverse()  # chronological order

    log.info(f"Downloading {len(dates)} days: {dates[0]} to {dates[-1]}")

    frames = []
    with httpx.Client() as client:
        for date_str in dates:
            df = download_day(client, date_str)
            if df is not None:
                frames.append(df)

    if not frames:
        log.error("No data downloaded. Exiting.")
        sys.exit(1)

    raw = pd.concat(frames, ignore_index=True)
    log.info(f"Combined raw: {len(raw)} rows")

    # Build clean DataFrame
    result = pd.DataFrame()
    # Binance uses microseconds (not milliseconds) for open_time
    open_time = raw["open_time"].astype(float)
    if open_time.iloc[0] > 1e15:  # microseconds
        open_time = open_time // 1000  # convert to ms
    result["timestamp"] = pd.to_datetime(open_time, unit="ms", utc=True)
    result["open"] = raw["open"].astype(float)
    result["high"] = raw["high"].astype(float)
    result["low"] = raw["low"].astype(float)
    result["close"] = raw["close"].astype(float)
    result["volume"] = raw["volume"].astype(float)
    result["buy_volume"] = raw["taker_buy_volume"].astype(float)
    result["sell_volume"] = result["volume"] - result["buy_volume"]
    result["trade_count"] = raw["count"].astype(int)

    # Sort and deduplicate
    result = result.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    # Save
    result.to_parquet(OUTPUT_FILE, index=False)
    log.info(f"Saved to {OUTPUT_FILE}")

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  Rows:       {len(result):,}")
    print(f"  Date range: {result['timestamp'].min()} -> {result['timestamp'].max()}")
    print(f"  Days:       {len(frames)} of {DAYS} downloaded")
    print(f"  File:       {OUTPUT_FILE}")
    print(f"  Size:       {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    print("\nSample (first 5 rows):")
    print(result.head().to_string(index=False))
    print("\nSample (last 5 rows):")
    print(result.tail().to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Kalshi Market Data Archiver — polls and archives market quotes + orderbook.

Runs as a background process alongside the collector.
Archives to D:/kalshi-data/market_archive/YYYY-MM-DD/{series}.parquet

Checkpoint-based: saves cursor/timestamp state for resumability.
"""
import json
import logging
import sys
import time
from datetime import datetime, timezone, date
from pathlib import Path

sys.path.insert(0, ".")

from engine.process_lock import ProcessLock
from kalshi.client import KalshiClient
from config.settings import settings

# Lock
_lock = ProcessLock("kalshi_archiver")
_lock.kill_existing()
if not _lock.acquire():
    print("Another archiver is running. Exiting.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("kalshi_archiver.log"),
    ],
)
log = logging.getLogger("kalshi_archiver")

ARCHIVE_DIR = Path(settings.DATA_DIR) / "market_archive"
STATE_FILE = Path(settings.DATA_DIR) / "archiver_state.json"

# Series to archive
CRYPTO_SERIES = ["KXBTC15M"]
WEATHER_SERIES_PREFIXES = [
    "KXHIGH", "KXLOW"
]

# Intervals
CRYPTO_INTERVAL_S = 30
WEATHER_INTERVAL_S = 300  # 5 min (denser for production weather strategy)
ORDERBOOK_DEPTH = 5


def _today_dir() -> Path:
    d = ARCHIVE_DIR / date.today().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"last_crypto_ts": 0, "last_weather_ts": 0, "records_today": 0}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state))


def _append_records(series: str, records: list[dict]) -> None:
    """Append quote records to a daily parquet file."""
    import pandas as pd

    if not records:
        return

    out_dir = _today_dir()
    path = out_dir / f"{series}.parquet"

    df_new = pd.DataFrame(records)

    if path.exists():
        try:
            df_existing = pd.read_parquet(path)
            df_new = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            pass

    tmp = path.with_suffix(".tmp")
    df_new.to_parquet(tmp, index=False)
    tmp.replace(path)


def archive_crypto(client: KalshiClient) -> int:
    """Archive crypto market quotes + orderbooks. Returns record count."""
    ts = datetime.now(timezone.utc).isoformat()
    count = 0

    for series in CRYPTO_SERIES:
        try:
            data = client._request(
                "GET", "/markets",
                params={"series_ticker": series, "status": "open", "limit": 20},
            )
            markets = data.get("markets", [])
        except Exception:
            log.warning("Failed to fetch %s markets", series, exc_info=True)
            continue

        records = []
        for mkt in markets:
            ticker = mkt.get("ticker", "")
            record = {
                "timestamp": ts,
                "ticker": ticker,
                "status": mkt.get("status", ""),
                "yes_bid": float(mkt.get("yes_bid", 0) or 0),
                "yes_ask": float(mkt.get("yes_ask", 0) or 0),
                "no_bid": float(mkt.get("no_bid", 0) or 0),
                "no_ask": float(mkt.get("no_ask", 0) or 0),
                "volume": int(mkt.get("volume", 0) or 0),
                "open_interest": int(mkt.get("open_interest", 0) or 0),
                "close_time": mkt.get("close_time", ""),
            }

            # Fetch orderbook — Kalshi returns orderbook_fp with yes_dollars/no_dollars
            try:
                ob = client.get_orderbook(ticker, depth=ORDERBOOK_DEPTH)
                ob_fp = ob.get("orderbook_fp", ob.get("orderbook", {}))
                yes_levels = ob_fp.get("yes_dollars", ob_fp.get("yes", []))
                no_levels = ob_fp.get("no_dollars", ob_fp.get("no", []))

                # Top-of-book for quick pair analysis
                if yes_levels:
                    record["yes_ask"] = float(yes_levels[0][0])
                if no_levels:
                    record["no_ask"] = float(no_levels[0][0])
                if yes_levels and no_levels:
                    record["pair_cost"] = float(yes_levels[0][0]) + float(no_levels[0][0])

                for i in range(ORDERBOOK_DEPTH):
                    record[f"yes_ask_{i}_price"] = int(float(yes_levels[i][0]) * 100) if i < len(yes_levels) else 0
                    record[f"yes_ask_{i}_qty"] = int(float(yes_levels[i][1])) if i < len(yes_levels) else 0
                    record[f"no_ask_{i}_price"] = int(float(no_levels[i][0]) * 100) if i < len(no_levels) else 0
                    record[f"no_ask_{i}_qty"] = int(float(no_levels[i][1])) if i < len(no_levels) else 0
            except Exception:
                log.debug("Orderbook fetch failed for %s", ticker, exc_info=True)

            records.append(record)
            count += 1

        if records:
            _append_records(series, records)

    return count


def archive_weather(client: KalshiClient) -> int:
    """Archive weather market quotes. Returns record count."""
    ts = datetime.now(timezone.utc).isoformat()
    count = 0

    for prefix in WEATHER_SERIES_PREFIXES:
        try:
            # Discover weather series by prefix
            data = client._request(
                "GET", "/markets",
                params={"status": "open", "limit": 200},
            )
            markets = [m for m in data.get("markets", [])
                       if m.get("ticker", "").startswith(prefix)]
        except Exception:
            log.warning("Failed to fetch weather markets for %s", prefix, exc_info=True)
            continue

        records = []
        for mkt in markets:
            ticker = mkt.get("ticker", "")
            records.append({
                "timestamp": ts,
                "ticker": ticker,
                "status": mkt.get("status", ""),
                "yes_bid": float(mkt.get("yes_bid", 0) or 0),
                "yes_ask": float(mkt.get("yes_ask", 0) or 0),
                "no_bid": float(mkt.get("no_bid", 0) or 0),
                "no_ask": float(mkt.get("no_ask", 0) or 0),
                "volume": int(mkt.get("volume", 0) or 0),
                "open_interest": int(mkt.get("open_interest", 0) or 0),
                "close_time": mkt.get("close_time", ""),
                "result": mkt.get("result", ""),
            })
            count += 1

        if records:
            _append_records(prefix, records)

    return count


def main():
    log.info("=" * 60)
    log.info("  Kalshi Market Data Archiver")
    log.info("  Crypto: every %ds | Weather: every %ds", CRYPTO_INTERVAL_S, WEATHER_INTERVAL_S)
    log.info("  Archive: %s", ARCHIVE_DIR)
    log.info("=" * 60)

    client = KalshiClient()
    state = _load_state()
    total_archived = 0

    while True:
        try:
            now = time.time()

            # Crypto — every 30s
            if now - state["last_crypto_ts"] >= CRYPTO_INTERVAL_S:
                n = archive_crypto(client)
                state["last_crypto_ts"] = now
                state["records_today"] = state.get("records_today", 0) + n
                total_archived += n
                if n:
                    log.info("Archived %d crypto market snapshots (total today: %d)",
                             n, state["records_today"])

            # Weather — every 15 min
            if now - state["last_weather_ts"] >= WEATHER_INTERVAL_S:
                n = archive_weather(client)
                state["last_weather_ts"] = now
                state["records_today"] = state.get("records_today", 0) + n
                total_archived += n
                if n:
                    log.info("Archived %d weather market snapshots (total today: %d)",
                             n, state["records_today"])

            # Reset daily counter at midnight
            if datetime.now().hour == 0 and datetime.now().minute == 0:
                state["records_today"] = 0

            _save_state(state)
            time.sleep(min(CRYPTO_INTERVAL_S, 10))

        except KeyboardInterrupt:
            log.info("Archiver stopped. Total records: %d", total_archived)
            break
        except Exception:
            log.error("Archiver error, continuing...", exc_info=True)
            time.sleep(10)


if __name__ == "__main__":
    main()

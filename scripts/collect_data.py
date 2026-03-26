"""
Live data collector: streams Coinbase BTC-USD trades + orderbook,
aggregates into 5s/1m bars, computes features, saves to parquet.

Run: python scripts/collect_data.py
Stop: Ctrl+C (saves final batch before exit)
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

# === SINGLETON LOCK — prevents duplicate instances ===
from engine.process_lock import ProcessLock
_lock = ProcessLock("collect_data")
_lock.kill_existing()
if not _lock.acquire():
    print("FATAL: Another collect_data is already running. Exiting.")
    sys.exit(1)

import pandas as pd

from config.settings import settings

# Use authenticated WebSocket if CDP key available, else fallback to unauthenticated
_cdp_key = getattr(settings, "COINBASE_CDP_KEY_ID", "") or ""
_cdp_secret = getattr(settings, "COINBASE_CDP_PRIVATE_KEY", "") or ""

if _cdp_key and _cdp_secret:
    from data.coinbase_auth_ws import CoinbaseAuthCollector as _Collector
    _USE_AUTH = True
else:
    from data.coinbase_ws import CoinbaseCollector as _Collector  # type: ignore[assignment]
    _USE_AUTH = False

from data.bar_aggregator import BarAggregator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("collect_data.log"),
    ],
)
log = logging.getLogger("collector")

# Globals
aggregator = BarAggregator(bar_interval_seconds=5)
latest_orderbook: dict = {"bids": [], "asks": []}
trade_count = 0
depth_count = 0
bar_5s_count = 0
bar_1m_count = 0
last_save_time = time.time()
SAVE_INTERVAL = 300  # Save every 5 minutes


async def on_trade(trade: dict) -> None:
    global trade_count, bar_5s_count, bar_1m_count

    trade_count += 1
    prev_1m_len = len(aggregator.bars_1m)
    completed_bar = aggregator.add_trade(
        price=trade["price"],
        qty=trade["qty"],
        is_buyer_maker=trade["is_buyer_maker"],
        timestamp_ms=trade["timestamp_ms"],
    )

    if completed_bar is not None:
        bar_5s_count += 1

        if len(aggregator.bars_1m) > prev_1m_len:
            bar_1m_count += 1

        # Periodic status (no legacy feature computation)
        if bar_5s_count % 12 == 0:
            log.info(
                "Status: %d trades, %d 5s bars, %d 1m bars, ob=%d/%d",
                trade_count, bar_5s_count, bar_1m_count,
                len(latest_orderbook.get("bids", [])),
                len(latest_orderbook.get("asks", [])),
            )

    # Periodic save
    global last_save_time
    if time.time() - last_save_time > SAVE_INTERVAL:
        save_data()
        last_save_time = time.time()


async def on_depth(data: dict) -> None:
    global latest_orderbook, depth_count
    depth_count += 1

    # Handle both formats: orderbook dict or ticker dict
    if "bids" in data and "asks" in data:
        # Orderbook format from unauthenticated collector
        latest_orderbook = data
    elif "best_bid" in data:
        # Ticker format from authenticated collector — build synthetic orderbook
        latest_orderbook = {
            "bids": [[data["best_bid"], 1.0]],
            "asks": [[data["best_ask"], 1.0]],
        }
    else:
        return

    # Orderbook stored for reference; honest features computed by trader on demand


async def on_kline(kline: dict) -> None:
    log.info(
        "1m candle closed: O=%.2f H=%.2f L=%.2f C=%.2f V=%.4f",
        kline["open"], kline["high"], kline["low"], kline["close"], kline["volume"],
    )


def save_data() -> None:
    """Save accumulated data to parquet."""
    data_dir = Path(settings.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).strftime("%Y%m%d")

    # Save 5s bars
    if aggregator.bars_5s:
        bars_df = aggregator.get_bars_5s_df()
        path = data_dir / "bars_5s" / f"{today}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            bars_df = pd.concat([existing, bars_df]).drop_duplicates(
                subset=["timestamp"]
            ).sort_values("timestamp")
        bars_df.to_parquet(path, index=False)
        log.info("Saved %d 5s bars to %s", len(bars_df), path)

    # Save 1m bars
    if aggregator.bars_1m:
        bars_df = aggregator.get_bars_1m_df()
        path = data_dir / "bars_1m" / f"{today}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            bars_df = pd.concat([existing, bars_df]).drop_duplicates(
                subset=["timestamp"]
            ).sort_values("timestamp")
        bars_df.to_parquet(path, index=False)
        log.info("Saved %d 1m bars to %s", len(bars_df), path)

    # Legacy feature writing removed — trader computes honest features from bars


async def main() -> None:
    log.info("=" * 50)
    log.info("LIVE DATA COLLECTOR - Coinbase BTC-USD")
    log.info("Saving to: %s", settings.DATA_DIR)
    log.info("=" * 50)

    if settings.COINBASE_CDP_KEY_ID and settings.COINBASE_CDP_PRIVATE_KEY:
        log.info("Using AUTHENTICATED Coinbase WebSocket (CDP key)")
        from data.coinbase_auth_ws import CoinbaseAuthCollector
        collector = CoinbaseAuthCollector(
            key_id=settings.COINBASE_CDP_KEY_ID,
            private_key_b64=settings.COINBASE_CDP_PRIVATE_KEY,
            symbol="BTC-USD",
            on_trade=on_trade,
            on_ticker=on_depth,
            on_kline=on_kline,
        )
    else:
        log.info("Using unauthenticated Coinbase WebSocket (REST orderbook polling)")
        from data.coinbase_ws import CoinbaseCollector
        collector = CoinbaseCollector(
            symbol="BTC-USD",
            on_trade=on_trade,
            on_depth=on_depth,
            on_kline=on_kline,
        )

    # Always poll REST orderbook for 20-level depth (WS only gives L1)
    import httpx as _httpx

    async def poll_orderbook():
        async with _httpx.AsyncClient(timeout=10) as client:
            while True:
                try:
                    resp = await client.get(
                        "https://api.exchange.coinbase.com/products/BTC-USD/book",
                        params={"level": 2},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        ob = {
                            "bids": [[float(b[0]), float(b[1])] for b in data.get("bids", [])[:20]],
                            "asks": [[float(a[0]), float(a[1])] for a in data.get("asks", [])[:20]],
                        }
                        await on_depth(ob)
                except Exception:
                    pass
                await asyncio.sleep(3)

    asyncio.get_event_loop().create_task(poll_orderbook())

    try:
        await collector.start()
    except KeyboardInterrupt:
        pass
    finally:
        await collector.stop()
        log.info("Saving final data...")
        save_data()
        log.info(
            "Final: %d trades, %d 5s bars, %d 1m bars, %d ob updates",
            trade_count, bar_5s_count, bar_1m_count, depth_count,
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Interrupted - saving...")
        save_data()

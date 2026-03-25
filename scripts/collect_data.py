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

import numpy as np
import pandas as pd

from config.settings import settings
from data.coinbase_ws import CoinbaseCollector
from data.bar_aggregator import BarAggregator
from features.pipeline import FeaturePipeline

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
pipeline = FeaturePipeline()
latest_orderbook: dict = {"bids": [], "asks": []}
feature_rows: list[dict] = []
trade_count = 0
depth_count = 0
bar_5s_count = 0
bar_1m_count = 0
last_save_time = time.time()
SAVE_INTERVAL = 300  # Save every 5 minutes


async def on_trade(trade: dict) -> None:
    global trade_count, bar_5s_count, bar_1m_count

    trade_count += 1
    completed_bar = aggregator.add_trade(
        price=trade["price"],
        qty=trade["qty"],
        is_buyer_maker=trade["is_buyer_maker"],
        timestamp_ms=trade["timestamp_ms"],
    )

    if completed_bar is not None:
        bar_5s_count += 1

        # Check for completed 1m bar
        completed_1m = aggregator._check_1m_bar(completed_bar)
        if completed_1m is not None:
            bar_1m_count += 1
            pipeline.update_1m_bar(completed_1m)

        # Compute features every 5s bar
        if len(aggregator.bars_5s) >= 24 and len(aggregator.bars_1m) >= 30:
            bars_5s_df = aggregator.get_bars_5s_df()
            bars_1m_df = aggregator.get_bars_1m_df()

            try:
                feats = pipeline.compute(latest_orderbook, bars_5s_df, bars_1m_df)
                feats["timestamp"] = datetime.now(timezone.utc).isoformat()
                feats["btc_price"] = trade["price"]
                feature_rows.append(feats)
            except Exception as e:
                log.debug("Feature compute error: %s", e)

        # Periodic status
        if bar_5s_count % 12 == 0:  # Every minute
            non_nan = sum(
                1 for v in (feature_rows[-1] if feature_rows else {}).values()
                if not (isinstance(v, float) and np.isnan(v))
            )
            log.info(
                "Status: %d trades, %d 5s bars, %d 1m bars, %d features (%d non-NaN), ob=%d/%d",
                trade_count, bar_5s_count, bar_1m_count,
                len(feature_rows), non_nan,
                len(latest_orderbook.get("bids", [])),
                len(latest_orderbook.get("asks", [])),
            )

    # Periodic save
    global last_save_time
    if time.time() - last_save_time > SAVE_INTERVAL:
        save_data()
        last_save_time = time.time()


async def on_depth(orderbook: dict) -> None:
    global latest_orderbook, depth_count
    latest_orderbook = orderbook
    depth_count += 1
    pipeline.update_orderbook(orderbook)


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

    # Save features
    if feature_rows:
        feat_df = pd.DataFrame(feature_rows)
        path = data_dir / "features" / f"{today}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            feat_df = pd.concat([existing, feat_df]).drop_duplicates(
                subset=["timestamp"]
            ).sort_values("timestamp")
        feat_df.to_parquet(path, index=False)
        log.info("Saved %d feature rows to %s", len(feat_df), path)


async def main() -> None:
    log.info("=" * 50)
    log.info("LIVE DATA COLLECTOR - Coinbase BTC-USD")
    log.info("Saving to: %s", settings.DATA_DIR)
    log.info("=" * 50)

    collector = CoinbaseCollector(
        symbol="BTC-USD",
        on_trade=on_trade,
        on_depth=on_depth,
        on_kline=on_kline,
    )

    try:
        await collector.start()
    except KeyboardInterrupt:
        pass
    finally:
        await collector.stop()
        log.info("Saving final data...")
        save_data()
        log.info(
            "Final: %d trades, %d 5s bars, %d 1m bars, %d features, %d ob updates",
            trade_count, bar_5s_count, bar_1m_count, len(feature_rows), depth_count,
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Interrupted - saving...")
        save_data()

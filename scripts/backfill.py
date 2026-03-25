"""Download historical Binance data and compute features + labels for training."""

import logging
import sys
import time
from datetime import datetime, timezone, timedelta

sys.path.insert(0, ".")

import pandas as pd

from config.settings import settings
from data.binance_rest import BinanceREST
from data.storage import DataStorage
from features.pipeline import FeaturePipeline
from features.labeler import Labeler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backfill")

BACKFILL_DAYS = 14
KLINE_LIMIT = 1000  # Binance max per request


def fetch_all_klines(
    rest: BinanceREST,
    days: int = BACKFILL_DAYS,
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
) -> pd.DataFrame:
    """Paginate through Binance klines to fetch `days` of 1m data."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    since_ms = int(start.timestamp() * 1000)

    all_dfs: list[pd.DataFrame] = []
    fetched = 0

    while True:
        logger.info(
            "Fetching klines since %s (%d fetched so far)",
            datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).isoformat(),
            fetched,
        )
        df = rest.fetch_klines(
            symbol=symbol, timeframe=timeframe,
            since=since_ms, limit=KLINE_LIMIT,
        )

        if df.empty:
            break

        all_dfs.append(df)
        fetched += len(df)

        # Move since to after the last candle
        last_ts = df["timestamp"].iloc[-1]
        if isinstance(last_ts, pd.Timestamp):
            since_ms = int(last_ts.timestamp() * 1000) + 60_000  # +1 minute
        else:
            since_ms = int(last_ts) + 60_000

        if len(df) < KLINE_LIMIT:
            break  # No more data

        time.sleep(0.5)  # respect rate limits

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    logger.info("Fetched %d total 1m klines over %d days", len(result), days)
    return result


def compute_features_on_bars(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute features over a rolling window of 1m bars."""
    pipeline = FeaturePipeline()

    # Build a dummy orderbook (no live data for backfill)
    dummy_orderbook = {
        "bids": [{"price": 0.0, "qty": 0.0}] * 20,
        "asks": [{"price": 0.0, "qty": 0.0}] * 20,
    }

    feature_rows: list[dict] = []
    window = 60  # need 60 bars of history for technical features

    for i in range(window, len(bars_1m)):
        window_df = bars_1m.iloc[i - window : i + 1].copy()

        try:
            feats = pipeline.compute(dummy_orderbook, window_df, window_df)
            feats["timestamp"] = bars_1m["timestamp"].iloc[i]
            feature_rows.append(feats)
        except Exception:
            continue

    if not feature_rows:
        return pd.DataFrame()

    features_df = pd.DataFrame(feature_rows)
    logger.info("Computed features for %d rows", len(features_df))
    return features_df


def main() -> None:
    logger.info("=== Backfill starting (%d days) ===", BACKFILL_DAYS)

    rest = BinanceREST()
    storage = DataStorage()

    # 1. Fetch historical 1m klines
    bars_1m = fetch_all_klines(rest, days=BACKFILL_DAYS)
    if bars_1m.empty:
        logger.error("No klines fetched, aborting")
        return

    # Save raw bars
    storage.save_bars(bars_1m, "bars_1m")
    logger.info("Saved %d 1m bars to storage", len(bars_1m))

    # 2. Compute features
    features_df = compute_features_on_bars(bars_1m)
    if features_df.empty:
        logger.error("No features computed, aborting")
        return

    # 3. Label the data
    labeler = Labeler(horizon_minutes=15, threshold_bps=3.0)
    labels = labeler.label_direction(bars_1m)

    # Align labels with feature rows (features start at row `window`)
    window = 60
    aligned_labels = labels.iloc[window:].reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)

    # Trim to matching length
    min_len = min(len(features_df), len(aligned_labels))
    features_df = features_df.iloc[:min_len]
    aligned_labels = aligned_labels.iloc[:min_len]

    features_df["label"] = aligned_labels.values

    # Drop rows with NaN labels (last horizon rows)
    features_df = features_df.dropna(subset=["label"]).reset_index(drop=True)
    features_df["label"] = features_df["label"].astype(int)

    # 4. Save feature matrix
    storage.save_features(features_df)

    logger.info(
        "=== Backfill complete === %d feature rows, label distribution:\n%s",
        len(features_df),
        features_df["label"].value_counts().to_string(),
    )


if __name__ == "__main__":
    main()

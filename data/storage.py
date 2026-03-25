"""
Parquet-based storage for bars and feature matrices.

Handles append-mode writes with deduplication and date-ranged reads.
Uses pyarrow via pandas for efficient columnar storage.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


class DataStorage:
    """Persistent Parquet storage organized by date and data type."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir or settings.DATA_DIR)

        # Create directory structure
        self.dirs = {
            "bars_5s": self.data_dir / "bars_5s",
            "bars_1m": self.data_dir / "bars_1m",
            "features": self.data_dir / "features",
            "logs": self.data_dir / "logs",
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        logger.info("DataStorage initialized at %s", self.data_dir)

    @staticmethod
    def _today_str() -> str:
        """Return today's date as YYYYMMDD in UTC."""
        return datetime.now(timezone.utc).strftime("%Y%m%d")

    def save_bars(
        self,
        bars_df: pd.DataFrame,
        timeframe: str,
        date: Optional[str] = None,
    ) -> None:
        """
        Save bars DataFrame to Parquet with append-mode deduplication.

        Args:
            bars_df: DataFrame containing bar data (must have a 'timestamp' column).
            timeframe: One of 'bars_5s' or 'bars_1m'.
            date: Date string YYYYMMDD; defaults to today (UTC).
        """
        if bars_df.empty:
            logger.debug("Empty DataFrame, skipping save for %s", timeframe)
            return

        date = date or self._today_str()
        dir_path = self.dirs.get(timeframe)
        if dir_path is None:
            logger.error("Unknown timeframe: %s", timeframe)
            return

        file_path = dir_path / f"{date}.parquet"

        try:
            if file_path.exists():
                existing = pd.read_parquet(file_path)
                combined = pd.concat([existing, bars_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
                combined = combined.sort_values("timestamp").reset_index(drop=True)
            else:
                combined = bars_df.copy()

            combined.to_parquet(file_path, engine="pyarrow", index=False)
            logger.info(
                "Saved %d bars to %s (%d total in file)",
                len(bars_df),
                file_path,
                len(combined),
            )
        except Exception:
            logger.exception("Failed to save bars to %s", file_path)

    def load_bars(
        self,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load bars from Parquet files within a date range.

        Args:
            timeframe: One of 'bars_5s' or 'bars_1m'.
            start_date: Start date YYYYMMDD (inclusive). Defaults to all available.
            end_date: End date YYYYMMDD (inclusive). Defaults to all available.

        Returns:
            Concatenated DataFrame of all matching files sorted by timestamp.
        """
        dir_path = self.dirs.get(timeframe)
        if dir_path is None:
            logger.error("Unknown timeframe: %s", timeframe)
            return pd.DataFrame()

        parquet_files = sorted(dir_path.glob("*.parquet"))
        if not parquet_files:
            logger.info("No parquet files found in %s", dir_path)
            return pd.DataFrame()

        selected = []
        for fp in parquet_files:
            file_date = fp.stem  # e.g. "20260324"
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue
            selected.append(fp)

        if not selected:
            logger.info("No files match date range %s to %s in %s", start_date, end_date, dir_path)
            return pd.DataFrame()

        dfs = []
        for fp in selected:
            try:
                dfs.append(pd.read_parquet(fp))
            except Exception:
                logger.exception("Failed to read %s, skipping", fp)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        result = result.sort_values("timestamp").reset_index(drop=True)
        logger.info("Loaded %d bars from %d files (%s)", len(result), len(selected), timeframe)
        return result

    def save_features(
        self,
        features_df: pd.DataFrame,
        date: Optional[str] = None,
    ) -> None:
        """
        Save a feature matrix to Parquet with append-mode deduplication.

        Args:
            features_df: DataFrame of computed features (must have a 'timestamp' column).
            date: Date string YYYYMMDD; defaults to today (UTC).
        """
        if features_df.empty:
            logger.debug("Empty features DataFrame, skipping save")
            return

        date = date or self._today_str()
        file_path = self.dirs["features"] / f"{date}.parquet"

        try:
            if file_path.exists():
                existing = pd.read_parquet(file_path)
                combined = pd.concat([existing, features_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
                combined = combined.sort_values("timestamp").reset_index(drop=True)
            else:
                combined = features_df.copy()

            combined.to_parquet(file_path, engine="pyarrow", index=False)
            logger.info(
                "Saved %d feature rows to %s (%d total in file)",
                len(features_df),
                file_path,
                len(combined),
            )
        except Exception:
            logger.exception("Failed to save features to %s", file_path)

    def load_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load feature matrices from Parquet files within a date range.

        Args:
            start_date: Start date YYYYMMDD (inclusive).
            end_date: End date YYYYMMDD (inclusive).

        Returns:
            Concatenated DataFrame of all matching feature files sorted by timestamp.
        """
        features_dir = self.dirs["features"]
        parquet_files = sorted(features_dir.glob("*.parquet"))

        if not parquet_files:
            logger.info("No feature files found in %s", features_dir)
            return pd.DataFrame()

        selected = []
        for fp in parquet_files:
            file_date = fp.stem
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue
            selected.append(fp)

        if not selected:
            logger.info("No feature files match date range %s to %s", start_date, end_date)
            return pd.DataFrame()

        dfs = []
        for fp in selected:
            try:
                dfs.append(pd.read_parquet(fp))
            except Exception:
                logger.exception("Failed to read %s, skipping", fp)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        result = result.sort_values("timestamp").reset_index(drop=True)
        logger.info("Loaded %d feature rows from %d files", len(result), len(selected))
        return result

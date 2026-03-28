"""Reader for archived Kalshi market data.

Loads parquet files from D:/kalshi-data/market_archive/YYYY-MM-DD/{series}.parquet
for backtesting and calibration.
"""

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

ARCHIVE_DIR = Path(settings.DATA_DIR) / "market_archive"


def load_archive(
    series: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Load archived market data for a series across date range.

    Args:
        series: Series name (e.g., "KXBTC15M", "KXHIGH")
        start_date: Earliest date to include (default: 7 days ago)
        end_date: Latest date to include (default: today)

    Returns:
        DataFrame with all archived snapshots, sorted by timestamp.
    """
    if start_date is None:
        start_date = date.today() - timedelta(days=7)
    if end_date is None:
        end_date = date.today()

    frames = []
    d = start_date
    while d <= end_date:
        path = ARCHIVE_DIR / d.isoformat() / f"{series}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                frames.append(df)
            except Exception:
                logger.warning("Failed to read %s", path)
        d += timedelta(days=1)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    if "timestamp" in result.columns:
        result = result.sort_values("timestamp").reset_index(drop=True)
    return result


def get_archive_status() -> dict:
    """Get archive health info for the dashboard."""
    if not ARCHIVE_DIR.exists():
        return {"exists": False, "days": 0, "total_files": 0}

    day_dirs = sorted([d for d in ARCHIVE_DIR.iterdir() if d.is_dir()])
    total_files = sum(len(list(d.glob("*.parquet"))) for d in day_dirs)
    total_records = 0

    # Check most recent day
    latest_day = day_dirs[-1] if day_dirs else None
    latest_file_time = None
    if latest_day:
        parquets = list(latest_day.glob("*.parquet"))
        if parquets:
            latest_file_time = max(p.stat().st_mtime for p in parquets)
            # Count records in latest day
            for p in parquets:
                try:
                    total_records += len(pd.read_parquet(p))
                except Exception:
                    pass

    return {
        "exists": True,
        "days": len(day_dirs),
        "total_files": total_files,
        "latest_day": latest_day.name if latest_day else None,
        "latest_records": total_records,
        "latest_file_age_s": int(
            __import__("time").time() - latest_file_time
        ) if latest_file_time else None,
    }

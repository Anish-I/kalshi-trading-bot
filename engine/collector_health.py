"""Check if the data collector is producing fresh bars."""
import time
from pathlib import Path

def check_collector_freshness(data_dir: str = "D:/kalshi-data", stale_seconds: int = 120) -> dict:
    """Returns dict with freshness info."""
    bars_dir = Path(data_dir) / "bars_1m"
    if not bars_dir.exists():
        return {"healthy": False, "reason": "bars_1m dir missing", "age_seconds": None}

    files = sorted(bars_dir.glob("*.parquet"))
    if not files:
        return {"healthy": False, "reason": "no parquet files", "age_seconds": None}

    latest = files[-1]
    age = time.time() - latest.stat().st_mtime
    healthy = age < stale_seconds

    return {
        "healthy": healthy,
        "age_seconds": int(age),
        "file": latest.name,
        "reason": "ok" if healthy else f"stale ({int(age)}s > {stale_seconds}s)",
    }
